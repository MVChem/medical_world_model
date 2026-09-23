"""Read-only medication pairs with a compact in-memory index and original records.

The prepared selection remains immutable. No clinical payloads or index caches are
written to disk. Patient medication records are loaded through CSV byte offsets,
rechecked with the selection predicates, and kept in a bounded, expiring cache.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import re
from threading import Event, RLock, Thread
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.json as paj

from data_preprocessing import medication_events
from data_preprocessing.medication_events import emar_event, event_overlaps, input_event, parse_timestamp
from .data import DATA_ROOT
from .memory import deep_size
from .patient_index import PatientIndex, fingerprint

SPLITS = ("train", "validate", "test", "unassigned")
TABLES = ("hosp.emar", "hosp.emar_detail", "icu.inputevents", "icu.d_items")
COUNTS = ("records", "emar", "inputevents", "distinct_recorded_medication_names",
          "repeated_recorded_name_rows", "maximum_rows_for_one_recorded_name",
          "continuous_segments", "continuous_active_at_source", "continuous_crosses_target",
          "records_with_positive_numeric_dose", "emar_records_with_detail")
STRINGS = ("id", "patient", "split", "source", "target", "source_study_id", "target_study_id",
           "source_time", "target_time", "source_view", "target_view", "source_image", "target_image",
           "source_report", "target_report", "evidence_patient")
SCHEMA = pa.schema([(key, pa.string()) for key in STRINGS] + [
    ("hours", pa.float64()), ("full_timeline_adjacent", pa.bool_()),
    ("medication", pa.struct([(key, pa.int32()) for key in COUNTS]))])
HIDDEN = ("source_image", "target_image", "source_report", "target_report", "evidence_patient")


class MedicationCohort:
    def __init__(self, root=None, *, expected_cxr_root=None, patient_cache_count=2, patient_cache_bytes=128 * 2**20,
                 cache_idle_seconds=300):
        self.root = Path(root or DATA_ROOT / "medworld_0923")
        self.expected_cxr_root = Path(expected_cxr_root).resolve() if expected_cxr_root is not None else None
        self.lock, self.records_lock = RLock(), RLock()
        self.stop = Event()
        self.state, self.error = "idle", ""
        self.manifest, self.summary = {}, {}
        self.table = None
        self.index, self.items = None, {}
        self.sources, self.snapshots = {}, {}
        self.cache = OrderedDict()
        self.cache_count, self.cache_bytes, self.ttl = patient_cache_count, patient_cache_bytes, cache_idle_seconds
        self.loader, self.janitor = None, None

    def bind_cxr_root(self, root):
        """Keep report provenance and the existing image endpoint on one source."""
        self.expected_cxr_root = Path(root).resolve()
        if hasattr(self, "cxr_root") and self.cxr_root != self.expected_cxr_root:
            self._fail(ValueError("Medication selection CXR root differs from the Atlas image source root"))

    def status(self):
        with self.lock:
            if self.state == "idle":
                self.state = "loading"
                self.loader = Thread(target=self._load, daemon=True, name="medication-cohort")
                self.loader.start()
                self.janitor = Thread(target=self._maintain, daemon=True, name="medication-memory")
                self.janitor.start()
            return {"state": self.state, "error": self.error, **self.summary}

    def _maintain(self):
        while not self.stop.wait(min(10, self.ttl / 2)):
            with self.records_lock:
                self._sweep()

    def _sweep(self):
        now = time.monotonic()
        for key, entry in list(self.cache.items()):
            if now - entry[0] >= self.ttl:
                del self.cache[key]

    def close(self):
        self.stop.set()
        if self.loader:
            self.loader.join(timeout=2)
        if self.janitor:
            self.janitor.join(timeout=2)
        with self.records_lock:
            self.cache.clear()

    def _fail(self, exc):
        with self.lock:
            self.error, self.state = f"Medication selection unavailable: {exc}", "error"
            self.table = None
        with self.records_lock:
            self.cache.clear()

    def _validate_sources(self):
        if self.expected_cxr_root is not None and self.cxr_root != self.expected_cxr_root:
            raise ValueError("Medication selection CXR root differs from the Atlas image source root")
        for label, expected in self.sources.items():
            if fingerprint(expected["path"]) != expected:
                raise ValueError(f"Source changed since selection: {label}; rebuild the medication selection")
        for path, expected in self.snapshots.items():
            if fingerprint(path) != expected:
                raise ValueError(f"Prepared selection changed while open: {Path(path).name}; restart the catalog")

    def _load(self):
        try:
            manifest_path = self.root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("schema") != "medworld-medication-selection-v1" or manifest.get("state") != "complete":
                raise ValueError("A complete medication selection v1 is required")
            if manifest.get("dry_run"):
                raise ValueError("Dry-run output cannot be used as a medication cohort")
            self.manifest = manifest
            self.snapshots[str(manifest_path)] = fingerprint(manifest_path)
            expected_code = manifest["implementation_sha256"]["medication_events.py"]
            if hashlib.sha256(Path(medication_events.__file__).read_bytes()).hexdigest() != expected_code:
                raise ValueError("Administration predicates changed since selection; rebuild the selection")
            self.cxr_root = Path(manifest["sources"]["cxr_root"]).resolve()
            self.sources = {**manifest["sources"]["clinical_tables"], **manifest["sources"]["cxr_tables"]}
            self._validate_sources()
            self.index = PatientIndex(manifest["sources"]["clinical_index"], cxr_root=self.cxr_root, validate=False)
            for name in TABLES:
                if self.index.table(name)["source"] != self.sources[name]:
                    raise ValueError(f"Clinical index disagrees with selection source: {name}")
            self.items = {row["itemid"]: row for row in self.index.iter_table("icu.d_items")}
            batches, loaded = [], 0
            for split in SPLITS:
                path = self.root / f"{split}.jsonl.gz"
                expected = manifest["outputs"][path.name]
                if not path.resolve().is_relative_to(self.root.resolve()):
                    raise ValueError("Selection shard escapes the dataset root")
                self.snapshots[str(path)] = fingerprint(path)
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(2**20), b""):
                        digest.update(block)
                if path.stat().st_size != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
                    raise ValueError(f"Selection shard checksum mismatch: {path.name}")
                count = 0
                if manifest["cohort"]["splits"][split].get("retained", 0):
                    with gzip.open(path, "rb") as stream:
                        reader = paj.open_json(stream, read_options=paj.ReadOptions(block_size=8 * 2**20),
                                               parse_options=paj.ParseOptions(explicit_schema=SCHEMA,
                                                                             unexpected_field_behavior="ignore"))
                        for batch in reader:
                            if self.stop.is_set():
                                return
                            table = pa.Table.from_batches([batch])
                            for key in SCHEMA.names:
                                if table[key].null_count:
                                    raise ValueError(f"Missing required pair field: {key}")
                            if (not pc.all(pc.equal(table["patient"], table["evidence_patient"])).as_py()
                                    or not pc.all(pc.equal(table["split"], split)).as_py()
                                    or not pc.all(pc.greater(table["hours"], 0)).as_py()):
                                raise ValueError("Invalid patient ownership, split, or chronology in selection")
                            # Repeated image/report identifiers are dictionary encoded within chunks.
                            for key in STRINGS:
                                if key != "id":
                                    table = table.set_column(table.schema.get_field_index(key), key,
                                                             pc.dictionary_encode(table[key]))
                            batches.append(table)
                            loaded += len(table)
                            count += len(table)
                            with self.lock:
                                self.summary["loaded_pairs"] = loaded
                if count != manifest["cohort"]["splits"][split].get("retained", 0):
                    raise ValueError(f"Selection count mismatch: {split}")
            table = pa.concat_tables(batches) if batches else pa.Table.from_batches([], schema=SCHEMA)
            if len(table) != manifest["cohort"]["retained_pairs"]:
                raise ValueError("Selection total count mismatch")
            if len(pc.unique(table["id"])) != len(table):
                raise ValueError("Duplicate pair identifiers in selection")
            self._validate_sources()
            with self.lock:
                if self.stop.is_set():
                    return
                self.table = table
                self.summary.update(pairs=len(table), patients=manifest["cohort"]["retained_patients"],
                                    splits=manifest["cohort"]["splits"], cohort=manifest["cohort"],
                                    rules=manifest["rules"], created_at=manifest["created_at"], index_bytes=table.nbytes)
                self.state = "ready"
        except Exception as exc:
            self._fail(exc)

    def require_ready(self):
        state = self.status()
        if state["state"] != "ready":
            raise RuntimeError(state.get("error") or "Medication selection is loading")
        try:
            self._validate_sources()
        except (OSError, ValueError) as exc:
            self._fail(exc)
            raise RuntimeError(self.error) from exc

    @staticmethod
    def _public(row):
        return {key: value for key, value in row.items() if key not in HIDDEN}

    def search(self, *, q="", subject_id="", split="all", page=1, limit=25):
        self.require_ready()
        table = self.table
        mask = None
        if split != "all":
            mask = pc.equal(table["split"], split)
        if subject_id:
            current = pc.equal(table["patient"], subject_id)
            mask = current if mask is None else pc.and_(mask, current)
        if q.strip():
            current = None
            for key in ("id", "patient", "source_study_id", "target_study_id"):
                match = pc.match_substring(pc.cast(table[key], pa.string()), q.strip(), ignore_case=True)
                current = match if current is None else pc.or_(current, match)
            mask = current if mask is None else pc.and_(mask, current)
        if mask is not None:
            # Gather only the requested page, never copy a million full pair rows.
            positions = pc.indices_nonzero(mask)
            total = len(positions)
            selected = table.take(positions.slice((page - 1) * limit, limit))
        else:
            total = len(table)
            selected = table.slice((page - 1) * limit, limit)
        return {"rows": [self._public(row) for row in selected.to_pylist()], "total": total,
                "page": page, "limit": limit}

    def _pair(self, identifier):
        self.require_ready()
        positions = pc.indices_nonzero(pc.equal(self.table["id"], identifier))
        if len(positions) != 1:
            raise KeyError(identifier)
        row = self.table.slice(positions[0].as_py(), 1).to_pylist()[0]
        if row["patient"] != row["evidence_patient"]:
            raise ValueError("Pair evidence patient ownership mismatch")
        lower, upper = parse_timestamp(row["source_time"]), parse_timestamp(row["target_time"])
        if lower is None or upper is None or upper <= lower:
            raise ValueError("Pair timestamps are not strictly chronological")
        return row

    def _endpoint(self, row, side):
        subject, study = row["patient"], row[f"{side}_study_id"]
        image_id = row[side].removeprefix("cxr:")
        if (not subject.isdigit() or not study.isdigit()
                or not re.fullmatch(r"[A-Za-z0-9-]+", image_id)):
            raise ValueError("Invalid CXR identifiers in selected pair")
        parent = Path("files") / f"p{subject[:2]}" / f"p{subject}"
        report = parent / f"s{study}.txt"
        image = parent / f"s{study}" / f"{image_id}.jpg"
        if row[f"{side}_report"] != str(report) or row[f"{side}_image"] != str(image):
            raise ValueError("Pair image or report does not belong to its patient and study")
        for relative in (report, image):
            path = (self.cxr_root / relative).resolve()
            if not path.is_relative_to(self.cxr_root) or not path.is_file():
                raise ValueError("Selected CXR image or report is unavailable within the source root")
        text = (self.cxr_root / report).read_text(encoding="utf-8")
        if not text:
            raise ValueError("Selected endpoint report is empty")
        return {"dicom_id": image_id, "study_id": study, "time": row[f"{side}_time"],
                "view": row[f"{side}_view"], "image_url": f"/api/images/{image_id}?size=1024",
                "report": text, "report_status": "present"}

    def detail(self, identifier):
        row = self._pair(identifier)
        return {**self._public(row), "endpoints": {side: self._endpoint(row, side) for side in ("source", "target")}}

    def _patient_events(self, subject):
        self._sweep()
        if subject in self.cache:
            _, size, events = self.cache.pop(subject)
            self.cache[subject] = (time.monotonic(), size, events)
            return events
        details = defaultdict(list)
        for ordinal, row in enumerate(self.index.read_subject("hosp.emar_detail", subject)):
            if row["subject_id"] != subject:
                raise ValueError("Clinical detail patient ownership mismatch")
            details[row["emar_id"]].append((ordinal, row))
        events = []
        for table in ("hosp.emar", "icu.inputevents"):
            for ordinal, row in enumerate(self.index.read_subject(table, subject)):
                if row["subject_id"] != subject:
                    raise ValueError("Clinical record patient ownership mismatch")
                children = details.get(row.get("emar_id"), []) if table == "hosp.emar" else []
                event, _ = (emar_event(row, (child for _, child in children)) if table == "hosp.emar"
                            else input_event(row, self.items.get(row.get("itemid"))))
                if event is None:
                    continue
                event.update(source_reference={"table": table, "patient_row_ordinal": ordinal},
                             raw_record=row, raw_details=[child for _, child in children],
                             detail_references=[{"table": "hosp.emar_detail", "patient_row_ordinal": position}
                                                for position, _ in children])
                events.append(event)
        events.sort(key=lambda event: (event["start"], event["end"], event["source_table"],
                                       event["source_reference"]["patient_row_ordinal"]))
        size = deep_size(events)
        if size <= self.cache_bytes and self.cache_count > 0:
            while self.cache and (len(self.cache) >= self.cache_count
                                  or sum(entry[1] for entry in self.cache.values()) + size > self.cache_bytes):
                self.cache.popitem(last=False)
            self.cache[subject] = (time.monotonic(), size, events)
        return events

    def medications(self, identifier, *, source="all", q="", page=1, limit=25):
        pair = self._pair(identifier)
        lower = parse_timestamp(pair["source_time"])
        with self.records_lock:
            events = self._patient_events(pair["patient"])
            matches = [event for event in events if event_overlaps(event, pair["patient"], lower, pair["target_time"])]
            query = q.strip().casefold()
            selected = [event for event in matches if (source == "all" or event["source_table"] == source)
                        and (not query or query in event["name"].casefold())]
            rows = []
            for event in selected[(page - 1) * limit: page * limit]:
                row = {key: value.isoformat() if isinstance(value, datetime) else value for key, value in event.items()}
                row.update(relative_start_hours=(event["start"] - lower).total_seconds() / 3600,
                           relative_end_hours=(event["end"] - lower).total_seconds() / 3600)
                rows.append(row)
            actual_counts = {"records": len(matches), "emar": sum(e["source_table"] == "hosp.emar" for e in matches),
                             "inputevents": sum(e["source_table"] == "icu.inputevents" for e in matches)}
            counts_match = all(pair["medication"][key] == value for key, value in actual_counts.items())
            if not counts_match:
                raise ValueError("Recomputed medication counts disagree with the prepared pair; rebuild the selection")
            return {"rows": rows, "total": len(selected), "unfiltered_total": len(matches), "page": page,
                    "limit": limit, "expected_records": pair["medication"]["records"], "counts_match": counts_match,
                    "source_time": pair["source_time"], "target_time": pair["target_time"],
                    "window": "Exact inclusive CXR interval; same patient; no admission restriction or time padding"}
