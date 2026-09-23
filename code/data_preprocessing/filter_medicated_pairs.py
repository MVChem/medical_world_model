"""Select all forward same-patient CXR pairs containing observed medication.

Admission, view matching, labels, and minimum/maximum gap are not requirements.
Outputs contain image/report references and medication evidence references only.
"""
from __future__ import annotations

import argparse
from array import array
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import gzip
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import math
from pathlib import Path
import shlex
import shutil
import sys
import time

from mimic_atlas import build_mimic_transitions as cxr
from mimic_atlas.patient_index import PatientIndex, fingerprint, validate_source
from .medication_events import emar_event, input_event

LOG = logging.getLogger(__name__)
TABLES = ("hosp.emar", "hosp.emar_detail", "icu.inputevents", "icu.d_items")
SPLITS = ("train", "validate", "test", "unassigned")
SCHEMA = "medworld-medication-selection-v1"
_INDEX = None
_ITEMS = None


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def stamp(value):
    return value.isoformat()


def load_patient_studies(index):
    """One PA-preferred frontal observation per study, own nonempty report."""
    for name, meta in index.tables.items():
        if name.startswith("cxr."):
            validate_source(meta)
    root = Path(index.manifest["cxr_root"])
    audit = cxr.Audit()
    studies = cxr.load_studies(root, "frontal", audit)
    cxr.attach_splits(root, studies, audit, strict=False)
    iv_subjects = set(index.iv_subjects())
    groups = defaultdict(list)
    counts = Counter(studies_loaded=len(studies), iv_patients=len(iv_subjects))
    original_order = defaultdict(list)
    for study in studies.values():
        original_order[study.subject_id].append(study)
    for patient, timeline in sorted(original_order.items()):
        if patient not in iv_subjects:
            counts["studies_without_iv_patient"] += len(timeline)
            continue
        timeline.sort(key=lambda study: (study.timestamp, study.study_id))
        known_splits = {study.split for study in timeline if study.split in SPLITS[:3]}
        # Preserve patient isolation if the official image split is inconsistent.
        split = max(known_splits, key=SPLITS.index) if known_splits else "unassigned"
        if len(known_splits) > 1:
            counts["patients_official_split_conflict_assigned_strictest"] += 1
        for order, study in enumerate(timeline):
            view = next((v for v in ("PA", "AP") if v in study.images_by_view), None)
            if view is None:
                counts["studies_without_frontal"] += 1
                continue
            selected = study.images_by_view[view]
            if not selected.image_path.is_file():
                counts["studies_missing_selected_image"] += 1
                continue
            if not study.report_path.is_file() or study.report_path.stat().st_size <= 0:
                counts["studies_missing_or_empty_report"] += 1
                continue
            groups[patient].append({"patient": patient, "study_id": study.study_id,
                "id": "cxr:" + selected.dicom_id, "time": selected.timestamp,
                "image": selected.relative_path, "report": study.report_relative_path,
                "view": view, "split": split, "original_order": order})
            counts["usable_studies"] += 1
        if len(original_order) and len(groups) % 5000 == 0:
            LOG.info("Validated actual image/report paths for %d patients", len(groups))
    for observations in groups.values():
        observations.sort(key=lambda row: (row["time"], row["study_id"]))
    eligible = {patient: rows for patient, rows in groups.items() if len(rows) >= 2}
    counts.update(usable_patients=len(groups), repeated_patients=len(eligible))
    return eligible, dict(counts)


def init_worker(index_root):
    global _INDEX, _ITEMS
    _INDEX = PatientIndex(index_root, validate=False)
    for name in TABLES:
        validate_source(_INDEX.table(name))
    _ITEMS = {row["itemid"]: row for row in _INDEX.iter_table("icu.d_items")}


class IntervalIndex:
    """O(log n) exact inclusive overlap counts, including point records."""
    def __init__(self, events):
        self.starts = sorted(event["start"] for event in events)
        self.ends = sorted(event["end"] for event in events)

    def count(self, lower, upper):
        if upper <= lower:
            raise ValueError("A CXR pair must have a strictly positive interval")
        return bisect_right(self.starts, upper) - bisect_left(self.ends, lower)

    def crosses(self, point):
        return bisect_left(self.starts, point) - bisect_right(self.ends, point)

    def active_at(self, point):
        return bisect_right(self.starts, point) - bisect_right(self.ends, point)


def dose_known(event):
    if event["source_table"] == "icu.inputevents":
        return True
    for detail in event.get("dose_details", ()):
        try:
            amount = float(detail.get("dose_given"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(amount) and amount > 0:
            return True
    return False


class EventIndexes:
    def __init__(self, events):
        self.all = IntervalIndex(events)
        self.emar = IntervalIndex([e for e in events if e["source_table"] == "hosp.emar"])
        self.icu = IntervalIndex([e for e in events if e["source_table"] == "icu.inputevents"])
        self.continuous = IntervalIndex([e for e in events if e["event_type"] == "infusion_segment"])
        self.quantified = IntervalIndex([e for e in events if dose_known(e)])
        self.details = IntervalIndex([e for e in events if e.get("dose_details")])
        names = defaultdict(list)
        for event in events:
            names[event["name"].strip().casefold()].append(event)
        self.names = [IntervalIndex(group) for group in names.values()]

    def query(self, lower, upper):
        records = self.all.count(lower, upper)
        if not records:
            return None
        named_counts = [index.count(lower, upper) for index in self.names]
        distinct = sum(n > 0 for n in named_counts)
        return {"records": records, "emar": self.emar.count(lower, upper),
                "inputevents": self.icu.count(lower, upper),
                "distinct_recorded_medication_names": distinct,
                "repeated_recorded_name_rows": records - distinct,
                "maximum_rows_for_one_recorded_name": max(named_counts),
                "continuous_segments": self.continuous.count(lower, upper),
                "continuous_active_at_source": self.continuous.active_at(lower),
                "continuous_crosses_target": self.continuous.crosses(upper),
                "records_with_positive_numeric_dose": self.quantified.count(lower, upper),
                "emar_records_with_detail": self.details.count(lower, upper)}


def summarize_histogram(counts):
    if not counts:
        return {key: None for key in ("median", "p90", "p95", "p99", "max", "mean")}
    values = sorted(counts)
    total = sum(counts.values())
    cumulative = []
    count = 0
    for value in values:
        count += counts[value]
        cumulative.append(count)
    def value_at(index):
        return values[bisect_right(cumulative, index)]
    result = {"max": values[-1], "mean": sum(value * counts[value] for value in values) / total}
    for key, quantile in (("median", .5), ("p90", .9), ("p95", .95), ("p99", .99)):
        index = (total - 1) * quantile
        low, high = math.floor(index), math.ceil(index)
        result[key] = value_at(low) + (value_at(high) - value_at(low)) * (index - low)
    return result


class CohortStats:
    def __init__(self):
        self.counts = Counter()
        self.splits = {split: Counter() for split in SPLITS}
        self.density = defaultdict(Counter)
        self.hours = array("d")

    def add(self, split, hours, stats, adjacent):
        self.counts["input_pairs"] += 1
        self.counts["input_full_timeline_adjacent"] += adjacent
        self.splits[split]["input"] += 1
        if stats is None:
            return
        self.counts["retained_pairs"] += 1
        self.counts["retained_full_timeline_adjacent"] += adjacent
        self.splits[split]["retained"] += 1
        self.counts["with_emar"] += bool(stats["emar"])
        self.counts["with_inputevents"] += bool(stats["inputevents"])
        self.counts["with_both"] += bool(stats["emar"] and stats["inputevents"])
        self.counts["with_positive_numeric_dose"] += bool(stats["records_with_positive_numeric_dose"])
        self.counts["with_emar_detail"] += bool(stats["emar_records_with_detail"])
        for key, value in stats.items():
            self.density[key][value] += 1
        self.hours.append(hours)

    def merge(self, other):
        self.counts.update(other.counts)
        for split, counts in other.splits.items():
            self.splits[split].update(counts)
        for key, counts in other.density.items():
            self.density[key].update(counts)
        self.hours.extend(other.hours)

    def summary(self):
        result = dict(self.counts)
        result["excluded_pairs"] = self.counts["input_pairs"] - self.counts["retained_pairs"]
        result["retained_fraction"] = self.counts["retained_pairs"] / max(1, self.counts["input_pairs"])
        result["splits"] = {key: dict(value) for key, value in self.splits.items()}
        result["density_per_retained_pair"] = {key: summarize_histogram(value) for key, value in self.density.items()}
        result["density_per_retained_pair"]["interval_hours"] = summarize_histogram(Counter(self.hours))
        return result


def compressed_json_lines(rows):
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, compresslevel=3) as output:
        for row in rows:
            output.write((json.dumps(row, separators=(",", ":")) + "\n").encode())
    return buffer.getvalue()


def process_patient(job):
    subject, observations = job
    details = defaultdict(list)
    detail_rows = _INDEX.read_subject("hosp.emar_detail", subject)
    for row in detail_rows:
        details[row["emar_id"]].append(row)
    emar_rows = _INDEX.read_subject("hosp.emar", subject)
    input_rows = _INDEX.read_subject("icu.inputevents", subject)
    audit = Counter({"patients_read": 1, "emar_source_rows": len(emar_rows),
                     "emar_detail_source_rows": len(detail_rows), "inputevents_source_rows": len(input_rows)})
    events, evidence = [], []
    for kind, rows in (("emar", emar_rows), ("inputevents", input_rows)):
        for ordinal, row in enumerate(rows):
            if str(row["subject_id"]) != subject:
                raise ValueError("Patient ownership mismatch after indexed read")
            event, reason = (emar_event(row, details.get(row["emar_id"], [])) if kind == "emar"
                             else input_event(row, _ITEMS.get(row["itemid"])))
            if event is None:
                audit[kind + ":rejected:" + reason] += 1
                continue
            events.append(event)
            reference = {"table": event["source_table"], "patient_row_ordinal": ordinal,
                         "record_id": event["record_id"], "start": stamp(event["start"]),
                         "end": stamp(event["end"]), "hadm_id": event["hadm_id"]}
            if kind == "emar":
                reference.update(emar_id=row["emar_id"], emar_seq=row["emar_seq"])
            else:
                reference.update(itemid=row["itemid"], orderid=row["orderid"],
                                 linkorderid=row["linkorderid"], stay_id=row["stay_id"])
            evidence.append(reference)
            audit[kind + ":accepted_source_rows"] += 1
    indexes = EventIndexes(events)
    stats = CohortStats()
    stats.counts["input_patients"] = 1
    def pairs():
        for i, source in enumerate(observations):
            for target in observations[i + 1:]:
                lower, upper = source["time"], target["time"]
                if upper <= lower:
                    audit["pairs_rejected_equal_selected_acquisition"] += 1
                    continue
                if source["patient"] != subject or target["patient"] != subject or source["split"] != target["split"]:
                    raise ValueError("CXR endpoint ownership/split mismatch")
                hours = (upper - lower).total_seconds() / 3600
                medication = indexes.query(lower, upper)
                adjacent = target["original_order"] == source["original_order"] + 1
                stats.add(source["split"], hours, medication, adjacent)
                if medication is None:
                    continue
                identity = subject + "\0" + source["id"] + "\0" + target["id"]
                yield {"id": "medpair:" + hashlib.sha256(identity.encode()).hexdigest()[:24],
                       "patient": subject, "split": source["split"], "source": source["id"], "target": target["id"],
                       "source_study_id": source["study_id"], "target_study_id": target["study_id"],
                       "source_time": stamp(lower), "target_time": stamp(upper), "hours": hours,
                       "source_view": source["view"], "target_view": target["view"],
                       "source_image": source["image"], "target_image": target["image"],
                       "source_report": source["report"], "target_report": target["report"],
                       "full_timeline_adjacent": adjacent, "medication": medication,
                       "evidence_patient": subject}
    pair_payload = compressed_json_lines(pairs())
    if stats.counts["retained_pairs"]:
        stats.counts["retained_patients"] = 1
    # Once per patient, no doses/text/table payload and no repeated per-pair IDs.
    evidence_payload = compressed_json_lines([{"patient": subject, "records": evidence}]) if stats.counts["retained_pairs"] else b""
    return observations[0]["split"], pair_payload, evidence_payload, stats, dict(audit)


def bounded_results(executor, jobs, pending_limit):
    iterator = iter(jobs)
    pending = deque()
    for _ in range(pending_limit):
        try:
            pending.append(executor.submit(process_patient, next(iterator)))
        except StopIteration:
            break
    while pending:
        yield pending.popleft().result()
        try:
            pending.append(executor.submit(process_patient, next(iterator)))
        except StopIteration:
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-link", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-patients", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8:
        parser.error("Use between 1 and 8 workers")
    if args.max_patients is not None and (args.max_patients < 1 or not args.dry_run):
        parser.error("--max-patients requires --dry-run and a positive limit")
    if args.output.exists() and not args.dry_run:
        parser.error("Output already exists; selection outputs are immutable")
    if args.project_link and (args.project_link.exists() or args.project_link.is_symlink()) and not args.dry_run:
        parser.error("Project link already exists")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.run_dir / "run.log")])
    start = time.monotonic()
    command = [sys.executable, "-m", "data_preprocessing.filter_medicated_pairs", *(argv or sys.argv[1:])]
    json_write(args.run_dir / "argv.json", command)
    (args.run_dir / "command.txt").write_text("PYTHONPATH=code " + shlex.join(command) + "\n")
    frozen = args.run_dir / "source"
    frozen.mkdir(exist_ok=True)
    for name in ("filter_medicated_pairs.py", "medication_events.py"):
        shutil.copy2(Path(__file__).with_name(name), frozen / name)
    index = PatientIndex(args.patient_index, validate=False)
    for name in TABLES:
        validate_source(index.table(name))
    LOG.info("Loading full CXR timeline; validating selected images and own nonempty reports")
    groups, endpoint_audit = load_patient_studies(index)
    LOG.info("Endpoint audit: %s", json.dumps(endpoint_audit))
    jobs = sorted(groups.items())
    if args.max_patients:
        jobs = jobs[:args.max_patients]
    meta = {"schema": SCHEMA, "state": "running", "created_at": datetime.now(timezone.utc).isoformat(),
            "dry_run": args.dry_run, "endpoint_audit": endpoint_audit,
            "sources": {"clinical_index": str(args.patient_index.resolve()),
                        "cxr_root": index.manifest["cxr_root"],
                        "clinical_tables": {name: index.table(name)["source"] for name in TABLES},
                        "cxr_tables": {name: table["source"] for name, table in index.tables.items() if name.startswith("cxr.")}},
            "rules": {"same_subject": True, "same_admission_required": False,
                      "minimum_gap_hours": None, "maximum_gap_hours": None,
                      "positive_chronological_acquisition_interval": True,
                      "same_view_required": False, "labels_required": False,
                      "observation": "one image per study: PA preferred, otherwise AP; largest pixel area per view, DICOM ID tie-break",
                      "reports": "each endpoint owns an existing report file with positive byte size; no shared report requirement",
                      "pairing": "all chronological combinations of usable selected study observations; no sampling",
                      "window": "source acquisition <= point administration <= target acquisition; infusion overlap inclusive",
                      "threshold": "at least one accepted observed source record",
                      "patient_split": "official CXR split; if conflicting, priority test > validate > train; no split is unassigned",
                      "evidence_ordinals": "zero-based position in PatientIndex.read_subject; exact source fingerprint required",
                      "evidence_lookup": "match evidence_patient then start <= target_time and end >= source_time; admission is metadata only",
                      "counting": "source records, not deduplicated doses or ingredients; strip/casefold recorded names only",
                      "missing_data": "no accepted record does not prove no treatment; unreadable/stale indexed tables fail"},
            "implementation_sha256": {p.name: digest(p) for p in frozen.glob("*.py")}}
    json_write(args.run_dir / "status.json", meta)
    if not args.dry_run:
        args.output.mkdir(parents=True)
        json_write(args.output / "manifest.json", meta)
    stats, audit = CohortStats(), Counter()
    with ExitStack() as stack:
        streams = {}
        if not args.dry_run:
            for split in SPLITS:
                streams[split] = stack.enter_context((args.output / f"{split}.jsonl.gz").open("wb"))
                streams[split].write(gzip.compress(b"", mtime=0))
            evidence_stream = stack.enter_context((args.output / "evidence.jsonl.gz").open("wb"))
            evidence_stream.write(gzip.compress(b"", mtime=0))
        executor = stack.enter_context(ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                                                          initargs=(str(args.patient_index),)))
        for done, (split, payload, evidence, patient_stats, patient_audit) in enumerate(bounded_results(executor, jobs, args.workers * 2), 1):
            audit.update(patient_audit)
            stats.merge(patient_stats)
            if not args.dry_run:
                streams[split].write(payload)
                evidence_stream.write(evidence)
            if done % 250 == 0 or done == len(jobs):
                progress = {"state": "running", "patients_done": done, "patients_total": len(jobs),
                            "elapsed_seconds": time.monotonic() - start, "pairs": dict(stats.counts)}
                json_write(args.run_dir / "progress.json", progress)
                LOG.info("Patients %d/%d; retained %d/%d pairs; %.1f seconds", done, len(jobs), stats.counts["retained_pairs"], stats.counts["input_pairs"], progress["elapsed_seconds"])
    for name in TABLES:
        validate_source(index.table(name))
    meta.update(state="complete", completed_at=datetime.now(timezone.utc).isoformat(),
                elapsed_seconds=time.monotonic() - start, patients_processed=len(jobs), audit=dict(audit),
                cohort=stats.summary())
    if not args.dry_run:
        meta["outputs"] = {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
                           for path in sorted(args.output.glob("*.jsonl.gz"))}
        json_write(args.output / "manifest.json", meta)
        if args.project_link:
            args.project_link.symlink_to(args.output.resolve(), target_is_directory=True)
    json_write(args.run_dir / "summary.json", meta)
    json_write(args.run_dir / "status.json", meta)
    json_write(args.run_dir / "progress.json", {"state": "complete", "patients_done": len(jobs),
                                              "patients_total": len(jobs), "pairs": dict(stats.counts)})
    LOG.info("Complete: %s", json.dumps(meta["cohort"]))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if "--run-dir" in sys.argv and not isinstance(exc, SystemExit):
            run_dir = Path(sys.argv[sys.argv.index("--run-dir") + 1])
            status_path = run_dir / "status.json"
            run_dir.mkdir(parents=True, exist_ok=True)
            previous = json.loads(status_path.read_text()) if status_path.exists() else {}
            previous.update(state="failed", failed_at=datetime.now(timezone.utc).isoformat(),
                            error_type=type(exc).__name__, error=str(exc))
            json_write(status_path, previous)
        raise
