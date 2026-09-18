"""Local MIMIC catalog and patient views backed by an optional prepared index."""

from __future__ import annotations

import logging
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from . import build_mimic_transitions as cxr
from . import link_mimic_iv_context as iv
from .clinical_tables import TABLES, ClinicalTables
from .cohort import CohortIndex
from .memory import DiscardedLoad, ImageMemory, PatientMemory, deep_size
from .patient_index import PatientIndex

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "data"
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class AtlasConfig:
    cxr_root: Path = DATA_ROOT / "MIMIC_CXR"
    iv_root: Path = DATA_ROOT / "mimic-iv-3.1"
    pairs_path: Path = ROOT / "representative_pairs_10.json"
    include_icu_inputs: bool = False
    preload_tables: tuple[str, ...] = ()
    index_root: Path | None = None
    patient_cache_count: int = 4
    patient_cache_bytes: int = 512 * 2**20
    cache_idle_seconds: float = 300
    image_cache_bytes: int = 64 * 2**20


class AtlasStore:
    """One catalog and a serialized IV scan queue shared by all HTTP clients."""

    def __init__(self, config: AtlasConfig):
        self.config = config
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mimic-data"
        )
        self.status = {"state": "loading", "stage": "读取 CXR 元数据", "error": None}
        self.studies = {}
        self.by_subject = defaultdict(list)
        self.images = {}
        self.patients = {}
        self.clinical = {}
        self.tables = {}
        self.rows = []
        self.curated = {}
        self.audit = cxr.Audit()
        self.cxr_root = config.cxr_root
        self.iv_root = config.iv_root
        self.index = (
            PatientIndex(
                config.index_root, iv_root=config.iv_root, cxr_root=config.cxr_root
            )
            if config.index_root
            else None
        )
        self.background = {}
        self.shared_dictionaries = {}
        self.memory = PatientMemory(
            self.lock,
            self._evict_patient,
            max_patients=config.patient_cache_count,
            max_bytes=config.patient_cache_bytes,
            ttl=config.cache_idle_seconds,
        )
        self.image_memory = ImageMemory(
            max_bytes=config.image_cache_bytes, ttl=config.cache_idle_seconds
        )
        self.extended = ClinicalTables(
            config.iv_root, index=self.index, memory=self.memory
        )
        self.stop = threading.Event()
        self.janitor = threading.Thread(
            target=self._maintain_memory, name="atlas-memory", daemon=True
        )
        self.auto_tables = [
            name
            for name, spec in TABLES.items()
            if self.index and spec.module + "." + name in self.index.tables
        ]
        self.include_icu_inputs = config.include_icu_inputs or bool(
            self.index and "icu.inputevents" in self.index.tables
        )
        self.cohort = CohortIndex()

    def start(self):
        self.janitor.start()
        self.executor.submit(self._initialize)

    def _maintain_memory(self):
        while not self.stop.wait(min(10, self.config.cache_idle_seconds / 2)):
            self.memory.sweep()
            self.image_memory.sweep()

    def _evict_patient(self, subject, token):
        # Invoked under the shared lock, atomically removing every patient-owned map.
        for mapping in (self.patients, self.clinical, self.tables):
            mapping.pop(subject, None)
        future = self.background.pop(subject, None)
        if future:
            future.cancel()
        self.extended.evict(subject)

    def _submit(self, subject, function, *args):
        future = self.executor.submit(function, *args)
        self.background[subject] = future

        def done(completed):
            with self.lock:
                if self.background.get(subject) is completed:
                    self.background.pop(subject, None)

        future.add_done_callback(done)

    def memory_status(self):
        return {
            "patients": self.memory.stats(),
            "images": self.image_memory.stats(),
            "scope": "Request caches only; shared whole-cohort metadata and dictionaries are excluded. One oversized patient or leased responses may temporarily exceed the patient byte budget.",
        }

    def close(self):
        self.stop.set()
        if self.janitor.is_alive():
            self.janitor.join(timeout=5)
        self.memory.clear()
        self.extended.close()
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.memory.clear()
        self.image_memory.clear()
        self.shared_dictionaries.clear()

    def _initialize(self):
        try:
            self.cxr_root = cxr.resolve_mimic_cxr_root(self.config.cxr_root)
            self.iv_root = iv.resolve_mimic_iv_root(self.config.iv_root)
            entries = cxr.load_curated_pairs(self.config.pairs_path)
            self.curated = {
                (p["subject_id"], p["source_study_id"], p["target_study_id"]): p
                for p in entries
            }
            if self.index:
                self.studies = self.index.load_studies(self.cxr_root)
            else:
                self.studies = cxr.load_studies(
                    self.cxr_root, "frontal", self.audit, all_views=True
                )
                self.status["stage"] = "读取官方 split 与 CheXpert 标签"
                cxr.attach_splits(self.cxr_root, self.studies, self.audit)
                cxr.attach_labels(self.cxr_root, self.studies, self.audit)
            for study in self.studies.values():
                self.by_subject[study.subject_id].append(study)
                for image in study.images:
                    self.images[image.dicom_id] = image
            featured = list(dict.fromkeys(p["subject_id"] for p in entries))
            for subject_id, studies in self.by_subject.items():
                studies.sort(key=lambda s: (s.timestamp, s.study_id))
                self.rows.append(
                    {
                        "subject_id": subject_id,
                        "split": studies[0].split,
                        "studies": len(studies),
                        "images": sum(len(s.images) for s in studies),
                        "first_time": studies[0].timestamp.isoformat(),
                        "last_time": studies[-1].timestamp.isoformat(),
                        "featured": subject_id in featured,
                    }
                )
            self.rows.sort(key=lambda r: r["subject_id"])
            self.featured = [
                next(r for r in self.rows if r["subject_id"] == pid)
                for pid in featured
                if pid in self.by_subject
            ]
            # Publish the catalog before the slower IV scan. CXR browsing already works.
            with self.lock:
                self.status = {"state": "ready", "stage": "就绪", "error": None}
            self.cohort.build(self)
            if self.config.preload_tables and not self.stop.is_set():
                for subject in featured[: self.config.patient_cache_count]:
                    if subject in self.by_subject:
                        self.patient(subject)
                        self.extended.request({subject}, self.config.preload_tables)
        except Exception as exc:
            LOG.exception("Catalog initialization failed")
            with self.lock:
                self.status = {
                    "state": "error",
                    "stage": "数据读取失败",
                    "error": str(exc),
                }

    def catalog(self):
        with self.lock:
            result = dict(self.status)
            result["storage"] = {
                "mode": "patient_index" if self.index else "source_scan",
                "format": self.index.manifest["storage"] if self.index else None,
                "stores_clinical_records": False if self.index else None,
                "prepared_at": self.index.manifest["created_at"]
                if self.index
                else None,
            }
            if result["state"] == "ready":
                result.update(
                    {
                        "counts": {
                            "patients": len(self.rows),
                            "studies": len(self.studies),
                            "images": len(self.images),
                            "featured_pairs": len(self.curated),
                        },
                        "featured": self.featured,
                        "cohort": self.cohort.catalog(),
                        "splits": dict(Counter(r["split"] for r in self.rows)),
                        "icu_inputs": self.include_icu_inputs,
                        "sources": {
                            "cxr": cxr.DATASET_NAME,
                            "iv": iv.MIMIC_IV_DATASET_NAME,
                        },
                    }
                )
            return result

    def search(
        self, query="", split="all", page=1, limit=30, longitudinal=False, **filters
    ):
        rows = self.cohort.rows if self.cohort.status["state"] == "ready" else self.rows
        matches = self.cohort.filter_patients(
            rows, query, split, longitudinal, **filters
        )
        offset = (page - 1) * limit
        return {
            "rows": matches[offset : offset + limit],
            "total": len(matches),
            "page": page,
            "limit": limit,
            "index_state": self.cohort.status["state"],
        }

    def _build_patient(self, subject_id):
        studies = self.by_subject.get(subject_id, [])
        cached_reports = self.index.reports(subject_id) if self.index else None

        def report_loader(study, max_chars):
            if cached_reports is None:
                return cxr.read_report(study.report_path, max_chars)
            entry = cached_reports[study.study_id]
            if entry["status"] == "missing":
                raise FileNotFoundError(study.report_relative_path)
            return cxr.extract_report_sections(entry["text"], max_chars=max_chars)

        config = cxr.BuildConfig(
            self.cxr_root,
            ROOT,
            split="all",
            view="frontal",
            num_examples=None,
            one_per_patient=False,
            render_gallery=False,
            asset_mode="none",
        )
        audit = cxr.Audit()
        candidates = cxr.find_candidates(studies, config, audit)
        transitions = []
        for candidate in candidates:
            try:
                packet = cxr.make_packet(
                    candidate,
                    20000,
                    report_loader=report_loader if self.index else None,
                )[0]
            except OSError:
                audit.increment("pairs_rejected_report_read_error")
                continue
            key = (subject_id, candidate.source.study_id, candidate.target.study_id)
            if key in self.curated:
                packet["curation"] = self.curated[key]
            transitions.append(
                {
                    "transition_id": packet["transition_id"],
                    "mimic_cxr_transition": packet,
                    "linkage_status": "loading",
                }
            )
        study_rows = []
        for study in studies:
            try:
                report = report_loader(study, 20000)
            except OSError:
                report = {}
            study_rows.append(
                {
                    "study_id": "s" + study.study_id,
                    "timestamp": study.timestamp.isoformat(),
                    "acquisition_span_seconds": study.acquisition_span_seconds,
                    "report": report,
                    "labels": cxr.label_state(study.labels) if study.labels else {},
                    "images": [
                        {
                            "dicom_id": im.dicom_id,
                            "view": im.view or "UNKNOWN",
                            "timestamp": im.timestamp.isoformat(),
                            "rows": im.rows,
                            "columns": im.columns,
                            "available": im.image_path.is_file(),
                            "url": f"/api/images/{im.dicom_id}?size=512&format=webp&quality=60",
                            "full_url": f"/api/images/{im.dicom_id}?size=1800&quality=88",
                        }
                        for im in study.images
                    ],
                }
            )
        preferred = next(
            (
                p["transition_id"]
                for p in transitions
                if p["mimic_cxr_transition"].get("curation")
            ),
            None,
        )
        self.patients[subject_id] = {
            "subject_id": subject_id,
            "split": studies[0].split if studies else None,
            "studies": study_rows,
            "transitions": transitions,
            "preferred_transition": preferred,
            "pairing_audit": audit.as_dict(),
            "admissions": [],
            "stays": [],
            "transfers": [],
        }

    def patient(self, subject_id):
        with self.lock:
            if (
                subject_id not in self.by_subject
                and subject_id not in self.cohort.by_id
            ):
                raise KeyError(subject_id)
            token = self.memory.touch(subject_id)
            if subject_id not in self.patients:
                self._build_patient(subject_id)
                self.memory.account(
                    subject_id, token, "base", deep_size(self.patients[subject_id])
                )
                self.clinical[subject_id] = {
                    "state": "loading",
                    "stage": "按患者索引读取 MIMIC-IV"
                    if self.index
                    else "等待扫描 MIMIC-IV",
                }
                self._submit(subject_id, self._load_clinical, {subject_id}, token)
                # Opening this patient prepares all of their indexed clinical tables.
                # Overview and directory requests never open patients.
                if self.index:
                    self.extended.request({subject_id}, self.auto_tables)
            return {
                **self.patients[subject_id],
                "clinical_status": dict(self.clinical[subject_id]),
                "icu_inputs": self.tables[subject_id].include_icu_inputs
                if subject_id in self.tables
                else self.include_icu_inputs,
                "auto_load_clinical": self.index is not None,
                "extended_tables": self.extended.manifest(subject_id),
            }

    def compact_patient(self, subject_id):
        data = self.patient(subject_id)
        studies = [
            {k: v for k, v in s.items() if k not in {"report", "labels"}}
            for s in data["studies"]
        ]
        transitions = []
        for row in data["transitions"]:
            packet = row["mimic_cxr_transition"]
            summary = {k: packet[k] for k in ("interval", "matched_view")}
            if packet.get("curation"):
                summary["curation"] = packet["curation"]
            for key in ("current_state", "future_state"):
                summary[key] = {
                    "study_id": packet[key]["study_id"],
                    "image": {
                        k: packet[key]["image"][k]
                        for k in ("dicom_id", "acquisition_timestamp")
                    },
                }
            transitions.append(
                {
                    "transition_id": row["transition_id"],
                    "mimic_cxr_transition": summary,
                    "linkage_status": row["linkage_status"],
                }
            )
        return {**data, "studies": studies, "transitions": transitions, "compact": True}

    def selection(self, subject_id, transition="", study=""):
        data = self.patient(subject_id)
        selected = next(
            (r for r in data["transitions"] if r["transition_id"] == transition), None
        )
        if transition and selected is None:
            raise KeyError(transition)
        ids = (
            {
                selected["mimic_cxr_transition"][key]["study_id"]
                for key in ("current_state", "future_state")
            }
            if selected
            else {study}
        )
        studies = [s for s in data["studies"] if s["study_id"] in ids]
        if study and not studies:
            raise KeyError(study)
        return {
            "studies": studies,
            "transition": selected,
            **{
                k: data[k]
                for k in (
                    "admissions",
                    "stays",
                    "transfers",
                    "clinical_status",
                    "icu_inputs",
                )
            },
        }

    def request_tables(self, subject_id, names):
        self.patient(subject_id)
        names = list(TABLES) if names == ["all"] else names
        subjects = {subject_id}
        self.extended.request(subjects, names)
        return self.extended.manifest(subject_id)

    def request_inputs(self, subject_id):
        """Upgrade one already loaded patient without scanning the base tables again."""
        with self.lock:
            patient = self.patient(subject_id)
            if (
                self.clinical[subject_id]["state"] == "ready"
                and not patient["icu_inputs"]
            ):
                self.clinical[subject_id] = {
                    "state": "loading",
                    "stage": "等待 ICU inputevents",
                }
                self._submit(
                    subject_id,
                    self._load_inputs,
                    subject_id,
                    self.memory.touch(subject_id),
                )
            return self.patient(subject_id)

    def _load_inputs(self, subject_id, token):
        try:
            with self.lock:
                if not self.memory.current(subject_id, token):
                    return
                self.clinical[subject_id] = {
                    "state": "loading",
                    "stage": "MIMIC-IV · inputevents",
                }
                original = self.tables[subject_id]
                patient = self.patients[subject_id]
            inputs = (
                self.index.read_subject("icu.inputevents", subject_id)
                if self.index
                else iv.load_subject_rows(
                    self.iv_root / "icu/inputevents.csv.gz", {subject_id}
                )
            )
            by_subject = {
                subject_id: {**original.by_subject[subject_id], "inputevents": inputs}
            }
            tables = iv.IVTables(
                by_subject,
                original.diagnosis_dictionary,
                original.procedure_dictionary,
                original.item_dictionary,
                True,
            )
            linked = iv.link_packets(
                [r["mimic_cxr_transition"] for r in patient["transitions"]],
                self.iv_root,
                include_icu_inputs=True,
                tables=tables,
            )
            size = deep_size((patient, linked, tables.by_subject[subject_id]))
            with self.lock:
                if not self.memory.current(subject_id, token):
                    return
                self.patients[subject_id] = {
                    **self.patients[subject_id],
                    "transitions": linked,
                }
                self.tables[subject_id] = tables
                self.clinical[subject_id] = {"state": "ready", "stage": "就绪"}
                self.memory.account(subject_id, token, "base", size)
        except Exception as exc:
            LOG.exception("ICU inputevents loading failed")
            with self.lock:
                if not self.memory.current(subject_id, token):
                    return
                self.clinical[subject_id] = {
                    "state": "error",
                    "stage": "ICU inputevents 读取失败",
                    "error": str(exc),
                }

    def _load_clinical(self, subjects, token):
        if not subjects:
            return

        def progress(stage):
            with self.lock:
                for pid in subjects:
                    if not self.memory.current(pid, token):
                        raise DiscardedLoad()
                    self.clinical[pid] = {"state": "loading", "stage": stage}

        try:
            tables = iv.load_iv_tables(
                self.iv_root,
                set(subjects),
                include_icu_inputs=self.include_icu_inputs,
                progress=progress,
                index=self.index,
                dictionaries=self.shared_dictionaries,
            )
            for pid in subjects:
                with self.lock:
                    if not self.memory.current(pid, token):
                        return
                    patient = self.patients[pid]
                linked = iv.link_packets(
                    [r["mimic_cxr_transition"] for r in patient["transitions"]],
                    self.iv_root,
                    include_icu_inputs=self.include_icu_inputs,
                    tables=tables,
                )
                raw = tables.by_subject[pid]
                study_rows = []
                for study in patient["studies"]:
                    timestamp = iv.parse_time(study["timestamp"])
                    matches = iv.find_common_admissions(
                        timestamp, timestamp, raw["admissions"]
                    )
                    study_rows.append(
                        {
                            **study,
                            "admission_link": {
                                "status": "unique"
                                if len(matches) == 1
                                else "ambiguous"
                                if matches
                                else "unmatched",
                                "hadm_ids": [m["hadm_id"] for m in matches],
                            },
                        }
                    )
                size = deep_size((patient, linked, study_rows, raw))
                with self.lock:
                    if not self.memory.current(pid, token):
                        return
                    self.tables[pid] = tables
                    self.patients[pid] = {
                        **patient,
                        "transitions": linked,
                        "studies": study_rows,
                        "admissions": [
                            iv._admission_record(r) for r in raw["admissions"]
                        ],
                        "stays": [
                            {**iv._stay_record(r), "hadm_id": r["hadm_id"]}
                            for r in raw["stays"]
                        ],
                        "transfers": [
                            {
                                k: r.get(k)
                                for k in (
                                    "hadm_id",
                                    "careunit",
                                    "intime",
                                    "outtime",
                                    "eventtype",
                                )
                            }
                            for r in raw["transfers"]
                        ],
                    }
                    self.clinical[pid] = {"state": "ready", "stage": "就绪"}
                    self.memory.account(pid, token, "base", size)
        except DiscardedLoad:
            pass
        except Exception as exc:
            LOG.exception("MIMIC-IV linkage failed for selected subjects")
            with self.lock:
                for pid in subjects:
                    if not self.memory.current(pid, token):
                        continue
                    self.clinical[pid] = {
                        "state": "error",
                        "stage": "MIMIC-IV 读取失败",
                        "error": str(exc),
                    }
