"""Persistent, lossless patient directory. No temporal pairing policy lives here.

Parquet stores original CSV strings; SQLite maps each subject to exact row spans.
Only the row groups intersecting that patient's span are read by a lookup.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

VERSION = 1
ORDINAL = "__mimic_source_row"
BUCKET = "__mimic_bucket"
ROW_GROUP = 32768


def fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def connect(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


class PatientIndex:
    def __init__(self, root, *, iv_root=None, cxr_root=None, validate=True):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if (
            self.manifest.get("version") != VERSION
            or self.manifest.get("state") != "ready"
        ):
            raise ValueError(
                "Patient index is incomplete or has an unsupported version"
            )
        for label, path in (("iv_root", iv_root), ("cxr_root", cxr_root)):
            if path is not None and str(Path(path).resolve()) != self.manifest[label]:
                raise ValueError(f"Index source root mismatch: {label}")
        self.tables = self.manifest["tables"]
        if validate:
            for name, meta in self.tables.items():
                if fingerprint(meta["source"]["path"]) != meta["source"]:
                    raise ValueError(
                        f"Stale index for {name}; rebuild into a new run directory"
                    )

    def table(self, name):
        if name not in self.tables:
            raise FileNotFoundError(f"Table absent from prepared index: {name}")
        return self.tables[name]

    def directory(self, subject):
        with closing(connect(self.root / "directory.sqlite")) as db:
            return dict(
                db.execute(
                    "SELECT table_name, row_count FROM patient_tables WHERE subject_id=?",
                    (str(subject),),
                )
            )

    def subjects(self):
        with closing(connect(self.root / "directory.sqlite")) as db:
            yield from (
                r[0]
                for r in db.execute("SELECT DISTINCT subject_id FROM patient_tables")
            )

    def iv_subjects(self):
        with closing(connect(self.root / "directory.sqlite")) as db:
            yield from (
                r[0]
                for r in db.execute(
                    "SELECT DISTINCT subject_id FROM patient_tables WHERE table_name LIKE 'hosp.%' OR table_name LIKE 'icu.%'"
                )
            )

    def read_subject(self, name, subject):
        return read_subject(self.root / "tables" / name, self.table(name), str(subject))

    def iter_table(self, name):
        meta = self.table(name)
        for path in sorted((self.root / "tables" / name).glob("*.parquet")):
            for batch in pq.ParquetFile(path).iter_batches(columns=meta["fields"]):
                yield from batch.to_pylist()

    def reports(self, subject):
        with closing(connect(self.root / "cxr.sqlite")) as db:
            return {
                r[0]: {"text": r[1], "status": r[2]}
                for r in db.execute(
                    "SELECT study_id, report, report_status FROM studies WHERE subject_id=?",
                    (str(subject),),
                )
            }

    def load_studies(self, root):
        from . import build_mimic_transitions as cxr

        studies = {}
        with closing(connect(self.root / "cxr.sqlite")) as db:
            for (payload,) in db.execute(
                "SELECT payload FROM studies ORDER BY subject_id, study_id"
            ):
                row = json.loads(payload)
                images = [
                    cxr.ImageInfo(
                        **{
                            **im,
                            "timestamp": datetime.fromisoformat(im["timestamp"]),
                            "image_path": Path(root) / im["relative_path"],
                        }
                    )
                    for im in row.pop("images")
                ]
                by_id = {im.dicom_id: im for im in images}
                best = row.pop("images_by_view")
                row["timestamp"] = datetime.fromisoformat(row["timestamp"])
                row["latest_image_timestamp"] = datetime.fromisoformat(
                    row["latest_image_timestamp"]
                )
                if row["labels"] is not None:
                    row["labels"] = tuple(row["labels"])
                study = cxr.Study(
                    **row,
                    report_path=Path(root) / row["report_relative_path"],
                    images=images,
                    images_by_view={v: by_id[i] for v, i in best.items()},
                )
                studies[study.subject_id, study.study_id] = study
        return studies


def read_subject(folder, meta, subject):
    if not meta["patient_table"]:
        raise ValueError("Table has no subject_id")
    with closing(connect(Path(folder) / "subjects.sqlite")) as db:
        found = db.execute(
            "SELECT filename, start_row, row_count FROM subjects WHERE subject_id=?",
            (subject,),
        ).fetchone()
    if not found:
        return []
    filename, start, count = found
    size = meta["row_group_size"]
    groups = list(range(start // size, (start + count - 1) // size + 1))
    table = pq.ParquetFile(Path(folder) / filename).read_row_groups(
        groups, columns=meta["fields"]
    )
    rows = table.slice(start % size, count).to_pylist()
    if len(rows) != count or any(r["subject_id"] != subject for r in rows):
        raise ValueError(f"Corrupt patient span in {folder}: {subject}")
    return rows
