"""Patient offsets into original CSVs and CXR file references; no copied records."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from contextlib import closing
from pathlib import Path

VERSION = 2


def fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def connect(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def validate_source(meta):
    try:
        valid = fingerprint(meta["source"]["path"]) == meta["source"]
    except FileNotFoundError:
        valid = False
    if not valid:
        raise ValueError(f"Stale index for {meta['name']}; rebuild the source offsets")


class PatientIndex:
    def __init__(self, root, *, iv_root=None, cxr_root=None, validate=True):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if (
            self.manifest.get("version") != VERSION
            or self.manifest.get("state") != "ready"
        ):
            raise ValueError("A complete CSV-offset index (version 2) is required")
        for label, path in (("iv_root", iv_root), ("cxr_root", cxr_root)):
            if path is not None and str(Path(path).resolve()) != self.manifest[label]:
                raise ValueError(f"Index source root mismatch: {label}")
        self.tables = self.manifest["tables"]
        if validate:
            for meta in self.tables.values():
                validate_source(meta)

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
        """Startup metadata/dictionaries come from their original CSV, once."""
        meta = self.table(name)
        validate_source(meta)
        with Path(meta["source"]["path"]).open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            yield from csv.DictReader(stream)

    def reports(self, subject):
        result = {}
        root = Path(self.manifest["cxr_root"])
        with closing(connect(self.root / "cxr.sqlite")) as db:
            for study, relative in db.execute(
                "SELECT study_id,report_path FROM studies WHERE subject_id=?",
                (str(subject),),
            ):
                path = (root / relative).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("Report path escapes CXR source directory")
                try:
                    text = path.read_text(encoding="utf-8")
                    status = "present" if text else "empty"
                except FileNotFoundError:
                    text, status = None, "missing"
                result[study] = {"text": text, "status": status}
        return result

    def load_studies(self, root):
        # The overview needs all CXR metadata. Read these small source tables once
        # at startup and keep them in memory, without a payload snapshot on disk.
        from . import build_mimic_transitions as cxr

        for name, meta in self.tables.items():
            if name.startswith("cxr."):
                validate_source(meta)
        audit = cxr.Audit()
        studies = cxr.load_studies(root, "frontal", audit, all_views=True)
        cxr.attach_splits(root, studies, audit, strict=False)
        cxr.attach_labels(root, studies, audit)
        return studies


def read_subject(folder, meta, subject):
    if not meta["patient_table"]:
        raise ValueError("Table has no subject_id")
    validate_source(meta)
    with closing(connect(Path(folder) / "subjects.sqlite")) as db:
        spans = db.execute(
            "SELECT start_byte,end_byte,row_count FROM spans WHERE subject_id=? ORDER BY start_byte",
            (str(subject),),
        ).fetchall()
    rows = []
    if not spans:
        return rows
    fields = meta["fields"]
    with Path(meta["source"]["path"]).open("rb") as source:
        for start, end, count in spans:
            if not meta["header_bytes"] <= start < end <= meta["source"]["bytes"]:
                raise ValueError(f"Corrupt patient byte span in {folder}: {subject}")
            source.seek(start)
            data = source.read(end - start)
            if len(data) != end - start:
                raise ValueError(f"Truncated source CSV in {folder}")
            selected = []
            for values in csv.reader(io.StringIO(data.decode("utf-8"), newline="")):
                if not values:
                    continue
                if len(values) != len(fields):
                    raise ValueError(f"Invalid CSV record in {folder}")
                row = dict(zip(fields, values))
                if row["subject_id"] != str(subject):
                    raise ValueError(
                        f"Patient ownership mismatch in {folder}: {subject}"
                    )
                selected.append(row)
            if len(selected) != count:
                raise ValueError(f"Patient count mismatch in {folder}: {subject}")
            rows.extend(selected)
    validate_source(meta)
    return rows
