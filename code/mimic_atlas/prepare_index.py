"""Build patient byte offsets into original CSVs; never copy clinical records."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from datetime import datetime
from pathlib import Path

from .patient_index import VERSION, connect, fingerprint, read_subject

LOG = logging.getLogger(__name__)
SAMPLES = ("10000032", "10004606", "12137189")
COMPILE_LOCK = threading.Lock()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def digest_rows(digest, rows, fields):
    for row in rows:
        digest.update(
            json.dumps(
                [row[f] for f in fields], ensure_ascii=False, separators=(",", ":")
            ).encode()
            + b"\n"
        )


def scanner_binary():
    source = Path(__file__).with_name("csv_spans.cpp")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    folder = source.parent / "runs" / "index_tools"
    folder.mkdir(parents=True, exist_ok=True)
    binary = folder / ("csv_spans_" + digest)
    with COMPILE_LOCK, (folder / ".compile.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not binary.is_file():
            compiler = shutil.which("g++")
            if not compiler:
                raise RuntimeError(
                    "g++ is required to build the streaming CSV offset scanner"
                )
            temporary = binary.with_suffix(".tmp")
            subprocess.run(
                [compiler, "-O3", "-std=c++17", str(source), "-o", str(temporary)],
                check=True,
            )
            os.replace(temporary, binary)
    return binary


def build_table(
    path,
    output,
    name,
    *,
    block_size=16 * 1024 * 1024,
    samples=SAMPLES,
    progress=None,
    reference=None,
):
    started, path, output = time.monotonic(), Path(path), Path(output)
    if path.suffix != ".csv":
        raise ValueError(
            f"CSV offsets require an uncompressed .csv source; unpack {path} first"
        )
    source = fingerprint(path)
    target = output / "tables" / name
    if (target / "metadata.json").is_file():
        meta = json.loads((target / "metadata.json").read_text())
        if meta["source"] != source or meta["version"] != VERSION:
            raise ValueError(
                f"Cannot reuse stale {name}; select a new output directory"
            )
        return meta
    if reference and reference["source"] != source:
        raise ValueError(f"Reference source differs for {name}")
    work = output / ".building" / name
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        fields = next(csv.reader(stream))
    if not fields or len(fields) != len(set(fields)):
        raise ValueError(f"Duplicate/missing CSV fields: {path}")
    patient_table = "subject_id" in fields
    column = fields.index("subject_id") if patient_table else -1
    summary = None
    last_log = 0
    with closing(sqlite3.connect(work / "subjects.sqlite")) as db:
        db.execute(
            "CREATE TABLE spans(subject_id TEXT NOT NULL, start_byte INTEGER NOT NULL, end_byte INTEGER NOT NULL, row_count INTEGER NOT NULL, PRIMARY KEY(subject_id,start_byte)) WITHOUT ROWID"
        )
        records = []
        with subprocess.Popen(
            [
                str(scanner_binary()),
                str(path),
                str(column),
                str(len(fields)),
                str(block_size),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as process:
            try:
                for line in process.stdout:
                    owner, start, end, count = line.rstrip("\n").split("\t")
                    if owner == "#summary":
                        summary = tuple(map(int, (start, end, count)))
                        continue
                    start, end, count = int(start), int(end), int(count)
                    records.append((owner, start, end, count))
                    if len(records) >= 10000:
                        db.executemany("INSERT INTO spans VALUES(?,?,?,?)", records)
                        records.clear()
                    if time.monotonic() - last_log > 5:
                        info = {
                            "table": name,
                            "phase": "source_byte_offsets",
                            "bytes_read": end,
                            "percent": round(100 * end / max(1, source["bytes"]), 1),
                        }
                        LOG.info("%s", info)
                        if progress:
                            progress(info)
                        last_log = time.monotonic()
                error = process.stderr.read()
                if process.wait() != 0:
                    raise ValueError(f"{name}: {error.strip()}")
            except BaseException:
                process.terminate()
                process.wait()
                raise
        db.executemany("INSERT INTO spans VALUES(?,?,?,?)", records)
        db.commit()
        if summary is None:
            raise ValueError(f"Missing scan summary for {name}")
        total, header_bytes, scanned_bytes = summary
        patients, spans, indexed_rows = db.execute(
            "SELECT count(DISTINCT subject_id),count(*),coalesce(sum(row_count),0) FROM spans"
        ).fetchone()
        if patient_table and indexed_rows != total:
            raise ValueError(f"Patient counts do not reconcile: {name}")
    if scanned_bytes != source["bytes"] or fingerprint(path) != source:
        raise ValueError(f"Source changed or bytes lost: {name}")
    if reference and (reference["rows"] != total or reference["patients"] != patients):
        raise ValueError(
            f"Source counts differ from independently verified reference: {name}"
        )
    meta = {
        "version": VERSION,
        "storage": "csv_offsets",
        "name": name,
        "source": source,
        "fields": fields,
        "patient_table": patient_table,
        "rows": total,
        "patients": patients,
        "spans": spans,
        "header_bytes": header_bytes,
        "sample_validation": {},
    }
    if patient_table:
        for subject in samples:
            rows = read_subject(work, meta, subject)
            digest = hashlib.sha256()
            digest_rows(digest, rows, fields)
            check = {"rows": len(rows), "sha256": digest.hexdigest()}
            if (
                reference
                and reference.get("sample_validation", {}).get(subject) != check
            ):
                raise ValueError(
                    f"Raw row fidelity differs from reference: {name} / {subject}"
                )
            meta["sample_validation"][subject] = check
    meta["reference_validated"] = bool(reference)
    meta["seconds"] = round(time.monotonic() - started, 3)
    meta["bytes"] = sum(p.stat().st_size for p in work.iterdir())
    atomic_json(work / "metadata.json", meta)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(work, target)
    LOG.info(
        "Completed %s: %s rows / %s patients / %.1f MiB in %.1fs",
        name,
        total,
        patients,
        meta["bytes"] / 2**20,
        meta["seconds"],
    )
    return meta


def build_cxr(root, output, progress=None):
    from . import build_mimic_transitions as cxr

    target, summary = output / "cxr.sqlite", output / "cxr_summary.json"
    if target.is_file() and summary.is_file():
        return json.loads(summary.read_text())
    temporary = output / ".building" / "cxr.sqlite"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.unlink(missing_ok=True)
    audit = cxr.Audit()
    studies = cxr.load_studies(root, "frontal", audit, all_views=True)
    cxr.attach_splits(root, studies, audit, strict=False)
    cxr.attach_labels(root, studies, audit)
    counts = Counter(studies=len(studies))
    with closing(sqlite3.connect(temporary)) as db:
        db.execute(
            "CREATE TABLE studies(subject_id TEXT, study_id TEXT, report_path TEXT NOT NULL, report_status TEXT NOT NULL, PRIMARY KEY(subject_id,study_id)) WITHOUT ROWID"
        )
        db.execute(
            "CREATE TABLE images(dicom_id TEXT PRIMARY KEY, subject_id TEXT NOT NULL, study_id TEXT NOT NULL, relative_path TEXT NOT NULL, available INTEGER NOT NULL) WITHOUT ROWID"
        )
        for n, study in enumerate(studies.values(), 1):
            for im in study.images:
                available = im.image_path.is_file()
                db.execute(
                    "INSERT INTO images VALUES(?,?,?,?,?)",
                    (
                        im.dicom_id,
                        study.subject_id,
                        study.study_id,
                        im.relative_path,
                        int(available),
                    ),
                )
                counts["images"] += 1
                counts["missing_images"] += not available
            try:
                report_status = (
                    "present" if study.report_path.stat().st_size else "empty"
                )
            except FileNotFoundError:
                report_status = "missing"
            counts[f"reports_{report_status}"] += 1
            db.execute(
                "INSERT INTO studies VALUES(?,?,?,?)",
                (
                    study.subject_id,
                    study.study_id,
                    study.report_relative_path,
                    report_status,
                ),
            )
            if n % 5000 == 0:
                db.commit()
                info = {
                    "table": "cxr.studies",
                    "phase": "file_references",
                    "rows": n,
                    "percent": round(n / len(studies) * 100, 1),
                }
                LOG.info("%s", info)
                if progress:
                    progress(info)
        db.commit()
        counts["patients"] = db.execute(
            "SELECT count(DISTINCT subject_id) FROM studies"
        ).fetchone()[0]
    result = {
        **counts,
        "audit": audit.as_dict(),
        "rules": {
            "patient": "Exact subject_id; never link different subjects",
            "study": "Exact (subject_id, study_id); retain every view and time point",
            "image": "Unique dicom_id and original file path; no copied image pixels",
            "report": "Original files/pXX/pSUBJECT/sSTUDY.txt path; read current text from source",
            "iv": "Byte spans into original CSV; identifiers, duplicates, missing values and source order preserved",
            "temporal_pairs": "Not computed; AB / BC / ABC / AC remain downstream choices",
        },
    }
    os.replace(temporary, target)
    atomic_json(summary, result)
    return result


def build_index(iv_root, cxr_root, output, *, workers=2, reference_index=None):
    if not 1 <= workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    iv_root, cxr_root, output = (
        Path(iv_root).resolve(),
        Path(cxr_root).resolve(),
        Path(output).resolve(),
    )
    reference = (
        json.loads((Path(reference_index) / "manifest.json").read_text())["tables"]
        if reference_index
        else {}
    )
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        tables = {}
        progress_lock = threading.Lock()
        active = {}

        def progress(info):
            with progress_lock:
                active[info["table"]] = info
                atomic_json(
                    output / "progress.json",
                    {
                        "state": "building",
                        "completed_tables": len(tables),
                        "elapsed_seconds": round(time.monotonic() - started, 1),
                        "active": list(active.values()),
                        **info,
                    },
                )

        paths = {}
        for module in ("hosp", "icu"):
            for path in sorted((iv_root / module).glob("*.csv*")):
                name = module + "." + path.name.split(".csv")[0]
                if name in paths:
                    raise ValueError(f"Ambiguous duplicate source table: {name}")
                paths[name] = path
        for path in sorted(cxr_root.glob("mimic-cxr-*.csv*")):
            paths["cxr." + path.name.split(".csv")[0]] = path
        if not paths:
            raise ValueError("No source tables found")
        try:
            # Reuse completed tables first, then start the largest remaining jobs.
            ordered = sorted(
                paths.items(),
                key=lambda item: (
                    not (output / "tables" / item[0] / "metadata.json").is_file(),
                    -item[1].stat().st_size,
                ),
            )
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="prepare-table"
            ) as pool:
                futures = {
                    pool.submit(
                        build_table,
                        path,
                        output,
                        name,
                        progress=progress,
                        reference=reference.get(name),
                    ): name
                    for name, path in ordered
                }
                try:
                    for future in as_completed(futures):
                        name = futures[future]
                        tables[name] = future.result()
                        with progress_lock:
                            active.pop(name, None)
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
            cxr_summary = build_cxr(cxr_root, output, progress)
            directory = output / ".building" / "directory.sqlite"
            directory.unlink(missing_ok=True)
            with closing(sqlite3.connect(directory)) as db:
                db.execute(
                    "CREATE TABLE patient_tables(subject_id TEXT, table_name TEXT, row_count INTEGER NOT NULL, PRIMARY KEY(subject_id,table_name)) WITHOUT ROWID"
                )
                for name, meta in tables.items():
                    if not meta["patient_table"]:
                        continue
                    with closing(
                        connect(output / "tables" / name / "subjects.sqlite")
                    ) as source:
                        db.executemany(
                            "INSERT INTO patient_tables VALUES(?,?,?)",
                            (
                                (s, name, n)
                                for s, n in source.execute(
                                    "SELECT subject_id, sum(row_count) FROM spans GROUP BY subject_id"
                                )
                            ),
                        )
                with closing(connect(output / "cxr.sqlite")) as source:
                    for table in ("studies", "images"):
                        db.executemany(
                            "INSERT INTO patient_tables VALUES(?,?,?)",
                            (
                                (s, "cxr." + table, n)
                                for s, n in source.execute(
                                    f"SELECT subject_id,count(*) FROM {table} GROUP BY subject_id"
                                )
                            ),
                        )
                db.commit()
                patients = db.execute(
                    "SELECT count(DISTINCT subject_id) FROM patient_tables"
                ).fetchone()[0]
            os.replace(directory, output / "directory.sqlite")
            for name, meta in tables.items():
                if fingerprint(meta["source"]["path"]) != meta["source"]:
                    raise ValueError(f"Source changed during preparation: {name}")
            manifest = {
                "state": "ready",
                "version": VERSION,
                "storage": "csv_offsets",
                "stores_clinical_records": False,
                "stores_report_text": False,
                "created_at": datetime.now().astimezone().isoformat(),
                "iv_root": str(iv_root),
                "cxr_root": str(cxr_root),
                "tables": tables,
                "patients": patients,
                "cxr": cxr_summary,
                "seconds_this_run": round(time.monotonic() - started, 2),
                "bytes": sum(
                    p.stat().st_size for p in output.rglob("*") if p.is_file()
                ),
            }
            atomic_json(output / "manifest.json", manifest)
            atomic_json(
                output / "progress.json",
                {k: v for k, v in manifest.items() if k != "tables"},
            )
            LOG.info(
                "INDEX READY: %s patients, %.2f GiB",
                patients,
                manifest["bytes"] / 2**30,
            )
            return manifest
        except Exception as exc:
            atomic_json(
                output / "progress.json",
                {"state": "error", "error": str(exc), "completed_tables": len(tables)},
            )
            raise


def main():
    from .data import DATA_ROOT, ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iv-root", type=Path, default=DATA_ROOT / "mimic-iv-3.1")
    parser.add_argument("--cxr-root", type=Path, default=DATA_ROOT / "MIMIC_CXR")
    parser.add_argument(
        "--workers",
        type=int,
        choices=range(1, 5),
        default=2,
        help="Concurrent table builders (default: 2; use 1 to reduce memory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "runs"
        / (
            "patient_index_"
            + datetime.now().astimezone().strftime("%Y%m%d")
            + "_offsets"
        ),
    )
    parser.add_argument(
        "--reference-index",
        type=Path,
        help="Verified old index for independent row-count and sample-hash comparison",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output / "build.log"),
        ],
    )
    build_index(
        args.iv_root,
        args.cxr_root,
        args.output,
        workers=args.workers,
        reference_index=args.reference_index,
    )


if __name__ == "__main__":
    main()
