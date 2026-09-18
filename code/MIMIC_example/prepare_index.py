"""Build all-patient IV/CXR caches under runs; resume at completed table boundaries.

Rules: exact subject_id ownership; exact (subject_id, study_id) CXR grouping;
dicom_id belongs to one study; original IV identifiers and missing values survive.
No adjacency, view, time-window, split or training eligibility filter is applied.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gzip
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from .patient_index import (
    BUCKET,
    ORDINAL,
    ROW_GROUP,
    VERSION,
    connect,
    fingerprint,
    read_subject,
)

LOG = logging.getLogger(__name__)
SAMPLES = ("10000032", "10004606", "12137189")


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


def build_table(
    path,
    output,
    name,
    *,
    block_size=16 * 1024 * 1024,
    row_group=ROW_GROUP,
    samples=SAMPLES,
    progress=None,
):
    """Stable subject grouping even for unsorted CSVs, preserving every raw row."""
    started, path, output = time.monotonic(), Path(path), Path(output)
    source = fingerprint(path)
    target = output / "tables" / name
    if (target / "metadata.json").is_file():
        meta = json.loads((target / "metadata.json").read_text())
        if meta["source"] != source or meta["version"] != VERSION:
            raise ValueError(
                f"Cannot reuse stale {name}; select a new output directory"
            )
        return meta
    work = output / ".building" / name
    if work.exists():
        shutil.rmtree(work)
    stage = work / "stage"
    stage.mkdir(parents=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        fields = next(csv.reader(stream))
    if len(fields) != len(set(fields)) or {ORDINAL, BUCKET} & set(fields):
        raise ValueError(f"Duplicate/reserved field in {path}")
    patient_table = "subject_id" in fields
    hashes = {s: hashlib.sha256() for s in samples} if patient_table else {}
    sample_counts = Counter()
    writers, scanned = {}, 0
    last_log = 0
    try:
        with pa.OSFile(str(path), "rb") as raw:
            stream = (
                pa.CompressedInputStream(raw, "gzip") if path.suffix == ".gz" else raw
            )
            reader = pacsv.open_csv(
                stream,
                read_options=pacsv.ReadOptions(
                    block_size=block_size, use_threads=False
                ),
                parse_options=pacsv.ParseOptions(newlines_in_values=True),
                convert_options=pacsv.ConvertOptions(
                    column_types={f: pa.string() for f in fields},
                    strings_can_be_null=False,
                ),
            )
            for batch in reader:
                table = pa.Table.from_batches([batch])
                if patient_table:
                    ids = table["subject_id"]
                    # Numeric buckets are only physical storage locations; string IDs
                    # remain exact keys, including leading zeros. Invalid IDs fail loudly.
                    valid = pc.match_substring_regex(ids, "^[0-9]+$")
                    if not pc.all(valid).as_py():
                        raise ValueError(
                            f"Missing/invalid subject_id in {name} near row {scanned}"
                        )
                    buckets = pc.divide(pc.cast(ids, pa.int64()), 100000)
                    selected = table.filter(
                        pc.is_in(ids, pa.array(list(hashes), type=pa.string()))
                    )
                    for subject, expected_hash in hashes.items():
                        rows = selected.filter(
                            pc.equal(selected["subject_id"], subject)
                        ).to_pylist()
                        digest_rows(expected_hash, rows, fields)
                        sample_counts[subject] += len(rows)
                else:
                    buckets = pa.array([0] * len(table), type=pa.int64())
                table = table.append_column(
                    ORDINAL,
                    pa.array(range(scanned, scanned + len(table)), type=pa.uint64()),
                )
                table = table.append_column(BUCKET, buckets).sort_by(
                    [(BUCKET, "ascending")]
                )
                runs = pc.run_end_encode(table[BUCKET].combine_chunks())
                start = 0
                for end, bucket in zip(
                    runs.run_ends.to_pylist(), runs.values.to_pylist()
                ):
                    piece = table.slice(start, end - start).drop([BUCKET])
                    if bucket not in writers:
                        writers[bucket] = pq.ParquetWriter(
                            stage / f"{bucket}.parquet",
                            piece.schema,
                            compression="zstd",
                            compression_level=1,
                        )
                    writers[bucket].write_table(piece)
                    start = end
                scanned += len(table)
                if time.monotonic() - last_log > 10:
                    info = {
                        "table": name,
                        "phase": "partition",
                        "rows": scanned,
                        "percent": round(raw.tell() / max(1, source["bytes"]) * 100, 1),
                    }
                    LOG.info("%s", info)
                    if progress:
                        progress(info)
                    last_log = time.monotonic()
    finally:
        for writer in writers.values():
            writer.close()
    total, patients = 0, 0
    with closing(sqlite3.connect(work / "subjects.sqlite")) as db:
        db.execute(
            "CREATE TABLE subjects(subject_id TEXT PRIMARY KEY, filename TEXT NOT NULL, start_row INTEGER NOT NULL, row_count INTEGER NOT NULL) WITHOUT ROWID"
        )
        for i, part in enumerate(sorted(stage.glob("*.parquet"))):
            table = pq.read_table(part)
            keys = (
                [("subject_id", "ascending"), (ORDINAL, "ascending")]
                if patient_table
                else [(ORDINAL, "ascending")]
            )
            table = table.sort_by(keys)
            filename = part.name
            pq.write_table(
                table,
                work / filename,
                row_group_size=row_group,
                compression="zstd",
                compression_level=3,
            )
            if patient_table:
                runs = pc.run_end_encode(table["subject_id"].combine_chunks())
                start, records = 0, []
                for end, subject in zip(
                    runs.run_ends.to_pylist(), runs.values.to_pylist()
                ):
                    records.append((subject, filename, start, end - start))
                    start = end
                db.executemany("INSERT INTO subjects VALUES(?,?,?,?)", records)
                patients += len(records)
            total += len(table)
            del table
            part.unlink()
            if progress:
                progress(
                    {
                        "table": name,
                        "phase": "sort_and_index",
                        "rows": total,
                        "percent": round((i + 1) / max(1, len(writers)) * 100, 1),
                    }
                )
        db.commit()
        if patient_table and (
            db.execute("SELECT coalesce(sum(row_count),0) FROM subjects").fetchone()[0]
            != scanned
        ):
            raise ValueError(f"Patient counts do not reconcile: {name}")
    if scanned != total or fingerprint(path) != source:
        raise ValueError(f"Source changed or rows lost: {name}")
    stage.rmdir()
    meta = {
        "version": VERSION,
        "name": name,
        "source": source,
        "fields": fields,
        "patient_table": patient_table,
        "rows": total,
        "patients": patients,
        "row_group_size": row_group,
        "sample_validation": {},
    }
    for subject, expected in hashes.items():
        rows = read_subject(work, meta, subject)
        actual = hashlib.sha256()
        digest_rows(actual, rows, fields)
        if actual.digest() != expected.digest() or len(rows) != sample_counts[subject]:
            raise ValueError(f"Raw row fidelity check failed: {name} / {subject}")
        meta["sample_validation"][subject] = {
            "rows": len(rows),
            "sha256": actual.hexdigest(),
        }
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
            "CREATE TABLE studies(subject_id TEXT, study_id TEXT, payload TEXT NOT NULL, report TEXT, report_status TEXT NOT NULL, PRIMARY KEY(subject_id,study_id)) WITHOUT ROWID"
        )
        db.execute(
            "CREATE TABLE images(dicom_id TEXT PRIMARY KEY, subject_id TEXT NOT NULL, study_id TEXT NOT NULL, relative_path TEXT NOT NULL, available INTEGER NOT NULL) WITHOUT ROWID"
        )
        for n, study in enumerate(studies.values(), 1):
            row = asdict(study)
            row.pop("report_path")
            row["timestamp"] = study.timestamp.isoformat()
            row["latest_image_timestamp"] = study.latest_image_timestamp.isoformat()
            row["images_by_view"] = {
                v: im.dicom_id for v, im in study.images_by_view.items()
            }
            for im, value in zip(study.images, row["images"]):
                value.pop("image_path")
                value["timestamp"] = im.timestamp.isoformat()
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
                report = study.report_path.read_text(encoding="utf-8")
                report_status = "present" if report else "empty"
            except FileNotFoundError:
                report, report_status = None, "missing"
            counts[f"reports_{report_status}"] += 1
            db.execute(
                "INSERT INTO studies VALUES(?,?,?,?,?)",
                (
                    study.subject_id,
                    study.study_id,
                    json.dumps(row, ensure_ascii=False),
                    report,
                    report_status,
                ),
            )
            if n % 5000 == 0:
                db.commit()
                info = {
                    "table": "cxr.studies",
                    "phase": "reports_and_images",
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
            "image": "Unique dicom_id; conflicting ownership is a build error",
            "report": "files/pXX/pSUBJECT/sSTUDY.txt; full text, missing/empty distinguished",
            "iv": "Preserve original identifiers, values, duplicates and source row order",
            "temporal_pairs": "Not computed; AB / BC / ABC / AC remain downstream choices",
        },
    }
    os.replace(temporary, target)
    atomic_json(summary, result)
    return result


def build_index(iv_root, cxr_root, output, *, workers=2):
    if not 1 <= workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    iv_root, cxr_root, output = (
        Path(iv_root).resolve(),
        Path(cxr_root).resolve(),
        Path(output).resolve(),
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
                        build_table, path, output, name, progress=progress
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
                                    "SELECT subject_id, row_count FROM subjects"
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
        / ("patient_index_" + datetime.now().astimezone().strftime("%Y%m%d")),
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
    pa.set_cpu_count(4)
    pa.set_io_thread_count(4)
    build_index(args.iv_root, args.cxr_root, args.output, workers=args.workers)


if __name__ == "__main__":
    main()
