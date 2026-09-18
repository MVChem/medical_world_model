"""Audit a prepared index and optionally measure real cold-patient API reads."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
import urllib.request
from contextlib import closing
from pathlib import Path

from .patient_index import PatientIndex, connect, validate_source
from .prepare_index import atomic_json, digest_rows


def verify(cache):
    result = {"tables": {}, "total_rows": 0, "raw_sample_checks": 0}
    assert cache.manifest["storage"] == "csv_offsets"
    assert not cache.manifest["stores_clinical_records"]
    assert not cache.manifest["stores_report_text"]
    assert not list(cache.root.rglob("*.parquet"))
    assert not list(cache.root.rglob("*.csv"))
    for name, meta in cache.tables.items():
        validate_source(meta)
        folder = cache.root / "tables" / name
        rows = meta["rows"]
        if meta["patient_table"]:
            total = 0
            previous_end = meta["header_bytes"]
            spans = 0
            with closing(connect(folder / "subjects.sqlite")) as db:
                assert [r[1] for r in db.execute("PRAGMA table_info(spans)")] == [
                    "subject_id",
                    "start_byte",
                    "end_byte",
                    "row_count",
                ]
                assert db.execute("PRAGMA quick_check").fetchone()[0] == "ok"
                for subject, start, end, count in db.execute(
                    "SELECT subject_id,start_byte,end_byte,row_count FROM spans ORDER BY start_byte"
                ):
                    assert (
                        subject
                        and count > 0
                        and previous_end <= start < end <= meta["source"]["bytes"]
                    ), (name, subject)
                    previous_end = end
                    total += count
                    spans += 1
                patients = db.execute(
                    "SELECT count(DISTINCT subject_id) FROM spans"
                ).fetchone()[0]
            assert patients == meta["patients"]
            assert total == rows and spans == meta["spans"], name
            for subject, expected in meta["sample_validation"].items():
                raw = cache.read_subject(name, subject)
                digest = hashlib.sha256()
                digest_rows(digest, raw, meta["fields"])
                assert (
                    len(raw) == expected["rows"]
                    and digest.hexdigest() == expected["sha256"]
                ), (name, subject)
                result["raw_sample_checks"] += 1
        result["tables"][name] = {
            "rows": rows,
            "patients": meta["patients"],
            "bytes": meta["bytes"],
        }
        result["total_rows"] += rows
    with closing(connect(cache.root / "cxr.sqlite")) as db:
        assert (
            db.execute(
                "SELECT count(*) FROM images i LEFT JOIN studies s ON i.subject_id=s.subject_id AND i.study_id=s.study_id WHERE s.study_id IS NULL"
            ).fetchone()[0]
            == 0
        )
        expected = {
            r[0]: (r[1], r[2])
            for r in db.execute("SELECT dicom_id,subject_id,study_id FROM images")
        }
        seen = set()
        for row in cache.iter_table("cxr.mimic-cxr-2.0.0-metadata"):
            assert row["dicom_id"] not in seen, row["dicom_id"]
            assert expected[row["dicom_id"]] == (row["subject_id"], row["study_id"]), (
                row["dicom_id"]
            )
            seen.add(row["dicom_id"])
        assert len(seen) == len(expected)
        result["cxr"] = {
            "images_checked": len(seen),
            "studies": db.execute("SELECT count(*) FROM studies").fetchone()[0],
            "reports": dict(
                db.execute(
                    "SELECT report_status,count(*) FROM studies GROUP BY report_status"
                )
            ),
        }
        assert [r[1] for r in db.execute("PRAGMA table_info(studies)")] == [
            "subject_id",
            "study_id",
            "report_path",
            "report_status",
        ]
        assert [r[1] for r in db.execute("PRAGMA table_info(images)")] == [
            "dicom_id",
            "subject_id",
            "study_id",
            "relative_path",
            "available",
        ]
        checked_reports = 0
        for subject in ("10000032", "10004606", "12137189"):
            reports = cache.reports(subject)
            for study, relative, status in db.execute(
                "SELECT study_id,report_path,report_status FROM studies WHERE subject_id=?",
                (subject,),
            ):
                path = Path(cache.manifest["cxr_root"]) / relative
                assert reports[study]["text"] == (
                    path.read_text() if path.exists() else None
                )
                assert reports[study]["status"] == status
                checked_reports += 1
        result["cxr"]["raw_reports_checked"] = checked_reports
    with closing(connect(cache.root / "directory.sqlite")) as db:
        count = db.execute(
            "SELECT count(DISTINCT subject_id) FROM patient_tables"
        ).fetchone()[0]
        assert count == cache.manifest["patients"]
        for name, meta in cache.tables.items():
            if meta["patient_table"]:
                assert (
                    db.execute(
                        "SELECT coalesce(sum(row_count),0) FROM patient_tables WHERE table_name=?",
                        (name,),
                    ).fetchone()[0]
                    == meta["rows"]
                )
        result["patients"] = count
    return result


def verify_api(cache, url):
    def request(path, payload=None):
        req = urllib.request.Request(
            url.rstrip("/") + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Accept-Encoding": "gzip", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            body = response.read()
            size = len(body)
            if response.headers.get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)
            return json.loads(body), size

    for _ in range(600):
        catalog, _ = request("/api/catalog")
        assert catalog["state"] != "error", catalog
        if catalog.get("cohort", {}).get("state") == "ready":
            break
        time.sleep(0.2)
    assert catalog["storage"]["mode"] == "patient_index", catalog
    output = {}
    for subject in ("10004606", "12137189", "10000032"):
        start = time.monotonic()
        for _ in range(300):
            data, _ = request(f"/api/patients/{subject}")
            assert data["clinical_status"]["state"] != "error", data["clinical_status"]
            if data["clinical_status"]["state"] == "ready":
                break
            time.sleep(0.1)
        assert data["clinical_status"]["state"] == "ready"
        base_seconds = time.monotonic() - start
        counts = cache.directory(subject)
        assert len(data["studies"]) == counts.get("cxr.studies", 0)
        start = time.monotonic()
        names = ["labevents", "chartevents", "emar_detail", "poe_detail"]
        for _ in range(600):
            manifest, _ = request(f"/api/patients/{subject}/tables")
            selected = [m for m in manifest if m["name"] in names]
            assert not any(m["state"] in {"error", "unavailable"} for m in selected), (
                selected
            )
            if all(m["state"] == "ready" for m in selected):
                break
            time.sleep(0.1)
        assert all(m["state"] == "ready" for m in selected)
        load_seconds = time.monotonic() - start
        for meta in selected:
            assert meta["count"] == counts.get(meta["module"] + "." + meta["name"], 0)
            assert meta["rows_scanned"] == 0 and meta["read_method"] == "patient_index"
            result, _ = request(
                f"/api/patients/{subject}/tables/{meta['name']}?scope=patient"
            )
            assert all(r["raw"]["subject_id"] == subject for r in result["rows"])
        labs, size = request(
            f"/api/patients/{subject}/tables/labevents?scope=patient&limit=50"
        )
        output[subject] = {
            "base_seconds": round(base_seconds, 3),
            "four_tables_additional_seconds": round(load_seconds, 3),
            "counts": {m["name"]: m["count"] for m in selected},
            "lab_page_rows": len(labs["rows"]),
            "lab_page_transfer_bytes": size,
            "source_rows_scanned": 0,
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--url")
    args = parser.parse_args()
    cache = PatientIndex(args.index_root)
    result = verify(cache)
    if args.url:
        result["api"] = verify_api(cache, args.url)
    result["state"] = "passed"
    atomic_json(args.index_root / "verification.json", result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "tables"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
