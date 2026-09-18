"""Exercise persistent grouping against unsorted raw data and independent patients."""
# ruff: noqa: F811 -- pytest fixture imported for injection

import csv
import json
import sqlite3
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from MIMIC_example.app import create_app
from MIMIC_example.patient_index import PatientIndex, read_subject
from MIMIC_example.prepare_index import build_index, build_table
from MIMIC_example.tests.test_atlas import await_patient, sources, table  # noqa: F401


def test_lossless_spans_cross_groups_and_restart(tmp_path):
    path = tmp_path / "raw.csv.gz"
    fields = ["row_id", "subject_id", "hadm_id", "value"]
    rows = [
        {
            "row_id": str(i),
            "subject_id": ["10000001", "10000002", "10100001"][i % 3],
            "hadm_id": "" if i % 2 else "3",
            "value": 'NULL, "001.20"\n第二行',
        }
        for i in range(120)
    ]
    rows += [dict(rows[0]), {**rows[0], "subject_id": "010000001"}]
    table(path, fields, rows)
    output = tmp_path / "index"
    meta = build_table(
        path,
        output,
        "hosp.events",
        block_size=1024,
        row_group=7,
        samples=["10000001", "10000002", "10100001", "010000001"],
    )
    assert meta["rows"] == len(rows) and meta["patients"] == 4
    for subject in ("10000001", "10000002", "10100001", "010000001", "999"):
        # Reopen metadata and SQLite/Parquet for every lookup, including absent IDs.
        persisted = json.loads(
            (output / "tables/hosp.events/metadata.json").read_text()
        )
        actual = read_subject(output / "tables/hosp.events", persisted, subject)
        assert actual == [r for r in rows if r["subject_id"] == subject]
    assert build_table(path, output, "hosp.events") == meta
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="stale"):
        build_table(path, output, "hosp.events")


def test_empty_table_and_invalid_owner(tmp_path):
    path = tmp_path / "empty.csv"
    table(path, ["subject_id", "value"], [])
    meta = build_table(path, tmp_path / "out", "empty")
    assert meta["rows"] == 0
    assert read_subject(tmp_path / "out/tables/empty", meta, "1") == []
    table(path, ["subject_id", "value"], [{"subject_id": "", "value": "unassigned"}])
    with pytest.raises(ValueError, match="subject_id"):
        build_table(path, tmp_path / "invalid", "empty")


def test_all_studies_images_reports_and_source_validation(sources, tmp_path):
    _, config = sources
    output = tmp_path / "index"
    manifest = build_index(config.iv_root, config.cxr_root, output)
    cache = PatientIndex(output, iv_root=config.iv_root, cxr_root=config.cxr_root)
    studies = cache.load_studies(config.cxr_root)
    assert len(studies) == 5
    patient = [s for s in studies.values() if s.subject_id == "10000002"]
    assert (
        len(patient) == 3
    )  # A, B (lateral), C all survive, regardless of pairing rules.
    assert sorted(im.view for s in patient for im in s.images) == [
        "AP",
        "AP",
        "LATERAL",
    ]
    reports = cache.reports("10000002")
    assert len(reports) == 3
    assert all(r["text"].endswith("</script><b>test</b>") for r in reports.values())
    assert cache.directory("10000002")["cxr.studies"] == 3
    assert cache.directory("999") == {} and cache.reports("999") == {}
    assert manifest["cxr"]["images"] == 5
    assert not manifest["cxr"].get("pairs")
    with pytest.raises(ValueError, match="root mismatch"):
        PatientIndex(output, iv_root=tmp_path)
    source = Path(next(iter(manifest["tables"].values()))["source"]["path"])
    source.touch()
    with pytest.raises(ValueError, match="Stale"):
        PatientIndex(output)


def test_api_new_patients_and_restart_never_scan_sources(sources, tmp_path):
    _, config = sources
    fields = ["subject_id", "hadm_id", "itemid", "value", "valuenum", "charttime"]
    labs = [
        {
            "subject_id": s,
            "hadm_id": "",
            "itemid": "1",
            "value": str(i),
            "valuenum": str(i),
            "charttime": "2180-01-01 12:00:00",
        }
        for i, s in enumerate(["10000002", "10000001", "10000002", "10999999"])
    ]
    table(config.iv_root / "hosp/labevents.csv.gz", fields, labs)
    table(
        config.iv_root / "hosp/d_labitems.csv.gz",
        ["itemid", "label"],
        [{"itemid": "1", "label": "Test lab"}],
    )
    output = tmp_path / "index"
    build_index(config.iv_root, config.cxr_root, output)
    config = replace(config, index_root=output)
    # These failures would expose a fallback scan even for a new patient/restart.
    with (
        patch(
            "MIMIC_example.clinical_tables.scan_subjects",
            side_effect=AssertionError("source scan"),
        ),
        patch(
            "MIMIC_example.link_mimic_iv_context.load_subject_rows",
            side_effect=AssertionError("source scan"),
        ),
        patch(
            "MIMIC_example.link_mimic_iv_context.load_dictionary",
            side_effect=AssertionError("source scan"),
        ),
        patch(
            "MIMIC_example.cohort.source_rows",
            side_effect=AssertionError("source scan"),
        ),
        patch(
            "MIMIC_example.build_mimic_transitions.load_studies",
            side_effect=AssertionError("metadata scan"),
        ),
        patch(
            "MIMIC_example.build_mimic_transitions.read_report",
            side_effect=AssertionError("report read"),
        ),
    ):
        for _ in range(2):
            with TestClient(create_app(config)) as client:
                for subject, expected in [
                    ("10000001", 1),
                    ("10000002", 2),
                    ("10999999", 1),
                ]:
                    data = await_patient(client, subject)
                    assert data["clinical_status"]["state"] == "ready"
                    manifest = client.get(f"/api/patients/{subject}/tables").json()
                    lab = next(m for m in manifest if m["name"] == "labevents")
                    assert (
                        lab["indexed"]
                        and lab["count"] == expected
                        and lab["state"] == "not_loaded"
                    )
                    client.post(
                        f"/api/patients/{subject}/tables",
                        json={"tables": ["labevents"]},
                    )
                    for attempt in range(200):
                        result = client.get(
                            f"/api/patients/{subject}/tables/labevents?scope=patient"
                        ).json()
                        if result["status"]["state"] == "ready":
                            break
                        assert result["status"]["state"] != "error", result
                        time.sleep(0.01)
                    assert result["total"] == expected
                    assert result["status"]["rows_scanned"] == 0
                    assert [r["raw"] for r in result["rows"]] == [
                        r for r in labs if r["subject_id"] == subject
                    ]
                assert (
                    client.get("/api/patients/10000002/directory").json()["tables"][
                        "cxr.studies"
                    ]
                    == 3
                )
                assert (
                    client.get("/api/patients/10000001/reports/50000003").status_code
                    == 404
                )
                assert (
                    client.get("/api/patients/10000002/reports/50000003").json()[
                        "status"
                    ]
                    == "present"
                )
                response = client.post("/api/patients/10000001/icu-inputs")
                assert response.status_code == 200
                assert await_patient(client)["icu_inputs"]


def test_missing_assets_survive_and_conflicting_image_ownership_blocks_publish(
    sources, tmp_path
):
    _, config = sources
    patient = config.cxr_root / "files/p10/p10000002"
    (patient / "s50000003.txt").unlink()
    (patient / "s50000004.txt").write_text("")
    (patient / "s50000003/50000003-image.jpg").unlink()
    splits = config.cxr_root / "mimic-cxr-2.0.0-split.csv"
    with splits.open() as stream:
        split_rows = list(csv.DictReader(stream))
    table(
        splits,
        list(split_rows[0]),
        [r for r in split_rows if r["study_id"] != "50000004"],
    )
    output = tmp_path / "missing"
    manifest = build_index(config.iv_root, config.cxr_root, output)
    cache = PatientIndex(output)
    assert manifest["cxr"]["studies"] == 5
    assert manifest["cxr"]["missing_images"] == 1
    assert cache.load_studies(config.cxr_root)["10000002", "50000004"].split is None
    assert {s: r["status"] for s, r in cache.reports("10000002").items()} == {
        "50000003": "missing",
        "50000004": "empty",
        "50000005": "present",
    }
    metadata = config.cxr_root / "mimic-cxr-2.0.0-metadata.csv"
    with metadata.open() as stream:
        rows = list(csv.DictReader(stream))
    rows.append({**rows[0], "subject_id": "10000002", "study_id": "50000003"})
    table(metadata, list(rows[0]), rows)
    rejected = tmp_path / "conflicting"
    with pytest.raises(sqlite3.IntegrityError):
        build_index(config.iv_root, config.cxr_root, rejected)
    assert not (rejected / "manifest.json").exists()
    assert json.loads((rejected / "progress.json").read_text())["state"] == "error"
