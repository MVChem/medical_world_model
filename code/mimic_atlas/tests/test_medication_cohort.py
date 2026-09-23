"""Medication browsing preserves exact interval semantics and source provenance."""
import gzip
import hashlib
import json
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from data_preprocessing import medication_events
from mimic_atlas.backend.medications import create_medication_router
from mimic_atlas.medication_cohort import COUNTS, MedicationCohort, SPLITS, TABLES
from mimic_atlas.patient_index import fingerprint


def emar(subject, identifier, timestamp, status="Administered", hadm="A"):
    return {"subject_id": subject, "emar_id": identifier, "charttime": timestamp,
            "medication": "Drug A", "event_txt": status, "hadm_id": hadm}


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    import mimic_atlas.medication_cohort as module
    cxr = tmp_path / "cxr"
    cxr.mkdir()
    root = tmp_path / "selection"
    root.mkdir()
    clinical = {name: [] for name in TABLES}
    clinical["hosp.emar"] = [
        emar("100", "start", "2180-01-01T05:00:00", hadm="A"),
        emar("100", "end", "2180-01-06T05:00:00", hadm="B"),
        emar("100", "before", "2180-01-01T04:59:59"),
        emar("100", "after", "2180-01-06T05:00:01"),
        emar("100", "notgiven", "2180-01-02T05:00:00", "Not Given"),
        emar("200", "other", "2180-01-02T05:00:00"),
    ]
    clinical["hosp.emar_detail"] = [
        {"subject_id": "100", "emar_id": "start", "dose_given": "2", "dose_given_unit": "mg",
         "route": "IV", "parent_field_ordinal": "1", "product_amount_given": "1", "product_unit": "mL"},
        {"subject_id": "100", "emar_id": "end", "dose_given": "1", "dose_given_unit": "mg", "route": "PO"},
    ]
    clinical["icu.inputevents"] = [
        {"subject_id": "100", "hadm_id": "C", "starttime": "2180-01-01T04:00:00",
         "endtime": "2180-01-01T06:00:00", "statusdescription": "FinishedRunning", "itemid": "1",
         "amount": "20", "amountuom": "mg", "rate": "10", "rateuom": "mg/hour", "orderid": "1"},
        {"subject_id": "100", "hadm_id": "C", "starttime": "2180-01-01T04:00:00",
         "endtime": "2180-01-01T06:00:00", "statusdescription": "FinishedRunning", "itemid": "1",
         "amount": "20", "amountuom": "mg", "rate": "0", "rateuom": "mg/hour", "orderid": "2"},
    ]
    clinical["icu.d_items"] = [{"itemid": "1", "linksto": "inputevents", "category": "Medications", "label": "Drug B"}]
    sources = {}
    for name in TABLES:
        path = tmp_path / (name + ".csv")
        path.write_text("original source " + name)
        sources[name] = fingerprint(path)

    class Index:
        def __init__(self, *args, **kwargs):
            pass

        def table(self, name):
            return {"source": sources[name]}

        def iter_table(self, name):
            return iter(clinical[name])

        def read_subject(self, name, subject):
            return [row for row in clinical[name] if row["subject_id"] == subject]

    monkeypatch.setattr(module, "PatientIndex", Index)
    counts = dict.fromkeys(COUNTS, 0)
    counts.update(records=3, emar=2, inputevents=1)
    pairs = []
    for i, (subject, split) in enumerate((("100", "train"), ("200", "train"), ("300", "test"))):
        row = {"id": f"medpair:{i}", "patient": subject, "split": split, "source": f"cxr:source-{subject}",
               "target": f"cxr:target-{subject}", "source_study_id": "1", "target_study_id": "3",
               "source_time": "2180-01-01T05:00:00", "target_time": "2180-01-06T05:00:00", "hours": 120,
               "source_view": "PA", "target_view": "AP", "full_timeline_adjacent": False,
               "medication": counts, "evidence_patient": subject}
        folder = cxr / "files" / f"p{subject[:2]}" / f"p{subject}"
        for side, study in (("source", "1"), ("target", "3")):
            image = folder / f"s{study}" / f"{side}-{subject}.jpg"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"image")
            report = folder / f"s{study}.txt"
            report.write_text(f"Original {side} report for {subject}")
            row[f"{side}_image"] = str(image.relative_to(cxr))
            row[f"{side}_report"] = str(report.relative_to(cxr))
        pairs.append(row)
    manifest = {"schema": "medworld-medication-selection-v1", "state": "complete", "created_at": "2026-09-23",
                "sources": {"clinical_index": str(tmp_path / "index"), "cxr_root": str(cxr),
                            "clinical_tables": sources, "cxr_tables": {}}, "rules": {"same_admission_required": False},
                "implementation_sha256": {"medication_events.py": hashlib.sha256(
                    module.Path(medication_events.__file__).read_bytes()).hexdigest()},
                "cohort": {"retained_pairs": 3, "retained_patients": 3,
                           "splits": {split: {"retained": sum(p["split"] == split for p in pairs)} for split in SPLITS}},
                "outputs": {}}

    def save():
        for split in SPLITS:
            # Deliberately create concatenated gzip members, as in real output.
            data = b"".join(gzip.compress((json.dumps(row) + "\n").encode()) for row in pairs if row["split"] == split)
            data = data or gzip.compress(b"")
            path = root / f"{split}.jsonl.gz"
            path.write_bytes(data)
            manifest["outputs"][path.name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        (root / "manifest.json").write_text(json.dumps(manifest))

    save()
    return root, cxr, pairs, clinical, manifest, save


def load(prepared, **kwargs):
    catalog = MedicationCohort(prepared[0], **kwargs)
    catalog._load()
    assert catalog.state == "ready", catalog.error
    return catalog


def test_paginated_catalog_keeps_nonadjacent_mixed_view_pairs(prepared):
    catalog = load(prepared)
    assert catalog.status()["pairs"] == 3
    assert catalog.search(limit=1)["rows"][0]["id"] == "medpair:0"
    assert catalog.search(page=2, limit=1)["rows"][0]["id"] == "medpair:1"
    assert catalog.search(page=20)["rows"] == []
    assert catalog.search(split="test")["total"] == 1
    assert catalog.search(subject_id="100")["total"] == 1
    assert catalog.search(subject_id="10")["total"] == 0
    assert catalog.search(q="MEDPAIR:2")["rows"][0]["patient"] == "300"
    row = catalog.detail("medpair:0")
    assert not row["full_timeline_adjacent"]
    assert row["endpoints"]["source"]["report"] == "Original source report for 100"
    assert row["endpoints"]["target"]["image_url"] == "/api/images/target-100?size=1024"
    assert row["endpoints"]["source"]["view"] != row["endpoints"]["target"]["view"]


def test_exact_interval_accepts_other_admissions_and_preserves_doses(prepared):
    catalog = load(prepared)
    result = catalog.medications("medpair:0", limit=2)
    assert result["total"] == result["expected_records"] == 3 and result["counts_match"]
    infusion, first = result["rows"]
    assert infusion["relative_start_hours"] == -1 and infusion["relative_end_hours"] == 1
    assert infusion["amount"] == "20" and infusion["amount_unit"] == "mg"
    assert infusion["dose_basis"] == "delivered amount for complete original ICU segment"
    assert first["relative_start_hours"] == 0
    assert first["dose_details"][0]["parent_field_ordinal"] == "1"
    assert first["raw_details"][0]["product_amount_given"] == "1"
    assert first["detail_references"][0]["patient_row_ordinal"] == 0
    second = catalog.medications("medpair:0", page=2, limit=2)["rows"][0]
    assert second["relative_start_hours"] == 120 and second["hadm_id"] == "B"
    assert catalog.medications("medpair:0", source="hosp.emar")["total"] == 2
    assert catalog.medications("medpair:0", q="drug b")["total"] == 1
    assert catalog.medications("medpair:0", page=30)["rows"] == []
    names = [row["raw_record"].get("emar_id") for row in catalog.medications("medpair:0")["rows"]]
    assert not set(names) & {"before", "after", "notgiven", "other"}


def test_cached_records_expire_and_stale_source_is_rejected(prepared):
    catalog = load(prepared, cache_idle_seconds=.01)
    catalog.medications("medpair:0")
    assert list(catalog.cache) == ["100"]
    time.sleep(.02)
    catalog._sweep()
    assert not catalog.cache
    catalog.medications("medpair:0")
    path = prepared[4]["sources"]["clinical_tables"]["hosp.emar"]["path"]
    from pathlib import Path
    Path(path).write_text("changed source")
    with pytest.raises(RuntimeError, match="Source changed"):
        catalog.medications("medpair:0")
    assert catalog.state == "error" and not catalog.cache


def test_pair_and_report_ownership_and_root_containment(prepared):
    prepared[2][0]["evidence_patient"] = "200"
    prepared[5]()
    catalog = MedicationCohort(prepared[0])
    catalog._load()
    assert catalog.state == "error" and catalog.table is None
    prepared[2][0]["evidence_patient"] = "100"
    prepared[2][0]["target_report"] = prepared[2][1]["target_report"]
    prepared[5]()
    catalog = load(prepared)
    with pytest.raises(ValueError, match="does not belong"):
        catalog.detail("medpair:0")
    prepared[2][0]["target_report"] = "files/p10/p100/s3.txt"
    prepared[5]()
    catalog = load(prepared)
    original = prepared[1] / "files/p10/p100/s3.txt"
    original.unlink()
    original.symlink_to(prepared[0] / "manifest.json")
    with pytest.raises(ValueError, match="within the source root"):
        catalog.detail("medpair:0")


def test_missing_shards_and_checksum_drift_do_not_publish_partial_index(prepared):
    (prepared[0] / "test.jsonl.gz").unlink()
    catalog = MedicationCohort(prepared[0])
    catalog._load()
    assert catalog.state == "error" and catalog.table is None
    prepared[5]()
    path = prepared[0] / "train.jsonl.gz"
    path.write_bytes(path.read_bytes() + b"tampered")
    catalog = MedicationCohort(prepared[0])
    catalog._load()
    assert catalog.state == "error" and "checksum" in catalog.error


def test_api_validation_unknown_pairs_and_recomputed_count_mismatch(prepared):
    catalog = load(prepared)
    app = FastAPI()
    app.include_router(create_medication_router(catalog))
    with TestClient(app) as client:
        assert client.get("/api/medication-cohort").json()["state"] == "ready"
        assert client.get("/api/medication-cohort/pairs?limit=1&page=2").json()["total"] == 3
        assert client.get("/api/medication-cohort/pairs/absent").status_code == 404
        for query in ("page=0", "limit=101", "split=bad"):
            assert client.get("/api/medication-cohort/pairs?" + query).status_code == 422
        assert client.get("/api/medication-cohort/pairs/medpair:0/medications?source=prescriptions").status_code == 422
        assert client.get("/api/medication-cohort/pairs/medpair:1/medications").status_code == 503


def test_clinical_reader_cannot_return_another_patients_records(prepared):
    catalog = load(prepared)
    original = catalog.index.read_subject
    catalog.index.read_subject = lambda table, subject: ([emar("200", "wrong", "2180-01-02T05:00:00")]
                                                        if table == "hosp.emar" else original(table, subject))
    with pytest.raises(ValueError, match="patient ownership"):
        catalog.medications("medpair:0")


def test_patient_cache_count_and_byte_limits_do_not_truncate_records(prepared):
    catalog = load(prepared, patient_cache_count=2)
    with catalog.records_lock:
        for subject in ("100", "200", "300"):
            catalog._patient_events(subject)
    assert list(catalog.cache) == ["200", "300"]
    assert catalog.medications("medpair:0")["unfiltered_total"] == 3
    catalog = load(prepared, patient_cache_bytes=1)
    assert catalog.medications("medpair:0")["unfiltered_total"] == 3
    assert not catalog.cache


def test_missing_dataset_api_reports_error(tmp_path):
    catalog = MedicationCohort(tmp_path)
    catalog._load()
    app = FastAPI()
    app.include_router(create_medication_router(catalog))
    with TestClient(app) as client:
        assert client.get("/api/medication-cohort").json()["state"] == "error"
        assert client.get("/api/medication-cohort/pairs").status_code == 503


def test_selection_must_share_the_atlas_image_source_root(prepared, tmp_path):
    catalog = MedicationCohort(prepared[0], expected_cxr_root=tmp_path / "different-cxr")
    catalog._load()
    assert catalog.state == "error" and "Atlas image source root" in catalog.error
    assert catalog.table is None
    catalog = load(prepared, expected_cxr_root=prepared[1])
    assert catalog.detail("medpair:0")["endpoints"]["source"]["report_status"] == "present"
    # Binding an already loaded injected catalog must also reject a different store.
    catalog.bind_cxr_root(tmp_path / "different-cxr")
    assert catalog.status()["state"] == "error"
    with pytest.raises(RuntimeError, match="Atlas image source root"):
        catalog.detail("medpair:0")
