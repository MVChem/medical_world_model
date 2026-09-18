"""Whole-cohort joins and network payload boundaries, using synthetic sources."""

# ruff: noqa: F811 -- pytest fixture imports are intentionally shadowed.

import csv
import io
import time
from unittest.mock import patch

from fastapi.testclient import TestClient

from mimic_atlas import build_mimic_transitions as cxr
from mimic_atlas import link_mimic_iv_context as iv
from mimic_atlas.app import create_app
from mimic_atlas.tests.test_atlas import await_patient, sources, table  # noqa: F401


def await_cohort(client):
    for _ in range(200):
        index = client.get("/api/cohort").json()
        assert index["state"] != "error", index
        if index["state"] == "ready":
            return index
        time.sleep(0.01)
    raise AssertionError("Cohort did not finish")


def test_full_union_does_not_load_patient_reports_images_or_big_tables(
    sources,
    tmp_path,
):
    _, config = sources
    table(
        config.iv_root / "hosp/patients.csv.gz",
        ["subject_id", "gender", "anchor_age"],
        [
            {"subject_id": "10000001", "gender": "F", "anchor_age": "44"},
            {"subject_id": "10000003", "gender": "M", "anchor_age": "52"},
        ],
    )
    before = set(tmp_path.rglob("*"))
    with (
        patch.object(
            cxr,
            "read_report",
            side_effect=AssertionError("no report previews in index"),
        ),
        patch.object(
            iv,
            "load_iv_tables",
            side_effect=AssertionError("no event scans on overview"),
        ),
        TestClient(create_app(config)) as client,
    ):
        data = await_cohort(client)
        assert data["counts"]["patients"] == 3
        assert data["counts"]["matched_patients"] == 1
        assert data["counts"]["cxr_only"] == 1
        assert data["counts"]["iv_only"] == 1
        assert data["counts"]["pairs"] == data["counts"]["linked_pairs"] == 1
        assert data["study_linkage"] == {"unique": 2, "unmatched": 3}
        assert client.app.state.store.patients == {}
        assert client.app.state.store.extended.states == {}
        for scope, expected in [
            ("matched", "10000001"),
            ("cxr_only", "10000002"),
            ("iv_only", "10000003"),
        ]:
            result = client.get("/api/patients", params={"coverage": scope}).json()
            assert result["total"] == 1 and result["rows"][0]["subject_id"] == expected
        first = client.get("/api/patients?limit=2").json()
        second = client.get("/api/patients?limit=2&page=2").json()
        assert first["total"] == second["total"] == 3
        assert len(first["rows"]) == 2 and len(second["rows"]) == 1
        assert (
            client.get("/api/patients?split=none").json()["rows"][0]["subject_id"]
            == "10000003"
        )
        assert client.get("/api/patients?paired=linked").json()["total"] == 1
        assert client.get("/api/patients?coverage=bad").status_code == 422
        pairs = client.get("/api/cohort/pairs?linkage=unique").json()
        assert pairs["total"] == 1 and pairs["rows"][0]["hadm_id"] == "20000001"
        assert pairs["rows"][0]["source_study_id"] == "s50000001"
        csv_rows = list(
            csv.DictReader(
                io.StringIO(
                    client.get("/api/cohort/patients.csv?coverage=iv_only").text
                )
            )
        )
        assert len(csv_rows) == 1 and csv_rows[0]["subject_id"] == "10000003"
        assert (
            len(
                list(
                    csv.DictReader(
                        io.StringIO(
                            client.get("/api/cohort/pairs.csv?linkage=unique").text
                        )
                    )
                )
            )
            == 1
        )
    assert set(tmp_path.rglob("*")) == before


def test_ambiguous_admissions_not_counted_as_training_link(sources):
    _, config = sources
    table(
        config.iv_root / "hosp/admissions.csv.gz",
        ["subject_id", "hadm_id", "admittime", "dischtime"],
        [
            {
                "subject_id": "10000001",
                "hadm_id": str(i),
                "admittime": "2180-01-01 00:00:00",
                "dischtime": "2180-01-04 00:00:00",
            }
            for i in [1, 2]
        ],
    )
    with TestClient(create_app(config)) as client:
        index = await_cohort(client)
        assert index["counts"]["pairs"] == 1 and index["counts"]["linked_pairs"] == 0
        rows = client.get("/api/cohort/pairs?linkage=ambiguous").json()["rows"]
        assert rows[0]["candidate_hadm_ids"] == ["1", "2"] and rows[0]["hadm_id"] == ""
        assert client.get("/api/patients?paired=linked").json()["total"] == 0


def test_compact_catalog_selection_and_iv_only_patient(sources):
    _, config = sources
    table(
        config.iv_root / "hosp/patients.csv.gz",
        ["subject_id"],
        [{"subject_id": "10000003"}],
    )
    with TestClient(create_app(config)) as client:
        await_cohort(client)
        full = await_patient(client)
        compact = client.get("/api/patients/10000001?compact=true").json()
        assert (
            "report" not in compact["studies"][0]
            and "labels" not in compact["studies"][0]
        )
        assert (
            "report"
            not in compact["transitions"][0]["mimic_cxr_transition"]["current_state"]
        )
        selected = client.get(
            "/api/patients/10000001/selection",
            params={"transition": full["preferred_transition"]},
        ).json()
        assert len(selected["studies"]) == 2 and selected["studies"][0]["report"]
        assert selected["transition"]["linkage_status"] == "unique_common_admission"
        assert (
            client.get("/api/patients/10000001/selection?study=s50000001").json()[
                "transition"
            ]
            is None
        )
        assert (
            client.get("/api/patients/10000001/selection?study=s00000000").status_code
            == 404
        )
        assert (
            client.get("/api/patients/10000001/clinical-status").json()["state"]
            == "ready"
        )
        raw = client.get("/api/patients/10000001/context/admissions?limit=1").json()
        assert raw["total"] == 1 and raw["rows"][0]["hadm_id"] == "20000001"
        exported = client.get("/api/patients/10000001/context/admissions?download=true")
        assert list(csv.DictReader(io.StringIO(exported.text))) == raw["rows"]
        assert (
            client.get("/api/patients/10000001/context/inputevents").status_code == 409
        )
        assert client.get("/api/patients/10000001/context/unknown").status_code == 422
        webp = client.get("/api/images/50000001-image?size=128&format=webp&quality=60")
        assert webp.status_code == 200 and webp.headers["content-type"] == "image/webp"
        iv_only = await_patient(client, "10000003")
        assert (
            iv_only["studies"] == []
            and iv_only["transitions"] == []
            and iv_only["split"] is None
        )
        assert client.get("/api/patients/10000003/selection").json()["studies"] == []
        assert len(iv_only["extended_tables"]) == 17
