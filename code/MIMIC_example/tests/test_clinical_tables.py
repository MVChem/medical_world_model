"""Verify raw CSV fidelity, temporal joins, unit isolation and complete exports."""

from __future__ import annotations

import csv
import io
import json
import time
from dataclasses import replace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from MIMIC_example import clinical_tables as clinical
from MIMIC_example.app import create_app
from MIMIC_example.tests.test_atlas import await_patient, sources, table  # noqa: F401


def test_streaming_csv_keeps_multiline_values_and_unsorted_subjects(tmp_path):
    path = tmp_path / "events.csv.gz"
    fields = ["row_id", "subject_id", "value", "comments"]
    rows = [
        {
            "row_id": "0001",
            "subject_id": "9",
            "value": "001.20",
            "comments": 'text, "quoted"\nnext line',
        },
        *[
            {
                "row_id": str(i),
                "subject_id": "8",
                "value": "text",
                "comments": "x" * 100,
            }
            for i in range(2, 40)
        ],
        {"row_id": "0040", "subject_id": "9", "value": "<5", "comments": "NULL"},
    ]
    table(path, fields, rows)
    progress = []
    selected, actual_fields, scanned = clinical.scan_subjects(
        path, {"9"}, lambda n, b: progress.append((n, b)), block_size=1024
    )
    assert selected["9"] == [rows[0], rows[-1]]
    assert actual_fields == fields and scanned == len(rows)
    assert len(progress) > 1 and progress[-1][0] == len(rows)
    assert list(selected) == ["9"]


def test_units_censored_values_date_precision_and_unassigned_admissions():
    base = {
        "subject_id": "1",
        "hadm_id": "A",
        "itemid": "test",
        "charttime": "2180-01-01 14:00:00",
        "storetime": "2180-01-02 10:00:00",
        "value": "5",
        "valuenum": "5",
        "valueuom": "mg/dL",
    }
    rows = [
        base,
        {**base, "hadm_id": "", "valueuom": "mmol/L"},
        {**base, "value": "<5"},
        {**base, "valuenum": "NaN"},
        {**base, "charttime": "bad timestamp"},
        {**base, "hadm_id": "B"},
    ]
    events = clinical.normalize(
        clinical.TABLES["labevents"], rows, {"test": {"label": "Measurement"}}
    )
    assert (
        len(clinical.summarize(events)) == 2
    )  # Same item, different units remain distinct.
    assert len(clinical.filter_events(events, scope="admission", hadm_id="A")) == 4
    window = clinical.filter_events(
        events, scope="window", start="2180-01-01 13:00:00", end="2180-01-01 15:00:00"
    )
    assert len(window) == 5 and sum(r["hadm_id"] is None for r in window) == 1
    assert (
        window[0]["storetime"] == "2180-01-02 10:00:00"
    )  # No false availability claim at charttime.
    assert next(r for r in events if r["value"] == "<5")["numeric_value"] is None
    assert (
        next(r for r in events if r["raw"]["valuenum"] == "NaN")["numeric_value"]
        is None
    )
    assert json.loads(json.dumps(events, allow_nan=False))
    omr = clinical.normalize(
        clinical.TABLES["omr"],
        [
            {
                "subject_id": "1",
                "chartdate": "2180-01-01",
                "result_name": "Weight (Lbs)",
                "result_value": "100",
            }
        ],
        {},
    )
    assert (
        len(
            clinical.filter_events(
                omr,
                scope="window",
                start="2180-01-01 13:00:00",
                end="2180-01-01 15:00:00",
            )
        )
        == 1
    )
    assert omr[0]["time_precision"] == "date" and clinical.summarize(omr) == []
    with pytest.raises(ValueError):
        clinical.filter_events(events, scope="admission")
    with pytest.raises(ValueError):
        clinical.filter_events(
            events, scope="window", start="2180-01-02", end="2180-01-01"
        )


def test_child_records_join_only_unique_parent_without_losing_or_summing_rows():
    parents = [
        {
            "subject_id": "1",
            "emar_id": "1-1",
            "hadm_id": "A",
            "charttime": "2180-01-01 12:00:00",
        },
        {
            "subject_id": "2",
            "emar_id": "1-1",
            "hadm_id": "WRONG",
            "charttime": "2180-01-01 12:00:00",
        },
    ]
    rows = [
        {
            "subject_id": "1",
            "emar_id": "1-1",
            "dose_given": "2",
            "dose_given_unit": "mg",
        },
        {
            "subject_id": "1",
            "emar_id": "1-1",
            "dose_given": "3",
            "dose_given_unit": "mg",
        },
        {"subject_id": "1", "emar_id": "missing", "dose_given": "4"},
    ]
    result = clinical.normalize(clinical.TABLES["emar_detail"], rows, {}, parents)
    assert len(result) == 3
    assert [r["value"] for r in result] == ["2", "3", "4"]
    assert result[0]["hadm_id"] == "A" and result[0]["time_source"] == "emar.charttime"
    assert result[2]["parent_link"] == "unmatched" and result[2]["time"] is None
    ambiguous = clinical.normalize(
        clinical.TABLES["emar_detail"], rows[:1], {}, parents + [parents[0]]
    )[0]
    assert (
        ambiguous["parent_link"] == "ambiguous"
        and ambiguous["hadm_id"] is None
        and ambiguous["time"] is None
    )


def test_plot_sampling_retains_extrema_and_advertises_sampling():
    rows = [
        {
            "series_id": "x",
            "numeric_value": 9999 if i == 3333 else -9999 if i == 5000 else i % 10,
            "time": f"2180-01-{1 + i // 1000:02d}T12:00:00",
            "time_precision": "timestamp",
            "value": str(i),
            "unit": "u",
            "hadm_id": "a",
            "stay_id": "b",
            "flag": "",
            "reference_lower": "",
            "reference_upper": "",
            "storetime": None,
        }
        for i in range(8000)
    ]
    output = clinical.chart_points(rows, "x")
    assert (
        output["sampled"] and output["total"] == 8000 and len(output["points"]) <= 1500
    )
    values = [p["numeric_value"] for p in output["points"]]
    assert min(values) == -9999 and max(values) == 9999
    assert (
        output["points"][0]["value"] == "0" and output["points"][-1]["value"] == "7999"
    )


@pytest.fixture
def extended_sources(sources):  # noqa: F811 -- pytest injects the imported fixture.
    root, config = sources
    ir = config.iv_root
    pid = "10000001"
    for spec in clinical.SPECS:
        table(
            ir / spec.module / f"{spec.name}.csv.gz",
            ["subject_id", "hadm_id", "charttime"],
            [],
        )
    table(
        ir / "hosp/d_labitems.csv.gz",
        ["itemid", "label", "fluid", "category"],
        [
            {
                "itemid": "100",
                "label": "Test measurement",
                "fluid": "Blood",
                "category": "Chemistry",
            }
        ],
    )
    table(ir / "hosp/d_hcpcs.csv.gz", ["code", "long_description"], [])
    labs = [
        {
            "subject_id": pid,
            "hadm_id": "20000001" if i % 2 == 0 else "",
            "charttime": f"2180-01-01 {12 + i:02d}:30:00",
            "itemid": "100",
            "value": str(i),
            "valuenum": str(i),
            "valueuom": "u",
            "comments": "quoted, field\nwith </script>",
        }
        for i in range(6)
    ]
    table(ir / "hosp/labevents.csv.gz", list(labs[0]), labs)
    parent = {
        "subject_id": pid,
        "hadm_id": "20000001",
        "emar_id": "one",
        "charttime": "2180-01-01 13:00:00",
        "medication": "Synthetic",
        "event_txt": "Not Given",
    }
    table(ir / "hosp/emar.csv.gz", list(parent), [parent])
    detail = {
        "subject_id": pid,
        "emar_id": "one",
        "dose_due": "2",
        "dose_given": "0",
        "dose_given_unit": "mg",
    }
    table(ir / "hosp/emar_detail.csv.gz", list(detail), [detail])
    return root, replace(config, preload_tables=()), labs


def await_tables(client):
    for _ in range(300):
        manifest = client.get("/api/patients/10000001/tables").json()
        assert not any(r["state"] == "error" for r in manifest), manifest
        if all(r["state"] == "ready" for r in manifest):
            return manifest
        time.sleep(0.01)
    raise AssertionError("Clinical tables did not finish")


def test_all_table_api_scans_once_pages_without_loss_and_exports_offline(
    extended_sources, tmp_path
):
    _, config, labs = extended_sources
    before = set(tmp_path.rglob("*"))
    with (
        patch.object(clinical, "scan_subjects", wraps=clinical.scan_subjects) as scan,
        TestClient(create_app(config)) as client,
    ):
        patient = await_patient(client)
        base = "/api/patients/10000001/tables"
        assert all(r["state"] == "not_loaded" for r in client.get(base).json())
        assert client.get(base + "/labevents/export.csv").status_code == 409
        assert client.post(base, json={"tables": ["unknown"]}).status_code == 422
        assert client.post(base, json={"tables": []}).status_code == 422
        assert client.post(base, json={"tables": ["all"]}).status_code == 200
        assert client.post(base, json={"tables": ["all"]}).status_code == 200
        manifest = await_tables(client)
        assert len(manifest) == 17 and scan.call_count == 17
        assert client.get(base + "/labevents?scope=admission").status_code == 422
        assert (
            client.get(base + "/labevents?scope=window&start=bad&end=bad").status_code
            == 422
        )
        assert client.get(base + "/labevents?scope=patient&page=0").status_code == 422
        one = client.get(base + "/labevents?scope=patient&page=1&limit=2").json()
        two = client.get(base + "/labevents?scope=patient&page=2&limit=2").json()
        assert one["total"] == 6 and one["rows"][1]["raw"] == labs[1]
        assert two["rows"][0]["raw"] == labs[2]
        admission = client.get(
            base + "/labevents?scope=admission&hadm_id=20000001"
        ).json()
        assert admission["total"] == 3
        exported = client.get(base + "/labevents/export.csv?scope=patient")
        assert list(csv.DictReader(io.StringIO(exported.text))) == labs
        assert (
            scan.call_count == 17
        )  # All subsequent reads use existing process memory.
        child = client.get(base + "/emar_detail").json()["rows"][0]
        assert (
            child["time_source"] == "emar.charttime" and child["hadm_id"] == "20000001"
        )
        assert (
            child["value"] == "0"
        )  # Never replace an actual zero with a planned dose.
        html = client.get(
            "/api/export/10000001.html",
            params={"transition": patient["preferred_transition"]},
        )
        assert html.status_code == 200 and 'src="/static/' not in html.text
        assert "Test measurement" in html.text and "\\u003c/script>" in html.text
        assert "window.ClinicalExplorer" in html.text
    assert set(tmp_path.rglob("*")) == before


def test_missing_table_and_retry_after_source_error(tmp_path):
    service = clinical.ClinicalTables(tmp_path)
    try:
        service.request({"1"}, ["labevents"])
        assert service.manifest("1")[0]["state"] == "unavailable"
        table(
            tmp_path / "hosp/labevents.csv.gz",
            ["subject_id", "itemid"],
            [{"subject_id": "1", "itemid": "x"}],
        )
        service.request({"1"}, ["labevents"])
        service.futures["1", "labevents"].result(timeout=3)
        assert (
            service.manifest("1")[0]["state"] == "error"
        )  # Missing dictionary is explicit.
        table(
            tmp_path / "hosp/d_labitems.csv.gz",
            ["itemid", "label"],
            [{"itemid": "x", "label": "Recovered"}],
        )
        service.request({"1"}, ["labevents"])
        service.futures["1", "labevents"].result(timeout=3)
        assert service.manifest("1")[0]["state"] == "ready"
        assert service.query("1", "labevents")["rows"][0]["label"] == "Recovered"
    finally:
        service.close()
