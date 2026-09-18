"""Behavior checks for raw-data browsing, joins, exports and in-memory computation."""

from __future__ import annotations

import csv
import gzip
import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from MIMIC_example import build_mimic_transitions as cxr
from MIMIC_example import link_mimic_iv_context as iv
from MIMIC_example.app import create_app
from MIMIC_example.data import AtlasConfig
from MIMIC_example.setup_data import setup_links


def table(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def sources(tmp_path):
    root = tmp_path / "MIMIC"
    cr, ir = root / "MIMIC_CXR", root / "mimic-iv-3.1"
    metadata, splits, labels = [], [], []
    for pid, visits in [
        ("10000001", [("50000001", 1, "AP"), ("50000002", 2, "AP")]),
        (
            "10000002",
            [("50000003", 1, "AP"), ("50000004", 2, "LATERAL"), ("50000005", 3, "AP")],
        ),
    ]:
        for sid, day, view in visits:
            patient = cr / f"files/p{pid[:2]}/p{pid}"
            folder = patient / f"s{sid}"
            folder.mkdir(parents=True)
            patient.joinpath(f"s{sid}.txt").write_text(
                "FINDINGS: Test report.\nIMPRESSION: </script><b>test</b>"
            )
            image = sid + "-image"
            Image.new("L", (48, 64), 128).save(folder / f"{image}.jpg")
            metadata.append(
                {
                    "subject_id": pid,
                    "study_id": sid,
                    "dicom_id": image,
                    "StudyDate": f"2180010{day}",
                    "StudyTime": "120000",
                    "ViewPosition": view,
                    "Rows": "64",
                    "Columns": "48",
                }
            )
            splits.append({"subject_id": pid, "study_id": sid, "split": "train"})
            labels.append(
                {
                    "subject_id": pid,
                    "study_id": sid,
                    **{name: 1 if day == 1 else 0 for name in cxr.LABEL_COLUMNS},
                }
            )
    table(cr / "mimic-cxr-2.0.0-metadata.csv", list(metadata[0]), metadata)
    table(cr / "mimic-cxr-2.0.0-split.csv", list(splits[0]), splits)
    table(cr / "mimic-cxr-2.0.0-chexpert.csv", list(labels[0]), labels)
    table(
        ir / "hosp/admissions.csv.gz",
        ["subject_id", "hadm_id", "admittime", "dischtime"],
        [
            {
                "subject_id": "10000001",
                "hadm_id": "20000001",
                "admittime": "2180-01-01 00:00:00",
                "dischtime": "2180-01-03 00:00:00",
            }
        ],
    )
    for name in [
        "hosp/transfers",
        "hosp/diagnoses_icd",
        "hosp/procedures_icd",
        "icu/icustays",
        "icu/procedureevents",
    ]:
        table(ir / f"{name}.csv.gz", ["subject_id", "hadm_id"], [])
    for name in ["d_icd_diagnoses", "d_icd_procedures"]:
        table(ir / f"hosp/{name}.csv.gz", ["icd_code", "icd_version", "long_title"], [])
    table(
        ir / "icu/d_items.csv.gz",
        ["itemid", "label"],
        [{"itemid": "1", "label": "Test input"}],
    )
    table(
        ir / "icu/inputevents.csv.gz",
        [
            "subject_id",
            "hadm_id",
            "itemid",
            "starttime",
            "endtime",
            "amount",
            "amountuom",
        ],
        [
            {
                "subject_id": "10000001",
                "hadm_id": "20000001",
                "itemid": "1",
                "starttime": "2180-01-01 13:00:00",
                "endtime": "2180-01-01 15:00:00",
                "amount": "2",
                "amountuom": "mL",
            }
        ],
    )
    pairs = tmp_path / "pairs.json"
    pairs.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "subject_id": "10000001",
                        "source_study_id": "50000001",
                        "target_study_id": "50000002",
                        "category": "test",
                        "audit_note": "Synthetic fixture",
                    }
                ]
            }
        )
    )
    return root, AtlasConfig(cr, ir, pairs)


def await_patient(client, pid="10000001"):
    for _ in range(200):
        catalog = client.get("/api/catalog").json()
        assert catalog["state"] != "error", catalog
        if catalog["state"] == "ready":
            result = client.get(f"/api/patients/{pid}")
            if (
                result.status_code == 200
                and result.json()["clinical_status"]["state"] == "ready"
            ):
                return result.json()
        time.sleep(0.01)
    raise AssertionError("Patient did not finish loading")


def test_prepare_has_no_filesystem_side_effects_and_keeps_adjacency(sources, tmp_path):
    _, config = sources
    before = set(tmp_path.rglob("*"))
    build = cxr.BuildConfig(
        config.cxr_root,
        tmp_path / "must-not-exist",
        num_examples=None,
        one_per_patient=False,
        view="frontal",
        render_gallery=False,
        asset_mode="none",
    )
    result = cxr.prepare_dataset(build)
    assert len(result.packets) == 1
    assert result.packets[0]["patient_id"] == "p10000001"
    assert result.output_paths == {}
    assert set(tmp_path.rglob("*")) == before
    assert all(
        "future" not in key and "target" not in key for key in result.input_rows[0]
    )


def test_api_browsing_inputs_image_boundaries_and_offline_export(sources, tmp_path):
    _, config = sources
    before = set(tmp_path.rglob("*"))
    with TestClient(create_app(config)) as client:
        patient = await_patient(client)
        assert client.get("/api/catalog").json()["counts"] == {
            "patients": 2,
            "studies": 5,
            "images": 5,
            "featured_pairs": 1,
        }
        assert (
            client.get("/api/patients?q=p10000002&longitudinal=true").json()["total"]
            == 1
        )
        assert client.get("/api/patients?q=missing").json()["total"] == 0
        assert client.get("/api/patients?page=0").status_code == 422
        assert client.get("/api/patients/9999").status_code == 404
        assert client.get("/api/images/unknown").status_code == 404
        image = client.get("/api/images/50000001-image?size=128")
        assert (
            image.status_code == 200 and image.headers["content-type"] == "image/jpeg"
        )
        assert image.headers["cache-control"] == "no-store"
        assert client.get("/api/images/50000001-image?size=99999").status_code == 422
        pair = patient["transitions"][0]
        assert pair["linkage_status"] == "unique_common_admission"
        assert (
            pair["retrospective_context_for_audit_only"]["interval_icu_inputevents"]
            is None
        )
        html = client.get(
            f"/api/export/10000001.html?transition={pair['transition_id']}"
        )
        assert html.status_code == 200
        assert "data:image/jpeg;base64," in html.text
        assert 'src="/static/' not in html.text and 'href="/static/' not in html.text
        assert "\\u003c/script>" in html.text
        assert (
            client.get("/api/export/10000001.html?transition=unknown").status_code
            == 404
        )
        assert client.post("/api/patients/10000001/icu-inputs").status_code == 200
        upgraded = await_patient(client)
        assert upgraded["icu_inputs"] is True
        event = upgraded["transitions"][0]["retrospective_context_for_audit_only"][
            "interval_icu_inputevents"
        ][0]
        assert event["label"] == "Test input" and event["amount_unit"] == "mL"
        other = await_patient(client, "10000002")
        assert len(other["studies"]) == 3 and other["transitions"] == []
        assert other["studies"][1]["images"][0]["view"] == "LATERAL"
        assert other["studies"][0]["admission_link"]["status"] == "unmatched"
        # A metadata-registered symlink must not escape the configured CXR root.
        im = client.app.state.store.images["50000001-image"].image_path
        outside = tmp_path / "outside.jpg"
        outside.write_bytes(im.read_bytes())
        im.unlink()
        im.symlink_to(outside)
        assert client.get("/api/images/50000001-image").status_code == 404
        outside.unlink()
        im.unlink()
        Image.new("L", (48, 64), 128).save(im)
    assert set(tmp_path.rglob("*")) == before


def test_preloaded_iv_tables_keep_ambiguous_and_unmatched(sources):
    _, config = sources
    build = cxr.BuildConfig(
        config.cxr_root, Path("/unused"), num_examples=1, view="frontal"
    )
    packets = cxr.prepare_dataset(build).packets
    tables = iv.load_iv_tables(config.iv_root, {"10000001"}, include_icu_inputs=False)
    admission = tables.by_subject["10000001"]["admissions"][0]
    tables.by_subject["10000001"]["admissions"].append(
        {**admission, "hadm_id": "20000002"}
    )
    with patch.object(
        iv, "load_subject_rows", side_effect=AssertionError("must reuse memory")
    ):
        ambiguous = iv.link_packets(
            packets, config.iv_root, include_icu_inputs=False, tables=tables
        )[0]
        assert ambiguous["linkage_status"] == "ambiguous"
        assert ambiguous["candidate_hadm_ids"] == ["20000001", "20000002"]
        assert ambiguous["retrospective_context_for_audit_only"] is None
        tables.by_subject["10000001"]["admissions"] = []
        assert (
            iv.link_packets(
                packets, config.iv_root, include_icu_inputs=False, tables=tables
            )[0]["linkage_status"]
            == "unmatched"
        )


def test_setup_links_is_repeatable_and_never_replaces_existing_data(sources, tmp_path):
    root, _ = sources
    links = tmp_path / "links"
    setup_links(root, links)
    setup_links(root, links)
    assert (links / "MIMIC_CXR").resolve() == root / "MIMIC_CXR"
    (links / "MIMIC_CXR").unlink()
    (links / "MIMIC_CXR").mkdir()
    with pytest.raises(FileExistsError):
        setup_links(root, links)
    assert (links / "MIMIC_CXR").is_dir() and not (links / "MIMIC_CXR").is_symlink()
