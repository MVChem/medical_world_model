"""Annotation identity, source-only links, and cross-task holdout regression tests."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mimic_atlas.data_processing.supervision import export_supervision


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_manifests(legacy):
    original = legacy / "current/observations.jsonl"
    dense = legacy / "dense/observations.jsonl"
    (legacy / "current/manifest.json").write_text(json.dumps({
        "observations_sha256": _fingerprint(original),
        "valid_count": len(original.read_text().splitlines()),
    }))
    (legacy / "current/segmentation.json").write_text(json.dumps({
        "teacher": "fixture", "observations_sha256": _fingerprint(original),
        "organs": ["right lung", "left lung", "heart"],
    }))
    (legacy / "dense/manifest.json").write_text(json.dumps({
        "cohort_sha256": _fingerprint(dense), "images": len(dense.read_text().splitlines()),
    }))


@pytest.fixture
def inputs(tmp_path):
    legacy, cxr, vqa = (tmp_path / part for part in ("legacy", "source/MIMIC_CXR/files", "vqa"))
    cxr.mkdir(parents=True)
    vqa.mkdir()
    images = [f"p00/p0000000{i}/s0/image{i}.jpg" for i in range(4)]
    original, dense = [], []
    for index, image in enumerate(images):
        source = cxr / image
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"original source; no image decoding needed")
        original.append({"index": index, "id": f"image{index}", "subject_id": str(index),
                         "image": f"/previous/tree/MIMIC_CXR/files/{image}"})
        if index < 3:
            dense.append({**original[-1], "split": ["train", "validate", "test"][index],
                          "kind": "mimic", "tasks": ["segmentation"],
                          "old_index": index, "box": [0, 12, 512, 488]})
    montgomery = legacy / "dense/montgomery"
    human_image = "CXR_png/example.png"
    masks = ["ManualMask/leftMask/example.png", "ManualMask/rightMask/example.png"]
    for name in [human_image, *masks]:
        path = montgomery / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"original human supervision")
    dense.append({"index": 3, "id": "human:example", "subject_id": "human:example",
                  "split": "human_test", "kind": "montgomery", "tasks": ["segmentation"],
                  "box": [0, 30, 512, 452],
                  "image": f"/previous/tree/montgomery/{human_image}",
                  "masks": [f"/previous/tree/montgomery/{mask}" for mask in masks]})
    _jsonl(legacy / "current/observations.jsonl", original)
    _jsonl(legacy / "dense/observations.jsonl", dense)
    _refresh_manifests(legacy)
    targets = np.lib.format.open_memmap(legacy / "current/seg_probs.npy", mode="w+",
                                      shape=(len(original), 3, 256, 256), dtype=np.float16)
    del targets
    for name, patients in (("train", [0, 1, 3]), ("valid", [2]), ("test", [0])):
        (vqa / f"{name}.json").write_text(json.dumps([
            {"idx": position, "subject_id": str(patient), "image_path": images[patient]}
            for position, patient in enumerate(patients)
        ]))
    return {"output": tmp_path / "staging", "legacy_root": legacy,
            "vqa_root": vqa, "cxr_root": cxr}


def test_links_preserve_annotation_identity_and_holdout_priority(inputs):
    result = export_supervision(**inputs)
    output = inputs["output"]
    rows = [json.loads(line) for line in (output / "segmentation.jsonl").read_text().splitlines()]
    assert [(row["id"], row["split"]) for row in rows] == [
        ("image0", "train"), ("image1", "validate"), ("image2", "test"),
        ("human:example", "human_test"),
    ]
    assert [row["target_index"] for row in rows[:3]] == [0, 1, 2]
    assert rows[0]["kind"] == "cxas"
    assert rows[0]["box"] == [0, 12, 512, 488]
    assert rows[3]["masks"] == ["segmentation/montgomery/ManualMask/leftMask/example.png",
                                "segmentation/montgomery/ManualMask/rightMask/example.png"]
    assert all(not Path(row["image"]).is_absolute() for row in rows)
    assert result["holdouts"] == {
        "0": "test", "1": "validate", "2": "test", "3": "train", "human:example": "human_test",
    }
    segmentation = result["summary"]["segmentation"]
    assert segmentation["counts"] == {"train": 1, "validate": 1, "test": 1, "human_test": 1}
    assert segmentation["provenance"]["cxas_array"]["shape"] == [4, 3, 256, 256]
    assert result["summary"]["vqa"]["counts"] == {"train": 3, "validate": 1, "test": 1}
    # No historical input arrays/features were needed, and links survive publish.
    published = output.with_name("published")
    output.rename(published)
    for relative in ("segmentation/cxas_probs.npy", "segmentation/montgomery", "vqa"):
        assert (published / relative).is_symlink()
        assert (published / relative).exists()
    assert (published / rows[3]["image"]).is_file()


@pytest.mark.parametrize("field,value,match", [
    ("old_index", -1, "outside"),
    ("old_index", 1, "identity mismatch"),
    ("subject_id", "different", "identity mismatch"),
    ("image", "/previous/tree/MIMIC_CXR/files/p00/p00000001/s0/image1.jpg", "source image mismatch"),
    ("box", [0, 0, 513, 512], "ROI"),
])
def test_rejects_corrupt_annotation_mapping(inputs, field, value, match):
    path = inputs["legacy_root"] / "dense/observations.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0][field] = value
    _jsonl(path, rows)
    _refresh_manifests(inputs["legacy_root"])
    with pytest.raises(ValueError, match=match):
        export_supervision(**inputs)
    assert not inputs["output"].exists()


def test_rejects_changed_metadata_fingerprint(inputs):
    path = inputs["legacy_root"] / "dense/observations.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        export_supervision(**inputs)


@pytest.mark.parametrize("relative", ["current/seg_probs.npy",
                                     "dense/montgomery/ManualMask/rightMask/example.png"])
def test_missing_annotations_fail_explicitly(inputs, relative):
    (inputs["legacy_root"] / relative).unlink()
    with pytest.raises(FileNotFoundError, match="Missing"):
        export_supervision(**inputs)


def test_missing_vqa_split_fails_explicitly(inputs):
    (inputs["vqa_root"] / "valid.json").unlink()
    with pytest.raises(FileNotFoundError, match="CXR-VQA split"):
        export_supervision(**inputs)
    assert not inputs["output"].exists()


def test_rejects_wrong_annotation_array_shape(inputs):
    np.save(inputs["legacy_root"] / "current/seg_probs.npy", np.zeros((1, 3, 256, 256), np.float16))
    with pytest.raises(ValueError, match="shape/dtype"):
        export_supervision(**inputs)


def test_refuses_to_replace_existing_dataset_rows(inputs):
    inputs["output"].mkdir()
    original = inputs["output"] / "segmentation.jsonl"
    original.write_text("existing data\n")
    with pytest.raises(FileExistsError):
        export_supervision(**inputs)
    assert original.read_text() == "existing data\n"
