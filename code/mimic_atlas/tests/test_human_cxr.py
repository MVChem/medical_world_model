"""Human mask identity, geometry, provenance, download, and holdout regressions."""
import csv
import hashlib
import io
from pathlib import Path
import zipfile

from PIL import Image
import pytest

from mimic_atlas.data_processing import human_cxr


def _checksums(root):
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            rows.append(hashlib.sha256(path.read_bytes()).hexdigest() + " " + str(path.relative_to(root)))
    (root / "SHA256SUMS.txt").write_text("\n".join(rows) + "\n")


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as source:
        writer = csv.DictWriter(source, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def inputs(tmp_path):
    annotation, cxr = tmp_path / "human", tmp_path / "cxr"
    links, splits = [], []
    for index in range(200):
        subject, study, identity = str(10000000 + index), str(50000000 + index), f"fixture-{index}"
        relative = Path("files/p10") / ("p" + subject) / ("s" + study) / (identity + ".jpg")
        image = cxr / relative
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (8, 10), 100).save(image)
        for name in ("lungs", "heart"):
            path = annotation / "mimic_masks" / name / f"{index:03d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            mask = Image.new("L", (8, 10))
            mask.paste(255, (1, 2, 7, 9))
            mask.save(path)
        links.append({"subject_id": subject, "study_id": study,
                      "MIMIC-CXR_path": str(relative.with_suffix(".dcm")),
                      "heart_mask_path": f"heart/{index:03d}.jpg", "lungs_mask_path": f"lungs/{index:03d}.jpg"})
        splits.append({"subject_id": subject, "study_id": study, "dicom_id": identity,
                       "split": "validate" if index == 198 else "test" if index == 199 else "train"})
    _csv(annotation / "mimic_masks/MIMIC_links.csv", links)
    _csv(cxr / "mimic-cxr-2.0.0-split.csv", splits)
    _checksums(annotation)
    return {"output": tmp_path / "prepared", "annotation_root": annotation, "cxr_root": cxr}


def test_export_preserves_all_images_and_existing_holdouts(inputs):
    holds = {"10000000": "test", "10000001": "validate", "unrelated": "test"}
    result = human_cxr.export_human_cxr(**inputs, holdouts=holds)
    rows = result["records"]
    assert len(rows) == 200
    assert result["summary"]["patient_counts"] == {"test": 30, "validate": 30, "train": 140}
    assert result["holdouts"]["unrelated"] == "test"
    for patient in ("10000000", "10000199"):
        assert result["holdouts"][patient] == "test"
    for patient in ("10000001", "10000198"):
        assert result["holdouts"][patient] == "validate"
    assert holds == {"10000000": "test", "10000001": "validate", "unrelated": "test"}
    for row in rows:
        assert row["kind"] == "human_cxr"
        assert row["channels"] == [0, 1]
        assert row["target_names"] == ["lungs", "heart"]
        assert row["annotation"] == "expert_reviewed"
        assert row["annotation_source"] == "human_reviewed"
        assert row["mask_geometry"] == {"native_size": [10, 8], "transform": "identity"}
        assert all(value % 2 == 0 for value in row["box"])
        assert all((inputs["output"] / path).is_file() for path in row["masks"])
        assert row["split"] == result["holdouts"][row["subject_id"]]
    link = inputs["output"] / "segmentation/heart_lung_human"
    assert link.is_symlink()
    published = inputs["output"].with_name("published")
    inputs["output"].rename(published)
    assert (published / rows[0]["masks"][0]).is_file()


def test_split_determinism_and_oversubscribed_existing_holdouts():
    patients = [str(index) for index in range(100)]
    holds = {patient: "test" for patient in patients[:60]}
    first = human_cxr.assign_patient_splits(patients, holds)
    second = human_cxr.assign_patient_splits(reversed(patients), holds)
    assert first == second
    assert sum(split == "test" for split in first.values()) == 60
    assert sum(split == "validate" for split in first.values()) == 15
    assert sum(split == "train" for split in first.values()) == 25


def test_rejects_mask_checksum_corruption(inputs):
    (inputs["annotation_root"] / "mimic_masks/lungs/000.png").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        human_cxr.export_human_cxr(**inputs)


def test_rejects_mask_image_geometry_mismatch(inputs):
    path = inputs["annotation_root"] / "mimic_masks/lungs/000.png"
    Image.new("L", (10, 8)).save(path)
    _checksums(inputs["annotation_root"])
    with pytest.raises(ValueError, match="geometry"):
        human_cxr.export_human_cxr(**inputs)


def test_rejects_grayscale_pseudo_mask(inputs):
    path = inputs["annotation_root"] / "mimic_masks/lungs/000.png"
    image = Image.new("L", (8, 10), 1)
    image.paste(255, (1, 2, 7, 9))
    image.save(path)
    _checksums(inputs["annotation_root"])
    with pytest.raises(ValueError, match="binary"):
        human_cxr.export_human_cxr(**inputs)


def test_rejects_wrong_patient_image_mapping(inputs):
    path = inputs["annotation_root"] / "mimic_masks/MIMIC_links.csv"
    rows = list(human_cxr._rows(path))
    rows[0]["subject_id"] = rows[1]["subject_id"]
    _csv(path, rows)
    _checksums(inputs["annotation_root"])
    with pytest.raises(ValueError, match="patient/study identity"):
        human_cxr.export_human_cxr(**inputs)


def test_download_checksums_and_reuses_existing_files(tmp_path, monkeypatch):
    source = tmp_path / "release"
    source.mkdir()
    (source / "LICENSE.txt").write_text("fixture attribution license")
    _checksums(source)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        for path in source.iterdir():
            archive.write(path, "publisher-release-long-name/" + path.name)
    calls = []

    def download(url, timeout):
        calls.append(url)
        return io.BytesIO(payload.getvalue())

    monkeypatch.setattr(human_cxr, "urlopen", download)
    destination = tmp_path / "downloaded"
    first = human_cxr.ensure_sources(destination)
    second = human_cxr.ensure_sources(destination)
    assert first == second
    assert first["files_verified"] == 1
    assert calls == [human_cxr.ZIP_URL]
    assert (destination / "LICENSE.txt").read_bytes() == (source / "LICENSE.txt").read_bytes()


def test_download_rejects_archive_traversal(tmp_path, monkeypatch):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("release/SHA256SUMS.txt", "empty")
        archive.writestr("release/../../escape", "bad")
    monkeypatch.setattr(human_cxr, "urlopen", lambda *args, **kwargs: io.BytesIO(payload.getvalue()))
    with pytest.raises(ValueError, match="Unsafe"):
        human_cxr.ensure_sources(tmp_path / "downloaded")
    assert not (tmp_path / "escape").exists()
