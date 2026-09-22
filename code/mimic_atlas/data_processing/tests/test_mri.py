"""Patient isolation, source-only slice selection and MRI target validation."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from mimic_atlas.data_processing.mri import _canonical_header, export_mri, patient_splits


def save(path, values, affine=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(values, np.eye(4) if affine is None else affine), path)


@pytest.fixture
def sources(tmp_path):
    image = np.zeros((6, 8, 5), dtype=np.float32)
    image[:, :, 1:4] = np.arange(1, 7)[:, None, None]
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[1:3, 2:4, 2] = 3
    ucsf, mu = tmp_path / "ucsf", tmp_path / "mu"
    for index in range(10):
        pid = str(100000 + index)
        for visit in (1, 2):
            for suffix in ("t1", "t1ce", "t2", "flair"):
                save(ucsf / pid / f"{pid}_time{visit}_{suffix}.nii.gz", image)
            save(ucsf / pid / f"{pid}_time{visit}_seg.nii.gz", mask)
        # Additional longitudinal change labels must not become visit targets.
        save(ucsf / pid / f"{pid}_flair_subtraction_seg.nii.gz", mask)
        pid = f"PatientID_{index:04d}"
        for visit in (1, 2):
            stem = mu / pid / f"Timepoint_{visit}" / f"{pid}_Timepoint_{visit}_"
            for suffix in ("brain_t1n", "brain_t1c", "brain_t2w", "brain_t2f"):
                save(Path(str(stem) + suffix + ".nii.gz"), image)
            if (index, visit) != (0, 2):
                save(Path(str(stem) + "tumorMask.nii.gz"), mask)
    return tmp_path / "out", ucsf, mu


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_all_reviewed_volumes_patient_split_and_original_links(sources):
    out, ucsf, mu = sources
    result = export_mri(out, ucsf, mu, workers=2)
    rows = read_rows(out / "mri_segmentation.jsonl")
    volumes = read_rows(out / "mri_volumes.jsonl")
    assert len(volumes) == 40
    assert len(rows) == 39 * 3
    assert sum(result["volumes"].values()) == 39
    assert sum(result["counts"].values()) == len(rows)
    assert (out / "mri/ucsf").is_symlink()
    assert (out / "mri/mu").is_symlink()
    assert sum(v["annotation"] == "missing" for v in volumes) == 1
    by_patient = {}
    for row in rows:
        by_patient.setdefault(row["subject_id"], set()).add(row["split"])
        assert row["slice_index"] in [1, 2, 3]
        assert row["channels"] == [2, 3, 4, 5]
        assert row["annotation"] == "expert_reviewed"
        assert (out / row["image"]).is_file()
        assert (out / row["mask_file"]).is_file()
    assert len(by_patient) == 20 and all(len(splits) == 1 for splits in by_patient.values())
    for stats in result["summary"]["datasets"].values():
        assert stats["patient_counts"] == {"train": 7, "validate": 1, "test": 2}
    assert all(set(volume["sequences"]) == {"t1", "t1ce", "t2", "flair"} for volume in volumes)
    assert all(len(row["t1ce_sha256"]) == len(row["mask_sha256"]) == 64 for row in rows)
    # No rendered pixels or converted masks are materialized in the output.
    assert {p.name for p in out.iterdir()} == {"mri", "mri_segmentation.jsonl", "mri_volumes.jsonl"}


def test_slice_selection_and_intensity_windows_do_not_depend_on_masks(sources):
    out, ucsf, mu = sources
    export_mri(out, ucsf, mu, axial_stride=2)
    before = read_rows(out / "mri_segmentation.jsonl")
    for path in ucsf.glob("*/*_seg.nii.gz"):
        changed_mask = np.zeros((6, 8, 5), dtype=np.uint8)
        changed_mask[:, :, 1:4] = 4
        save(path, changed_mask)
    after_out = out.with_name("after")
    export_mri(after_out, ucsf, mu, axial_stride=2)
    after = read_rows(after_out / "mri_segmentation.jsonl")
    assert [{key: value for key, value in row.items() if key != "mask_sha256"} for row in before] == [
        {key: value for key, value in row.items() if key != "mask_sha256"} for row in after]
    assert {row["slice_index"] for row in after} == {1, 3}


def test_patient_split_reproducible_without_relying_on_input_order():
    patients = [str(i) for i in range(100)]
    expected = patient_splits(patients, "mu", seed=42)
    assert expected == patient_splits(reversed(patients), "mu", seed=42)
    assert expected != patient_splits(patients, "mu", seed=7)


@pytest.mark.parametrize("problem", ["labels", "geometry"])
def test_rejects_invalid_original_masks(sources, problem):
    out, ucsf, mu = sources
    target = next(ucsf.glob("*/*_time1_seg.nii.gz"))
    values = np.zeros((6, 8, 5), dtype=np.float32)
    affine = np.eye(4)
    if problem == "labels":
        values[1, 1, 1] = .5
    else:
        affine[0, 3] = 12
    save(target, values, affine)
    with pytest.raises(ValueError, match="labels|grid"):
        export_mri(out, ucsf, mu)


def test_refuses_to_replace_prepared_mri(sources):
    out, ucsf, mu = sources
    export_mri(out, ucsf, mu)
    before = (out / "mri_segmentation.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        export_mri(out, ucsf, mu)
    assert (out / "mri_segmentation.jsonl").read_bytes() == before


def test_foreground_outside_image_selected_extent_is_not_silently_dropped(sources):
    out, ucsf, mu = sources
    target = next(ucsf.glob("*/*_time1_seg.nii.gz"))
    values = np.zeros((6, 8, 5), dtype=np.uint8)
    values[1, 1, 0] = 3
    save(target, values)
    with pytest.raises(ValueError, match="outside image-defined axial extent"):
        export_mri(out, ucsf, mu)


def test_header_only_canonical_geometry_matches_actual_orientation(tmp_path):
    path = tmp_path / "permuted.nii.gz"
    affine = np.array([[0, 0, -2, 17], [3, 0, 0, -11], [0, -4, 0, 23], [0, 0, 0, 1.]])
    save(path, np.arange(24, dtype=np.float32).reshape(2, 3, 4), affine)
    shape, canonical_affine = _canonical_header(path)
    actual = nib.as_closest_canonical(nib.load(path))
    assert shape == actual.shape
    np.testing.assert_allclose(canonical_affine, actual.affine)
