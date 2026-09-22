"""Original NIfTI orientation and discrete supervised channel alignment."""
import nibabel as nib
import numpy as np
import pytest
import torch

from medworld.datasets.mri_pixels import mri_source_canvas, mri_target


@pytest.fixture
def record(tmp_path):
    image = np.ones((4, 4, 3), dtype=np.float32)
    image[0, 1, 1] = 3
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[0, 1, 1] = 3
    affine = np.diag([-1., 1., 1., 1.])
    for name, values in (("image", image), ("mask", mask)):
        nib.save(nib.Nifti1Image(values, affine), tmp_path / f"{name}.nii.gz")
    return {"image": str(tmp_path / "image.nii.gz"), "mask_file": str(tmp_path / "mask.nii.gz"),
            "box": [0, 0, 512, 512], "slice_index": 1, "canonical_shape": [4, 4, 3],
            "normalization": [1., 3.], "channels": [2, 3, 4, 5]}


def test_canonical_orientation_matches_source_and_discrete_targets(record):
    source = mri_source_canvas(record)
    target = mri_target(record)
    assert source.shape == (1, 512, 512)
    assert target.shape == (6, 256, 256)
    # Original x=0 becomes canonical x=3; y=1 displays on row2 of4.
    assert source[0, 320, 448] > .9
    assert target[4, 160, 224] == 1
    assert target[4].sum() == 64 * 64
    assert target[:2].sum() == 0
    assert torch.all((target == 0) | (target == 1))


def test_source_image_decoding_does_not_open_a_mask(record):
    record["mask_file"] = "/missing/mask.nii.gz"
    assert mri_source_canvas(record).shape == (1, 512, 512)


def test_empty_tumor_slice_remains_supervised(record):
    record["slice_index"] = 0
    assert mri_target(record).sum() == 0


def test_shape_and_geometry_mismatch_are_rejected(record):
    record["canonical_shape"] = [4, 4, 4]
    with pytest.raises(ValueError, match="shape"):
        mri_source_canvas(record)
    record["canonical_shape"] = [4, 4, 3]
    affine = np.diag([-1., 1., 1., 1.])
    affine[0, 3] = 20
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 3), dtype=np.uint8), affine), record["mask_file"])
    with pytest.raises(ValueError, match="grids"):
        mri_target(record)


def test_original_file_hash_mismatch_is_rejected_at_decode(record):
    record["t1ce_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        mri_source_canvas(record)
    record.pop("t1ce_sha256")
    record["mask_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        mri_target(record)
