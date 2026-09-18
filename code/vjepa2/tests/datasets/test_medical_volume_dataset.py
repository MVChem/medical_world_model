import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch
from app.vjepa_2_1.transforms import make_transforms
from src.datasets.medical_volume_dataset import MedicalVolumeDataset


class TestMedicalVolumeDataset(unittest.TestCase):
    def test_nifti_volume_uses_original_rgb_video_shape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            volume = np.arange(24 * 20 * 12, dtype=np.float32).reshape(24, 20, 12)
            volume_path = Path(temp_dir) / "sample.nii.gz"
            nib.save(nib.Nifti1Image(volume, np.eye(4)), volume_path)

            dataset = MedicalVolumeDataset(
                data_paths=[temp_dir],
                dataset_fpcs=[8],
                random_clip_sampling=False,
                transform=make_transforms(
                    random_horizontal_flip=False,
                    random_resize_aspect_ratio=(1.0, 1.0),
                    random_resize_scale=(1.0, 1.0),
                    crop_size=16,
                ),
            )
            clips, label, clip_indices = dataset[0]

            self.assertEqual(clips[0].shape, (3, 8, 16, 16))
            self.assertTrue(clips[0].isfinite().all())
            self.assertEqual(label, 0)
            self.assertEqual(len(clip_indices[0]), 8)

    def test_manifest_glob_groups_dti_directions_as_one_sample(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            first = np.arange(8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
            second = np.flip(first, axis=0).copy()
            nib.save(
                nib.Nifti1Image(first, np.eye(4)),
                temp_path / "subject-DTI-00.nii.gz",
            )
            nib.save(
                nib.Nifti1Image(second, np.eye(4)),
                temp_path / "subject-DTI-01.nii.gz",
            )
            manifest = temp_path / "dti.csv"
            manifest.write_text(f"{temp_path}/*-DTI-*.nii.gz 0\n")

            dataset = MedicalVolumeDataset(
                data_paths=[manifest],
                dataset_fpcs=[4],
                random_clip_sampling=False,
            )

            self.assertEqual(len(dataset), 1)
            self.assertEqual(len(dataset.samples[0]), 2)
            with patch("numpy.random.randint", return_value=0):
                first_clip = dataset[0][0][0]
            with patch("numpy.random.randint", return_value=1):
                second_clip = dataset[0][0][0]
            self.assertFalse(torch.equal(first_clip, second_clip))

    def test_manifest_sample_weights_are_used_by_sampler(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest = temp_path / "weighted.csv"
            for index in range(4):
                volume_path = temp_path / f"sample-{index}.nii.gz"
                nib.save(
                    nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4)),
                    volume_path,
                )
            manifest.write_text(
                f"{temp_path / 'sample-0.nii.gz'} 0 1\n"
                f"{temp_path / 'sample-1.nii.gz'} 0 1\n"
                f"{temp_path / 'sample-2.nii.gz'} 0 1\n"
                f"{temp_path / 'sample-3.nii.gz'} 0 9\n"
            )

            dataset = MedicalVolumeDataset(
                data_paths=[manifest],
                dataset_fpcs=[4],
            )

            self.assertAlmostEqual(sum(dataset.sample_weights[:3]), 0.25)
            self.assertAlmostEqual(sum(dataset.sample_weights[3:]), 0.75)
