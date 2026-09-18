# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from src.datasets.utils.dataloader import ConcatIndices
from src.datasets.utils.weighted_sampler import DistributedWeightedSampler


def make_medical_volume_dataset(
    data_paths,
    batch_size,
    dataset_fpcs=None,
    transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=0,
    pin_mem=False,
    persistent_workers=False,
    random_clip_sampling=True,
):
    dataset = MedicalVolumeDataset(
        data_paths=data_paths,
        dataset_fpcs=dataset_fpcs,
        transform=transform,
        datasets_weights=datasets_weights,
        random_clip_sampling=random_clip_sampling,
    )
    if dataset.sample_weights is not None:
        dist_sampler = DistributedWeightedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )
    return dataset, data_loader, dist_sampler


class MedicalVolumeDataset(torch.utils.data.Dataset):
    """Expose 3-D NIfTI volumes through the unchanged V-JEPA 2.1 video API.

    A volume is sampled along its final canonical axis and converted to a
    three-channel clip with shape ``[C, T, H, W]``. Repeating the grayscale
    slices keeps the original RGB patch embedding and checkpoint shapes
    untouched.
    """

    _NIFTI_SUFFIXES = (".nii", ".nii.gz")

    def __init__(
        self,
        data_paths,
        dataset_fpcs=None,
        transform=None,
        datasets_weights=None,
        random_clip_sampling=True,
    ):
        if isinstance(data_paths, (str, os.PathLike)):
            data_paths = [data_paths]
        if not data_paths:
            raise ValueError("MedicalVolumeDataset requires at least one data path")

        expanded_paths = [
            Path(os.path.expandvars(os.path.expanduser(str(p)))) for p in data_paths
        ]
        if dataset_fpcs is None:
            raise ValueError(
                "dataset_fpcs must specify one clip length per medical dataset"
            )
        if len(dataset_fpcs) != len(expanded_paths):
            raise ValueError("dataset_fpcs must have the same length as data_paths")
        if datasets_weights is not None and len(datasets_weights) != len(
            expanded_paths
        ):
            raise ValueError("datasets_weights must have the same length as data_paths")

        self.transform = transform
        self.random_clip_sampling = random_clip_sampling
        self.dataset_fpcs = [int(v) for v in dataset_fpcs]
        self.datasets_weights = datasets_weights

        self.samples = []
        self.labels = []
        self.num_samples_per_dataset = []
        per_dataset_sample_weights = []
        for data_path in expanded_paths:
            samples, labels, sample_weights = self._read_data_path(data_path)
            if not samples:
                raise ValueError(f"No NIfTI volumes found in {data_path}")
            self.samples.extend(samples)
            self.labels.extend(labels)
            self.num_samples_per_dataset.append(len(samples))
            per_dataset_sample_weights.append(sample_weights)

        self.per_dataset_indices = ConcatIndices(self.num_samples_per_dataset)
        self.sample_weights = None
        if self.datasets_weights is not None or any(
            weights is not None for weights in per_dataset_sample_weights
        ):
            dataset_weights = self.datasets_weights or [1.0] * len(expanded_paths)
            self.sample_weights = []
            for dataset_weight, num_samples, sample_weights in zip(
                dataset_weights,
                self.num_samples_per_dataset,
                per_dataset_sample_weights,
            ):
                if sample_weights is None:
                    sample_weights = np.ones(num_samples, dtype=np.float64)
                else:
                    sample_weights = np.asarray(sample_weights, dtype=np.float64)
                if (
                    not np.isfinite(sample_weights).all()
                    or np.any(sample_weights < 0)
                    or sample_weights.sum() <= 0
                ):
                    raise ValueError(
                        "Manifest sample weights must be finite and non-negative"
                    )
                sample_weights /= sample_weights.sum()
                self.sample_weights.extend((dataset_weight * sample_weights).tolist())

    @classmethod
    def _is_nifti(cls, path):
        return str(path).lower().endswith(cls._NIFTI_SUFFIXES)

    @classmethod
    def _read_data_path(cls, data_path):
        if data_path.is_dir():
            samples = sorted(
                str(path)
                for path in data_path.rglob("*")
                if path.is_file() and cls._is_nifti(path)
            )
            return samples, [0] * len(samples), None
        if cls._is_nifti(data_path):
            return [str(data_path)], [0], None
        if data_path.suffix.lower() in (".csv", ".txt"):
            frame = pd.read_csv(
                data_path, header=None, sep=r"\s+", engine="python", comment="#"
            )
            if frame.empty:
                return [], []
            samples = []
            for value in frame.iloc[:, 0]:
                sample = os.path.expandvars(os.path.expanduser(str(value)))
                if glob.has_magic(sample):
                    matches = tuple(
                        path
                        for path in sorted(glob.glob(sample))
                        if cls._is_nifti(path)
                    )
                    if not matches:
                        raise ValueError(f"NIfTI glob matched no files: {sample}")
                    sample = matches
                samples.append(sample)
            labels = (
                frame.iloc[:, 1].tolist() if frame.shape[1] > 1 else [0] * len(samples)
            )
            sample_weights = frame.iloc[:, 2].tolist() if frame.shape[1] > 2 else None
            return samples, labels, sample_weights
        raise ValueError(f"Unsupported medical data path: {data_path}")

    @staticmethod
    def _sample_indices(num_slices, frames_per_clip, random_clip_sampling):
        if num_slices >= frames_per_clip:
            if random_clip_sampling:
                start = np.random.randint(0, num_slices - frames_per_clip + 1)
            else:
                start = (num_slices - frames_per_clip) // 2
            return np.arange(start, start + frames_per_clip, dtype=np.int64)
        return (
            np.linspace(0, num_slices - 1, num=frames_per_clip).round().astype(np.int64)
        )

    @staticmethod
    def _scale_to_uint8_range(volume):
        finite = volume[np.isfinite(volume)]
        if finite.size == 0:
            raise ValueError("Volume contains no finite voxels")
        lower, upper = np.percentile(finite, (0.5, 99.5))
        if upper <= lower:
            return np.zeros_like(volume, dtype=np.float32)
        volume = np.nan_to_num(volume, nan=lower, posinf=upper, neginf=lower)
        return (
            np.clip((volume - lower) / (upper - lower), 0.0, 1.0).astype(np.float32)
            * 255.0
        )

    def __getitem__(self, index):
        dataset_index, _ = self.per_dataset_indices[index]
        frames_per_clip = self.dataset_fpcs[dataset_index]

        sample = self.samples[index]
        if isinstance(sample, tuple):
            sample = sample[np.random.randint(len(sample))]

        image = nib.as_closest_canonical(nib.load(sample))
        volume = np.asarray(image.dataobj, dtype=np.float32).squeeze()
        if volume.ndim != 3:
            raise ValueError(
                f"Expected a 3-D NIfTI volume, got shape {volume.shape} from {sample}"
            )
        volume = self._scale_to_uint8_range(volume)

        indices = self._sample_indices(
            volume.shape[-1], frames_per_clip, self.random_clip_sampling
        )
        clip = np.moveaxis(volume[..., indices], -1, 0)  # T, H, W
        clip = np.repeat(clip[..., None], 3, axis=-1)  # T, H, W, RGB

        if self.transform is not None:
            clip = self.transform(clip)
        else:
            clip = torch.from_numpy(clip).permute(3, 0, 1, 2).contiguous()

        return [clip], self.labels[index], [indices.astype(np.int32)]

    def __len__(self):
        return len(self.samples)
