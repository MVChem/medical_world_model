from __future__ import annotations

import gc
import json
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from app.vjepa_2_1.models import vision_transformer as video_vit
from torch.utils.data import Dataset

IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)
SUBJECT_PATTERN = re.compile(r"IXI(?P<subject>\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class VolumeRecord:
    path: Path
    subject_id: str
    age: float | None = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _clean_encoder_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    cleaned = {}
    prefixes = ("module.", "backbone.", "_orig_mod.")
    for key, value in state_dict.items():
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key.removeprefix(prefix)
                    changed = True
        cleaned[key] = value
    return cleaned


def load_backbone(
    config_path: str | Path,
    checkpoint_path: str | Path,
    checkpoint_key: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    """Rebuild the exact 2.1 encoder and load one encoder branch."""
    config_path = Path(config_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    data_config = config["data"]
    model_config = config["model"]
    model_name = model_config["model_name"]
    frames_per_clip = int(max(data_config["dataset_fpcs"]))
    crop_size = int(data_config["crop_size"])
    patch_size = int(data_config["patch_size"])
    tubelet_size = int(data_config["tubelet_size"])

    backbone = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=frames_per_clip,
        tubelet_size=tubelet_size,
        uniform_power=model_config.get("uniform_power", False),
        use_sdpa=config.get("meta", {}).get("use_sdpa", False),
        use_silu=model_config.get("use_silu", False),
        wide_silu=model_config.get("wide_silu", True),
        use_activation_checkpointing=False,
        is_causal=model_config.get("is_causal", False),
        use_rope=model_config.get("use_rope", False),
        init_type=model_config.get("init_type", "default"),
        img_temporal_dim_size=model_config.get("img_temporal_dim_size"),
        n_registers=model_config.get("n_registers", 0),
        has_cls_first=model_config.get("has_cls_first", False),
        interpolate_rope=model_config.get("interpolate_rope", False),
        modality_embedding=model_config.get("modality_embedding", False),
    )

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=True
    )
    if checkpoint_key not in checkpoint:
        raise KeyError(
            f"Checkpoint has no {checkpoint_key!r}; available keys: {list(checkpoint)}"
        )
    state_dict = _clean_encoder_state_dict(checkpoint[checkpoint_key])
    backbone.load_state_dict(state_dict, strict=True)
    del checkpoint, state_dict
    gc.collect()

    backbone.requires_grad_(False)
    backbone.eval().to(device)
    metadata = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_key": checkpoint_key,
        "model_name": model_name,
        "embed_dim": int(backbone.embed_dim),
        "frames_per_clip": frames_per_clip,
        "crop_size": crop_size,
        "patch_size": patch_size,
        "tubelet_size": tubelet_size,
    }
    return backbone, metadata


def discover_ixi_volumes(
    volume_root: str | Path, sequence: str = "T1"
) -> list[VolumeRecord]:
    volume_root = Path(volume_root).expanduser().resolve()
    sequence = sequence.upper()
    paths = sorted(volume_root.rglob(f"IXI*-{sequence}.nii.gz"))
    records = []
    seen_subjects = set()
    for path in paths:
        match = SUBJECT_PATTERN.search(path.name)
        if match is None:
            continue
        subject_number = int(match.group("subject"))
        subject_id = f"IXI{subject_number:03d}"
        if subject_id in seen_subjects:
            raise ValueError(
                f"Found more than one {sequence} volume for subject {subject_id}"
            )
        seen_subjects.add(subject_id)
        records.append(VolumeRecord(path=path, subject_id=subject_id))
    if not records:
        raise ValueError(f"No IXI {sequence} volumes found below {volume_root}")
    return records


def _read_ixi_metadata(metadata_path: str | Path) -> pd.DataFrame:
    metadata_path = Path(metadata_path).expanduser().resolve()
    if metadata_path.suffix.lower() == ".csv":
        frame = pd.read_csv(metadata_path)
    else:
        try:
            frame = pd.read_excel(metadata_path)
        except ImportError as error:
            raise RuntimeError(
                "Reading the official IXI.xls needs xlrd. Either install xlrd or "
                "convert the file to CSV with LibreOffice."
            ) from error
    frame.columns = [str(column).strip().upper() for column in frame.columns]
    return frame


def attach_ixi_ages(
    records: Iterable[VolumeRecord], metadata_path: str | Path
) -> list[VolumeRecord]:
    frame = _read_ixi_metadata(metadata_path)
    subject_column = next(
        (
            name
            for name in (
                "IXI_ID",
                "XNAT:MRSESSIONDATA/LABEL",
                "LABEL",
                "SUBJECT_ID",
                "ID",
            )
            if name in frame.columns
        ),
        None,
    )
    age_column = next(
        (
            name
            for name in ("AGE", "AGE_YEARS", "XNAT:MRSESSIONDATA/AGE")
            if name in frame.columns
        ),
        None,
    )
    if subject_column is None or age_column is None:
        raise ValueError(
            "IXI metadata must contain a subject ID/label and AGE "
            f"columns; found {list(frame.columns)}"
        )

    ages = pd.to_numeric(frame[age_column], errors="coerce")
    age_by_subject = {}
    for subject, age in zip(frame[subject_column], ages):
        if pd.isna(subject) or pd.isna(age):
            continue
        match = re.match(r"(?:IXI)?0*(\d+)", str(subject).strip(), re.IGNORECASE)
        if match is not None:
            age_by_subject[f"IXI{int(match.group(1)):03d}"] = float(age)
    with_ages = [
        VolumeRecord(record.path, record.subject_id, age_by_subject[record.subject_id])
        for record in records
        if record.subject_id in age_by_subject
    ]
    if not with_ages:
        raise ValueError("No IXI volumes could be matched to an age in the metadata")
    return with_ages


def split_records(
    records: list[VolumeRecord], seed: int, val_fraction: float, test_fraction: float
) -> dict[str, list[VolumeRecord]]:
    if len(records) < 3:
        raise ValueError("At least three subjects are required for train/val/test")
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError(
            "val_fraction and test_fraction must be positive and sum to < 1"
        )

    ordered = sorted(records, key=lambda record: record.subject_id)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(ordered), generator=generator).tolist()
    shuffled = [ordered[index] for index in permutation]
    num_test = max(1, round(len(shuffled) * test_fraction))
    num_val = max(1, round(len(shuffled) * val_fraction))
    num_train = len(shuffled) - num_val - num_test
    if num_train < 1:
        raise ValueError("The requested split leaves no training subjects")
    return {
        "train": shuffled[:num_train],
        "val": shuffled[num_train : num_train + num_val],
        "test": shuffled[num_train + num_val :],
    }


def split_summary(splits: dict[str, list[VolumeRecord]]) -> dict:
    return {
        name: {
            "count": len(records),
            "subject_ids": [record.subject_id for record in records],
        }
        for name, records in splits.items()
    }


class IXIVolumeDataset(Dataset):
    """Deterministic IXI clips that match the medical pretraining normalization."""

    def __init__(
        self,
        records: list[VolumeRecord],
        frames_per_clip: int,
        crop_size: int,
        num_clips: int = 1,
        return_target: bool = False,
    ) -> None:
        if frames_per_clip < 1 or crop_size < 1 or num_clips < 1:
            raise ValueError(
                "frames_per_clip, crop_size, and num_clips must be positive"
            )
        self.records = records
        self.frames_per_clip = frames_per_clip
        self.crop_size = crop_size
        self.num_clips = num_clips
        self.return_target = return_target

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _load_scaled_volume(path: Path) -> np.ndarray:
        image = nib.as_closest_canonical(nib.load(path))
        volume = np.asarray(image.dataobj, dtype=np.float32).squeeze()
        if volume.ndim != 3:
            raise ValueError(
                f"Expected a 3-D NIfTI volume, got {volume.shape} from {path}"
            )
        finite = volume[np.isfinite(volume)]
        if finite.size == 0:
            raise ValueError(f"Volume contains no finite voxels: {path}")
        lower, upper = np.percentile(finite, (0.5, 99.5))
        if upper <= lower:
            return np.zeros_like(volume, dtype=np.float32)
        volume = np.nan_to_num(volume, nan=lower, posinf=upper, neginf=lower)
        return np.clip((volume - lower) / (upper - lower), 0.0, 1.0).astype(np.float32)

    def _clip_indices(self, num_slices: int) -> list[np.ndarray]:
        if num_slices < self.frames_per_clip:
            indices = (
                np.linspace(0, num_slices - 1, num=self.frames_per_clip)
                .round()
                .astype(np.int64)
            )
            return [indices.copy() for _ in range(self.num_clips)]

        max_start = num_slices - self.frames_per_clip
        if self.num_clips == 1:
            starts = [max_start // 2]
        else:
            starts = np.linspace(0, max_start, self.num_clips + 2)[1:-1]
            starts = np.round(starts).astype(np.int64).tolist()
        return [
            np.arange(start, start + self.frames_per_clip, dtype=np.int64)
            for start in starts
        ]

    def _resize_and_crop(self, clip: torch.Tensor) -> torch.Tensor:
        # Treat slices as a batch so that every frame receives the same geometry.
        _, height, width = clip.shape
        scale = self.crop_size / min(height, width)
        resized_height = max(self.crop_size, round(height * scale))
        resized_width = max(self.crop_size, round(width * scale))
        clip = F.interpolate(
            clip.unsqueeze(1),
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        top = (resized_height - self.crop_size) // 2
        left = (resized_width - self.crop_size) // 2
        return clip[:, top : top + self.crop_size, left : left + self.crop_size]

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        volume = self._load_scaled_volume(record.path)
        inputs = []
        targets = []
        for indices in self._clip_indices(volume.shape[-1]):
            clip = torch.from_numpy(volume[..., indices].copy()).permute(2, 0, 1)
            clip = self._resize_and_crop(clip)
            target = clip.unsqueeze(0)
            rgb_clip = target.repeat(3, 1, 1, 1)
            normalized = (rgb_clip - IMAGENET_MEAN[:, None, None, None]) / IMAGENET_STD[
                :, None, None, None
            ]
            inputs.append(normalized)
            if self.return_target:
                targets.append(target)

        item = {
            "input": torch.stack(inputs),
            "subject_id": record.subject_id,
        }
        if record.age is not None:
            item["age"] = torch.tensor(record.age, dtype=torch.float32)
        if self.return_target:
            item["target"] = torch.stack(targets)
        return item


def autocast_enabled(device: torch.device, disable_amp: bool) -> bool:
    return device.type == "cuda" and not disable_amp
