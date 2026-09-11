from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset

from .data import ALLOWED_SPLITS, iter_jsonl, record_from_row

STATE_KEYS = frozenset({"source_state", "target_state"})


@dataclass(frozen=True)
class JoinedSeriousRecord:
    """Current-only conditioning fields joined to a stopped-gradient state pair."""

    transition_id: str
    patient_id: str
    split: str
    source_image: Path
    prompt: str
    state_file: Path


def _state_file_from_row(row: dict[str, Any], manifest: Path) -> Path:
    value = row.get("state_file")
    legacy_value = row.get("feature_file")
    if value is not None and legacy_value is not None:
        if not isinstance(value, str) or not isinstance(legacy_value, str):
            raise TypeError("state_file and legacy feature_file must be strings")

        def resolved(candidate: str) -> Path:
            path = Path(candidate)
            if not path.is_absolute():
                path = manifest.parent / path
            return path.resolve(strict=False)

        if resolved(value) != resolved(legacy_value):
            raise ValueError("state_file and legacy feature_file disagree")
    if value is None:
        value = legacy_value
    if not isinstance(value, str) or not value:
        raise TypeError("state_file must be a nonempty string")
    path = Path(value)
    if not path.is_absolute():
        path = manifest.parent / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_joined_serious_records(
    forecast_manifest: str | Path,
    state_manifest: str | Path,
    *,
    split: str,
    require_image_files: bool = True,
    strict: bool = True,
) -> list[JoinedSeriousRecord]:
    """Join raw transitions and state-only features by ``transition_id``.

    The online prompt is deliberately rebuilt by :func:`record_from_row`; the
    possibly stale ``prompt`` field in the raw manifest is never trusted.  The
    resulting records expose neither a target image nor any future report or
    disease-state field to the online Qwen collator.

    With ``strict=True`` (the training default), both selected split manifests
    must contain exactly the same transition IDs.  This prevents silently
    training on a partial or mis-versioned cache.
    """

    if split not in ALLOWED_SPLITS:
        raise ValueError(f"split must be one of {sorted(ALLOWED_SPLITS)}")
    forecast_manifest = Path(forecast_manifest)
    state_manifest = Path(state_manifest)

    transitions: dict[str, Any] = {}
    patient_ids: dict[str, str] = {}
    transition_order: list[str] = []
    for row in iter_jsonl(forecast_manifest):
        if row.get("split") != split:
            continue
        record = record_from_row(row, require_files=require_image_files)
        if record.transition_id in transitions:
            raise ValueError(
                f"duplicate transition_id in forecast manifest: {record.transition_id}"
            )
        transitions[record.transition_id] = record
        patient_id = row.get("patient_id")
        if not isinstance(patient_id, str) or not patient_id:
            raise ValueError("patient_id must be a nonempty string")
        patient_ids[record.transition_id] = patient_id
        transition_order.append(record.transition_id)

    state_files: dict[str, Path] = {}
    for row in iter_jsonl(state_manifest):
        if row.get("split") != split:
            continue
        transition_id = row.get("transition_id")
        if not isinstance(transition_id, str) or not transition_id:
            raise ValueError("state manifest transition_id must be a nonempty string")
        if transition_id in state_files:
            raise ValueError(
                f"duplicate transition_id in state manifest: {transition_id}"
            )
        state_files[transition_id] = _state_file_from_row(row, state_manifest)

    if not transitions:
        raise ValueError(f"no {split!r} transitions in {forecast_manifest}")
    if not state_files:
        raise ValueError(f"no {split!r} state features in {state_manifest}")

    transition_ids = set(transitions)
    state_ids = set(state_files)
    missing = sorted(transition_ids - state_ids)
    extra = sorted(state_ids - transition_ids)
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"state cache is missing {len(missing)} {split!r} transitions; first: {preview}"
        )
    if strict and extra:
        preview = ", ".join(extra[:5])
        raise ValueError(
            f"state cache has {len(extra)} unmatched {split!r} transitions; first: {preview}"
        )

    return [
        JoinedSeriousRecord(
            transition_id=transition_id,
            patient_id=patient_ids[transition_id],
            split=split,
            source_image=transitions[transition_id].source_image,
            prompt=transitions[transition_id].prompt,
            state_file=state_files[transition_id],
        )
        for transition_id in transition_order
    ]


class SeriousTransitionDataset(Dataset[dict[str, Any]]):
    """Loads cached V-JEPA states while leaving Qwen conditioning online."""

    def __init__(
        self,
        forecast_manifest: str | Path,
        state_manifest: str | Path,
        *,
        split: str,
        require_image_files: bool = True,
        strict: bool = True,
    ):
        self.records = load_joined_serious_records(
            forecast_manifest,
            state_manifest,
            split=split,
            require_image_files=require_image_files,
            strict=strict,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        tensors = load_file(record.state_file, device="cpu")
        if set(tensors) != STATE_KEYS:
            raise ValueError(
                f"{record.state_file} has keys {sorted(tensors)}, "
                f"expected state-only keys {sorted(STATE_KEYS)}"
            )
        source_state = tensors["source_state"]
        target_state = tensors["target_state"]
        if source_state.shape != target_state.shape:
            raise ValueError(
                f"source/target state shapes differ in {record.state_file}: "
                f"{tuple(source_state.shape)} versus {tuple(target_state.shape)}"
            )
        if not source_state.is_floating_point() or not target_state.is_floating_point():
            raise TypeError(
                f"state tensors must be floating point: {record.state_file}"
            )
        if (
            not torch.isfinite(source_state).all()
            or not torch.isfinite(target_state).all()
        ):
            raise ValueError(f"state tensors must be finite: {record.state_file}")
        return {
            "dataset_index": index,
            "transition_id": record.transition_id,
            "patient_id": record.patient_id,
            "source_image": record.source_image,
            "prompt": record.prompt,
            "source_state": source_state,
            "target_state": target_state,
        }


def batched_query_positions(
    input_ids: torch.Tensor, token_id: int, expected_per_example: int
) -> torch.Tensor:
    """Return ordered query positions with shape ``[batch, queries]``."""

    if input_ids.ndim != 2:
        raise ValueError(
            f"input_ids must have shape [B, L], got {tuple(input_ids.shape)}"
        )
    if expected_per_example <= 0:
        raise ValueError("expected_per_example must be positive")
    rows: list[torch.Tensor] = []
    for batch_index in range(input_ids.shape[0]):
        positions = torch.nonzero(
            input_ids[batch_index] == token_id, as_tuple=False
        ).flatten()
        if positions.numel() != expected_per_example:
            raise ValueError(
                f"example {batch_index} has {positions.numel()} query markers; "
                f"expected {expected_per_example}"
            )
        rows.append(positions)
    return torch.stack(rows)
