from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mimic_atlas.forecast_contract import (
    build_coarse_horizon_prompt,
    horizon_bin_from_elapsed_hours,
    validate_horizon_bin,
)

SAFE_TRANSITION_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
ALLOWED_SPLITS = {"train", "validate", "test"}
TARGET_ONLY_FIELD_NAMES = {
    "future_report_for_eval_only",
    "state_delta_for_eval_only",
    "target_image",
    "target_image_relative",
    "target_state_for_eval_only",
    "target_study_id",
}


@dataclass(frozen=True)
class TransitionRecord:
    transition_id: str
    split: str
    source_image: Path
    target_image: Path
    prompt: str
    elapsed_hours: float
    horizon_bin: str


def build_forecasting_prompt(row: dict[str, Any]) -> str:
    """Construct the online text only from explicitly whitelisted current fields."""

    elapsed_hours = row.get("elapsed_hours")
    computed_horizon_bin = horizon_bin_from_elapsed_hours(elapsed_hours)
    provided_horizon_bin = row.get("horizon_bin")
    if provided_horizon_bin is None:
        # Backward compatibility for the four-case smoke manifest. New
        # manifests always persist the bin explicitly.
        horizon_bin = computed_horizon_bin
    else:
        horizon_bin = validate_horizon_bin(provided_horizon_bin)
        if horizon_bin != computed_horizon_bin:
            raise ValueError(
                "horizon_bin is inconsistent with elapsed_hours: "
                f"{horizon_bin!r} versus {computed_horizon_bin!r}"
            )

    source_report = row.get("source_report")
    return build_coarse_horizon_prompt(source_report, horizon_bin)


def _validate_path(value: Any, field: str, require_files: bool) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty path string")
    path = Path(value)
    if require_files and not path.is_file():
        raise FileNotFoundError(f"{field} does not exist: {path}")
    return path


def record_from_row(
    row: dict[str, Any], require_files: bool = True
) -> TransitionRecord:
    if row.get("task_type") != "forecasting":
        raise ValueError("only task_type='forecasting' is accepted")

    transition_id = row.get("transition_id")
    if not isinstance(transition_id, str) or not SAFE_TRANSITION_ID.fullmatch(
        transition_id
    ):
        raise ValueError(
            "transition_id is missing or unsafe for use as a feature filename"
        )

    split = row.get("split")
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"split must be one of {sorted(ALLOWED_SPLITS)}")

    source_image = _validate_path(
        row.get("source_image"), "source_image", require_files
    )
    target_image = _validate_path(
        row.get("target_image"), "target_image", require_files
    )
    same_image = (
        source_image.samefile(target_image)
        if require_files
        else source_image.resolve(strict=False) == target_image.resolve(strict=False)
    )
    if same_image:
        raise ValueError("source_image and target_image must differ")

    elapsed_hours = float(row["elapsed_hours"])
    horizon_bin = horizon_bin_from_elapsed_hours(elapsed_hours)
    if row.get("horizon_bin") is not None:
        horizon_bin = validate_horizon_bin(row["horizon_bin"])
    prompt = build_forecasting_prompt(row)
    for field_name in TARGET_ONLY_FIELD_NAMES:
        if field_name in prompt:
            raise AssertionError(
                f"target-only field name leaked into prompt: {field_name}"
            )

    return TransitionRecord(
        transition_id=transition_id,
        split=split,
        source_image=source_image,
        target_image=target_image,
        prompt=prompt,
        elapsed_hours=elapsed_hours,
        horizon_bin=horizon_bin,
    )


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"row at {path}:{line_number} must be an object")
            yield row


def load_transition_records(
    manifest: str | Path,
    *,
    split: str | None = None,
    max_records: int | None = None,
    require_files: bool = True,
) -> list[TransitionRecord]:
    manifest = Path(manifest)
    if split is not None and split not in ALLOWED_SPLITS:
        raise ValueError(f"split must be one of {sorted(ALLOWED_SPLITS)}")
    if max_records is not None and max_records <= 0:
        raise ValueError("max_records must be positive")

    records: list[TransitionRecord] = []
    seen: set[str] = set()
    for row in iter_jsonl(manifest):
        if split is not None and row.get("split") != split:
            continue
        record = record_from_row(row, require_files=require_files)
        if record.transition_id in seen:
            raise ValueError(f"duplicate transition_id: {record.transition_id}")
        seen.add(record.transition_id)
        records.append(record)
        if max_records is not None and len(records) >= max_records:
            break

    if not records:
        raise ValueError(f"no matching transition records in {manifest}")
    return records
