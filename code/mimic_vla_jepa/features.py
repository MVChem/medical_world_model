from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors.torch import load_file
from torch.utils.data import Dataset

from .data import iter_jsonl


@dataclass(frozen=True)
class FeatureRecord:
    transition_id: str
    split: str
    feature_file: Path


def load_feature_manifest(
    path: str | Path, split: str | None = None
) -> list[FeatureRecord]:
    path = Path(path)
    records: list[FeatureRecord] = []
    seen: set[str] = set()
    for row in iter_jsonl(path):
        if split is not None and row.get("split") != split:
            continue
        transition_id = row.get("transition_id")
        if not isinstance(transition_id, str) or transition_id in seen:
            raise ValueError(f"missing or duplicate transition_id in {path}")
        feature_file = row.get("feature_file")
        if not isinstance(feature_file, str):
            raise TypeError("feature_file must be a string")
        feature_path = Path(feature_file)
        if not feature_path.is_absolute():
            feature_path = path.parent / feature_path
        if not feature_path.is_file():
            raise FileNotFoundError(feature_path)
        records.append(
            FeatureRecord(
                transition_id=transition_id,
                split=str(row.get("split")),
                feature_file=feature_path,
            )
        )
        seen.add(transition_id)
    if not records:
        raise ValueError(f"no feature rows in {path}")
    return records


class CachedFeatureDataset(Dataset[dict[str, Any]]):
    def __init__(self, manifest: str | Path, split: str | None = None):
        self.records = load_feature_manifest(manifest, split=split)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        tensors = load_file(record.feature_file, device="cpu")
        expected = {"source_state", "target_state", "query_state"}
        if set(tensors) != expected:
            raise ValueError(
                f"{record.feature_file} has keys {sorted(tensors)}, expected {sorted(expected)}"
            )
        return {
            "transition_id": record.transition_id,
            **tensors,
        }


def read_metadata(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError("feature metadata must be an object")
    return value
