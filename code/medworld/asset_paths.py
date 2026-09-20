"""Relocate retained assets without changing frozen manifest provenance.

Old paths remain logical identifiers in protocol hashes, not filesystem aliases.
This also lets a sweep started before cleanup load its frozen configuration.
"""
from pathlib import Path


ASSET_PATHS = {
    "code/medworld_stage1/data/overnight_20260910": "code/data/medworld/current",
    "code/medworld_dense_baselines/runs/frozen_slots_20260913/data": "code/data/medworld/dense",
    "code/medworld_stage1/data/slot44_20260911_derived_v2": "code/data/medworld/selection",
    "code/medworld_open_baselines/runs/comparators_20260913/dense_4096/dinov2_vitb14": "code/data/medworld/classification",
    "code/medworld_baselines/runs/raw_models_20260911": "code/data/medworld/baseline",
    "code/medworld_table1/data/linked_20260913_16k": "code/data/medworld/temporal",
    "code/medworld_table1/weights/Qwen3.5-0.8B": "code/data/medworld/weights/Qwen3.5-0.8B",
}


def relocate_asset(path, root):
    """Translate paths from old configurations and unchanged source manifests."""
    root, path = Path(root), Path(path)
    absolute = path if path.is_absolute() else root / path
    aliases = {**ASSET_PATHS,
               "code/medworld_dense_baselines/runs/dense_20260912/data": "code/data/medworld/dense"}
    for old, new in aliases.items():
        try:
            suffix = absolute.relative_to(root / old)
        except ValueError:
            continue
        return root / new / suffix
    return absolute


def provenance_path(path, root):
    """Keep historical protocol keys stable when assets move without edits."""
    path, root = Path(path), Path(root)
    for old, new in ASSET_PATHS.items():
        try:
            suffix = path.relative_to(root / new)
        except ValueError:
            continue
        return root / old / suffix
    return path
