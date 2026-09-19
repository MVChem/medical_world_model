"""Explicit, local prototype configuration; budgets count optimizer updates."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path

PROJECT = Path(os.environ.get("MEDWORLD_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
DEFAULTS = {
    "current_data": "code/medworld_stage1/data/overnight_20260910",
    "dense_data": "code/medworld_dense_baselines/runs/frozen_slots_20260913/data",
    "baseline_data": "code/medworld_baselines/runs/raw_models_20260911",
    "selection_file": "code/medworld_stage1/data/slot44_20260911_derived_v2/selection.json",
    "classification_data": "code/medworld_open_baselines/runs/comparators_20260913/dense_4096/dinov2_vitb14",
    "clinical_weights": "code/medworld_table1/weights",

    "qwen": "code/medworld_table1/weights/Qwen3.5-0.8B",
    "jepa": "code/medworld_table1/weights/vjepa2_1_vitb.pt",
    "temporal_data": "code/medworld_table1/data/linked_20260913_16k",
    "seed": 42, "lora_rank": 8, "lora_alpha": 16,
    "vision_pixels": 256, "report_tokens": 384, "context_tokens": 384,
    "generation_tokens": 64, "predictor_width": 512, "predictor_depth": 4,
    "ema_momentum": 0.99, "bidirectional": True,
    "learning_rate": 1e-4, "lora_learning_rate": 5e-5, "max_grad_norm": 1.0,
    "batch_size": 1, "accumulation": 8,
    "steps": 4800,
    "latent_weight": 1.0, "report_weight": 1.0, "finding_weight": 0.5,
    "save_every": 100, "validate_every": 200, "validation_samples": 8,
    "ce_chunk_tokens": 32, "amp": False, "task_batch_sizes": {}, "prefetch_batches": 2, "image_workers": 1,
    "visual_consistency_weight": 0.0, "visual_consistency_views": 2,
    "total_hours": 0.0,
}


def load_config(path=None, overrides=None, root=None):
    root = Path(root or PROJECT).resolve()
    supplied = json.loads(Path(path).read_text()) if path else {}
    supplied.update(overrides or {})
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    cfg = deepcopy(DEFAULTS)
    cfg.update(supplied)
    integers = ("lora_rank", "lora_alpha", "vision_pixels", "report_tokens", "context_tokens",
                "generation_tokens", "predictor_width", "predictor_depth", "batch_size",
                "accumulation", "steps",
                "save_every", "validate_every", "validation_samples", "ce_chunk_tokens", "prefetch_batches", "image_workers", "visual_consistency_views")
    for key in integers:
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["report_tokens"] < 2 or cfg["predictor_width"] % 8:
        raise ValueError("report_tokens >= 2; predictor_width must be divisible by 8")
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    if type(cfg["bidirectional"]) is not bool:
        raise ValueError("bidirectional must be boolean")
    if type(cfg["amp"]) is not bool:
        raise ValueError("amp must be boolean")
    sizes = cfg["task_batch_sizes"]
    if (not isinstance(sizes, dict)
            or set(sizes) - {"classification", "report", "segmentation", "sr", "temporal"}
            or any(type(value) is not int or value <= 0 for value in sizes.values())):
        raise ValueError("task_batch_sizes must map known tasks to positive per-rank batch sizes")
    for key in ("ema_momentum", "learning_rate", "lora_learning_rate", "max_grad_norm",
                "latent_weight", "report_weight", "finding_weight", "total_hours", "visual_consistency_weight"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        if value < 0 or (key.endswith("learning_rate") or key == "max_grad_norm") and value == 0:
            raise ValueError(f"Invalid {key}")
    if not 0 <= cfg["ema_momentum"] < 1:
        raise ValueError("ema_momentum must be in [0, 1)")
    if cfg["report_weight"] <= 0:
        raise ValueError("Predicted-state report supervision must have positive weight")
    for key in ('qwen', 'jepa', 'temporal_data', 'current_data', 'dense_data', 'baseline_data', 'selection_file', 'classification_data', 'clinical_weights'):
        p = Path(cfg[key]).expanduser()
        p = root / p if not p.is_absolute() else p
        # Preserve manifest aliases: their logical paths are part of the existing protocol hash.
        cfg[key] = str(p.resolve() if key in ("qwen", "jepa", "temporal_data") else p.absolute())
    return cfg
