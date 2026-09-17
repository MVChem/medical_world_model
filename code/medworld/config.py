"""Explicit, local prototype configuration; budgets count optimizer updates."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path

PROJECT = Path(os.environ.get("MEDWORLD_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
DEFAULTS = {
    "qwen": "code/medworld_table1/weights/Qwen3.5-0.8B",
    "jepa": "code/medworld_table1/weights/vjepa2_1_vitb.pt",
    "temporal_data": "code/medworld_table1/data/linked_20260913_16k",
    "seed": 42, "lora_rank": 8, "lora_alpha": 16,
    "vision_pixels": 256, "report_tokens": 384, "context_tokens": 384,
    "generation_tokens": 64, "predictor_width": 512, "predictor_depth": 4,
    "ema_momentum": 0.99, "bidirectional": True,
    "learning_rate": 1e-4, "lora_learning_rate": 5e-5, "max_grad_norm": 1.0,
    "batch_size": 1, "stage1_accumulation": 8, "stage2_accumulation": 32,
    "stage1_steps": 2400, "stage2_steps": 2400,
    "latent_weight": 1.0, "report_weight": 1.0, "finding_weight": 0.5,
    "replay_every": 4, "replay_weight": 1.0,
    "save_every": 100, "validate_every": 200, "validation_samples": 8,
    "ce_chunk_tokens": 32, "amp": False, "task_batch_sizes": {}, "prefetch_batches": 2, "image_workers": 1,
    "total_hours": 0.0, "stage1_hours": 0.0,
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
                "stage1_accumulation", "stage2_accumulation", "stage1_steps", "stage2_steps",
                "save_every", "validate_every", "validation_samples", "ce_chunk_tokens", "prefetch_batches", "image_workers")
    for key in integers:
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["report_tokens"] < 2 or cfg["predictor_width"] % 8:
        raise ValueError("report_tokens >= 2; predictor_width must be divisible by 8")
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    if type(cfg["replay_every"]) is not int or cfg["replay_every"] < 0:
        raise ValueError("replay_every must be a nonnegative integer")
    if type(cfg["bidirectional"]) is not bool:
        raise ValueError("bidirectional must be boolean")
    if type(cfg["amp"]) is not bool:
        raise ValueError("amp must be boolean")
    sizes = cfg["task_batch_sizes"]
    if (not isinstance(sizes, dict) or set(sizes) - {"classification", "report", "segmentation", "sr", "temporal"}
            or any(type(value) is not int or value <= 0 for value in sizes.values())):
        raise ValueError("task_batch_sizes must map known tasks to positive per-rank batch sizes")
    for key in ("ema_momentum", "learning_rate", "lora_learning_rate", "max_grad_norm",
                "latent_weight", "report_weight", "finding_weight", "replay_weight", "total_hours", "stage1_hours"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        if value < 0 or (key.endswith("learning_rate") or key == "max_grad_norm") and value == 0:
            raise ValueError(f"Invalid {key}")
    if not 0 <= cfg["ema_momentum"] < 1:
        raise ValueError("ema_momentum must be in [0, 1)")
    if cfg["report_weight"] <= 0:
        raise ValueError("Predicted-state report supervision must have positive weight")
    if cfg["total_hours"] > 0 and not 0 < cfg["stage1_hours"] < cfg["total_hours"]:
        raise ValueError("Timed runs require 0 < stage1_hours < total_hours")
    if cfg["total_hours"] == 0 and cfg["stage1_hours"] != 0:
        raise ValueError("stage1_hours requires a timed run")
    for key in ("qwen", "jepa", "temporal_data"):
        p = Path(cfg[key]).expanduser()
        cfg[key] = str((root / p).resolve() if not p.is_absolute() else p.resolve())
    return cfg
