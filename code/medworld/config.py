"""Explicit, local prototype configuration; budgets count optimizer updates."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
from .downstream_tasks.registry import TASKS
from .architecture import RAW_INPUT, architecture, normalize_baseline
from .evaluation.protocol import FUTURE_TASKS

PROJECT = Path(os.environ.get("MEDWORLD_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
DEFAULTS = {
    "architecture": RAW_INPUT,
    "prepared_data": "code/data/medworld_0922",
    "vqa_data": "code/data/medworld_0922/vqa",
    "image_root": "code/data/medworld_0922/images",
    "slot_conditioning": True,
    "testing": {"enabled": True, "tasks": list(TASKS), "future_tasks": list(FUTURE_TASKS), "human_segmentation": True, "reuse_completed": True, "vqa_per_type": 0, "vqa_seed": 42},
    "baselines": {"no_slots": True, "qwen": True},
    "decoder_width": 256, "decoder_depth": 2, "task_patch_size": 16,
    "segmentation_channels": 6,
    "segmentation_sampling": "balanced_dataset",
    "observation_prompt": "Medical image observation.",

    "qwen": "code/data/medworld/weights/Qwen3.5-0.8B",
    "jepa": "code/vjepa2/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt",
    "temporal_data": "code/data/medworld_0923",
    "seed": 42, "lora_rank": 8, "lora_alpha": 16,
    "vision_pixels": 256, "answer_tokens": 384, "context_tokens": 384,
    "generation_tokens": 384, "predictor_width": 512, "predictor_depth": 4,
    "ema_momentum": 0.99, "bidirectional": True,
    "learning_rate": 1e-4, "lora_learning_rate": 5e-5, "max_grad_norm": 1.0,
    "batch_size": 1, "accumulation": 8,
    "steps": 4800,
    "latent_weight": 1.0,
    "save_every": 100, "validate_every": 200, "validation_samples": 8,
    "amp": False, "task_batch_sizes": {}, "prefetch_batches": 2, "image_workers": 1,
    "visual_consistency_weight": 0.0, "visual_consistency_views": 2,
    "total_hours": 0.0,
    "require_fast_kernels": False,
    "cpu_threads": 8,
    "cpu_cores_per_rank": 8,
    "prefetch_preprocessing": False,
    "batched_vision_attention": False,
    "decoded_image_cache": 0,
    "future_enabled": False,
    "future_data": "code/data/medworld_0923/table12_v1",
    "future_source_bytes": 383,
    "future_report_tokens": 2048,
    "future_weight": 1.0,
    "radgraph_assets": "code/data/medworld_evaluation/radgraph-xl",
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
    architecture(cfg)
    if not isinstance(cfg["prepared_data"], str) or not cfg["prepared_data"].strip():
        raise ValueError("prepared_data must be a nonempty reviewed-manifest directory path")
    if cfg["segmentation_channels"] != 6 or type(cfg["segmentation_channels"]) is not int:
        raise ValueError("segmentation_channels must be 6 (reviewed CXR/MRI)")
    if cfg["segmentation_sampling"] not in ("uniform", "balanced_dataset"):
        raise ValueError("segmentation_sampling must be uniform or balanced_dataset")
    if not isinstance(cfg["observation_prompt"], str) or not cfg["observation_prompt"].strip():
        raise ValueError("observation_prompt must be nonempty text")
    baselines = supplied.get("baselines", {})
    if (not isinstance(baselines, dict) or set(baselines) - set(DEFAULTS["baselines"])
            or any(type(value) is not bool for value in baselines.values())):
        raise ValueError("baselines accepts boolean no_slots and qwen switches")
    cfg["baselines"] = {**DEFAULTS["baselines"], **baselines}
    testing = supplied.get("testing", {})
    if not isinstance(testing, dict) or set(testing) - set(DEFAULTS["testing"]):
        raise ValueError("Unknown testing configuration")
    cfg["testing"] = {**deepcopy(DEFAULTS["testing"]), **testing}
    testing = cfg["testing"]
    for key in ("enabled", "human_segmentation", "reuse_completed"):
        if type(testing[key]) is not bool:
            raise ValueError(f"testing.{key} must be boolean")
    for key in ("vqa_per_type", "vqa_seed"):
        if type(testing[key]) is not int or testing[key] < 0:
            raise ValueError(f"testing.{key} must be a nonnegative integer")
    tasks = testing["tasks"]
    if (not isinstance(tasks, list) or any(not isinstance(t, str) or t not in TASKS for t in tasks)
            or len(set(tasks)) != len(tasks) or (testing["enabled"] and not tasks)):
        raise ValueError("testing.tasks must list distinct supported tasks (nonempty when enabled)")
    future_tasks = testing["future_tasks"]
    if (not isinstance(future_tasks, list) or len(set(future_tasks)) != len(future_tasks)
            or any(task not in FUTURE_TASKS for task in future_tasks)):
        raise ValueError("testing.future_tasks must list distinct Table 1 tasks")
    if type(cfg["future_enabled"]) is not bool:
        raise ValueError("future_enabled must be boolean")
    if cfg["future_enabled"] and testing["enabled"] and set(future_tasks) != set(FUTURE_TASKS):
        raise ValueError("A completed two-table experiment must evaluate all five future tasks")
    integers = ("decoder_width", "decoder_depth", "task_patch_size", "lora_rank", "lora_alpha", "vision_pixels", "answer_tokens", "context_tokens",
                "generation_tokens", "predictor_width", "predictor_depth", "batch_size",
                "accumulation", "steps",
                "save_every", "validate_every", "validation_samples", "prefetch_batches", "image_workers", "visual_consistency_views", "cpu_threads", "cpu_cores_per_rank", "future_source_bytes", "future_report_tokens")
    for key in integers:
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["decoder_width"] % 8:
        raise ValueError("decoder_width must be divisible by 8")
    if cfg["vision_pixels"] % cfg["task_patch_size"]:
        raise ValueError("vision_pixels must be divisible by task_patch_size")
    if cfg["answer_tokens"] < 2 or cfg["predictor_width"] % 8:
        raise ValueError("answer_tokens >= 2; predictor_width must be divisible by 8")
    if cfg["future_source_bytes"] >= cfg["context_tokens"] and cfg["future_enabled"]:
        raise ValueError("future_source_bytes must leave room for EOS within the shared raw report context")
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    if type(cfg["bidirectional"]) is not bool:
        raise ValueError("bidirectional must be boolean")
    if type(cfg["slot_conditioning"]) is not bool:
        raise ValueError("slot_conditioning must be boolean")
    if type(cfg["amp"]) is not bool:
        raise ValueError("amp must be boolean")
    if type(cfg["require_fast_kernels"]) is not bool:
        raise ValueError("require_fast_kernels must be boolean")
    if type(cfg["prefetch_preprocessing"]) is not bool:
        raise ValueError("prefetch_preprocessing must be boolean")
    if type(cfg["batched_vision_attention"]) is not bool:
        raise ValueError("batched_vision_attention must be boolean")
    if type(cfg["decoded_image_cache"]) is not int or cfg["decoded_image_cache"] < 0:
        raise ValueError("decoded_image_cache must be a nonnegative integer")
    sizes = cfg["task_batch_sizes"]
    if (not isinstance(sizes, dict)
            or set(sizes) - (set(TASKS) | set(FUTURE_TASKS) | {"temporal"})
            or any(type(value) is not int or value <= 0 for value in sizes.values())):
        raise ValueError("task_batch_sizes must map known tasks to positive per-rank batch sizes")
    for key in ("ema_momentum", "learning_rate", "lora_learning_rate", "max_grad_norm",
                "latent_weight", "total_hours", "visual_consistency_weight", "future_weight"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        if value < 0 or (key.endswith("learning_rate") or key == "max_grad_norm") and value == 0:
            raise ValueError(f"Invalid {key}")
    if not 0 <= cfg["ema_momentum"] < 1:
        raise ValueError("ema_momentum must be in [0, 1)")
    for key in ('qwen', 'jepa', 'prepared_data', 'temporal_data', 'vqa_data', 'image_root', 'future_data', 'radgraph_assets'):
        p = Path(cfg[key]).expanduser()
        p = p if p.is_absolute() else root / p
        # Manifest identities use the configured project links, not symlink targets.
        cfg[key] = str(p.resolve() if key in ("qwen", "jepa", "temporal_data") else p.absolute())
    return normalize_baseline(cfg)
