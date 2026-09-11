from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from PIL import Image
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler, Subset
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from .backbones import load_letterboxed_rgb
from .io_utils import atomic_write_json, atomic_write_text, jsonl_text, secure_directory
from .predictor import PredictorConfig, VLAJEPAPredictor
from .serious_data import SeriousTransitionDataset, batched_query_positions


@dataclass(frozen=True)
class QwenSettings:
    query_token: str = "<|latent_0|>"
    query_tokens: int = 24
    image_size: int = 224
    max_length: int = 2048
    gradient_checkpointing: bool = True


@dataclass(frozen=True)
class LoraSettings:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
    )


@dataclass(frozen=True)
class SeriousTrainSettings:
    seed: int = 42
    expected_world_size: int = 4
    train_split: str = "train"
    eval_split: str = "validate"
    batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_steps: int | None = None
    max_duration_hours: float = 24.0
    predictor_learning_rate: float = 1.0e-5
    lora_learning_rate: float = 2.0e-5
    query_learning_rate: float = 1.0e-4
    warmup_fraction: float = 0.03
    min_lr_ratio: float = 0.1
    weight_decay: float = 1.0e-8
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    log_every: int = 10
    checkpoint_interval_hours: float = 8.0
    eval_interval_hours: float = 8.0
    eval_at_start: bool = True
    max_eval_examples: int | None = 256


@dataclass(frozen=True)
class SeriousConfig:
    predictor: PredictorConfig
    qwen: QwenSettings
    lora: LoraSettings
    train: SeriousTrainSettings

    def to_dict(self) -> dict[str, Any]:
        return {
            "predictor": self.predictor.to_dict(),
            "qwen": asdict(self.qwen),
            "lora": asdict(self.lora),
            "train": asdict(self.train),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Qwen LoRA, learned transition queries, and a VLA-JEPA "
            "predictor from online current-only conditioning and cached V-JEPA states"
        )
    )
    parser.add_argument("--forecast-manifest", type=Path, required=True)
    parser.add_argument("--state-features", type=Path, required=True)
    parser.add_argument("--eval-forecast-manifest", type=Path, required=True)
    parser.add_argument("--eval-state-features", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Optional smoke-test cap; the 24-hour wall-clock cap remains active",
    )
    return parser.parse_args()


def validate_config(config: SeriousConfig) -> None:
    qwen = config.qwen
    lora = config.lora
    train = config.train
    if qwen.query_tokens <= 0:
        raise ValueError("qwen.query_tokens must be positive")
    if qwen.query_tokens != config.predictor.query_tokens:
        raise ValueError("Qwen and predictor query token counts must match")
    if qwen.image_size != 224 or qwen.max_length <= qwen.query_tokens:
        raise ValueError("Qwen image size must be 224 and max length must fit queries")
    if lora.rank <= 0 or lora.alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    if not 0.0 <= lora.dropout < 1.0:
        raise ValueError("LoRA dropout must be in [0, 1)")
    if not lora.target_modules:
        raise ValueError("LoRA target_modules must not be empty")
    if train.train_split == train.eval_split:
        raise ValueError("train_split and eval_split must differ")
    if train.expected_world_size != 4:
        raise ValueError("serious training is intentionally configured for four GPUs")
    if train.train_split not in {"train", "validate", "test"}:
        raise ValueError("invalid train_split")
    if train.eval_split not in {"train", "validate", "test"}:
        raise ValueError("invalid eval_split")
    if train.batch_size <= 0 or train.eval_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if train.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if train.max_steps is not None and train.max_steps <= 0:
        raise ValueError("max_steps must be positive or null")
    if not 0.0 < train.max_duration_hours <= 24.0:
        raise ValueError("max_duration_hours must be in (0, 24]")
    if train.checkpoint_interval_hours <= 0 or train.eval_interval_hours <= 0:
        raise ValueError("checkpoint/eval intervals must be positive")
    if train.checkpoint_interval_hours > train.max_duration_hours:
        raise ValueError("checkpoint interval cannot exceed max duration")
    if train.eval_interval_hours > train.max_duration_hours:
        raise ValueError("eval interval cannot exceed max duration")
    if train.log_every <= 0:
        raise ValueError("log_every must be positive")
    if train.grad_clip <= 0:
        raise ValueError("grad_clip must be positive")
    if train.max_eval_examples is not None and train.max_eval_examples < 2:
        raise ValueError("max_eval_examples must be at least 2 or null")
    for name, value in (
        ("predictor_learning_rate", train.predictor_learning_rate),
        ("lora_learning_rate", train.lora_learning_rate),
        ("query_learning_rate", train.query_learning_rate),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0.0 <= train.warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    if not 0.0 < train.min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in (0, 1]")


def load_config(path: str | Path) -> SeriousConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    required = {"predictor", "qwen", "lora", "train"}
    if not isinstance(raw, dict) or set(raw) != required:
        raise TypeError(f"config must contain exactly {sorted(required)}")
    if not all(isinstance(raw[name], dict) for name in required):
        raise TypeError("each config section must be an object")
    lora_values = dict(raw["lora"])
    if "target_modules" in lora_values:
        lora_values["target_modules"] = tuple(lora_values["target_modules"])
    train_values = dict(raw["train"])
    if "betas" in train_values:
        train_values["betas"] = tuple(train_values["betas"])
    config = SeriousConfig(
        predictor=PredictorConfig(**raw["predictor"]),
        qwen=QwenSettings(**raw["qwen"]),
        lora=LoraSettings(**lora_values),
        train=SeriousTrainSettings(**train_values),
    )
    validate_config(config)
    return config


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_cache_identity(state_manifest: Path) -> dict[str, Any]:
    metadata_path = state_manifest.parent / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"state cache metadata is required beside the manifest: {metadata_path}"
        )
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema_version") != "mimic-vla-jepa-states-v1"
    ):
        raise ValueError(f"invalid state cache metadata: {metadata_path}")
    return {
        "manifest": str(state_manifest.resolve()),
        "manifest_sha256": file_sha256(state_manifest),
        "metadata": str(metadata_path.resolve()),
        "metadata_sha256": file_sha256(metadata_path),
        "metadata_schema": metadata["schema_version"],
        "source_manifest": metadata.get("source_manifest"),
        "records": metadata.get("records"),
        "backbone": metadata.get("backbone"),
    }


def _qwen_artifact_identity(model_path: Path) -> dict[str, Any]:
    resolved = model_path.resolve()
    config_path = resolved / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Qwen config is missing: {config_path}")
    weight_paths = sorted(resolved.glob("*.safetensors"))
    if not weight_paths:
        raise FileNotFoundError(f"Qwen weights are missing under {resolved}")
    weights = [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in weight_paths
    ]
    aggregate = hashlib.sha256(
        json.dumps(weights, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    input_contract_names = (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    )
    input_contract = {
        name: file_sha256(resolved / name)
        for name in input_contract_names
        if (resolved / name).is_file()
    }
    required_input_files = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "preprocessor_config.json",
    }
    missing = sorted(required_input_files - set(input_contract))
    if missing:
        raise FileNotFoundError(
            f"Qwen input-contract files are missing under {resolved}: {missing}"
        )
    return {
        "source": str(resolved),
        "config_sha256": file_sha256(config_path),
        "weights_sha256": aggregate,
        "weight_files": weights,
        "input_contract_sha256": input_contract,
    }


def build_input_identity(
    *,
    train_forecast_manifest: Path,
    train_state_manifest: Path,
    eval_forecast_manifest: Path,
    eval_state_manifest: Path,
    qwen_model: Path,
) -> dict[str, Any]:
    """Strongly identify immutable inputs without hashing every state tensor."""

    train_state_cache = _state_cache_identity(train_state_manifest)
    eval_state_cache = _state_cache_identity(eval_state_manifest)
    for label, raw_manifest, state_cache in (
        ("train", train_forecast_manifest, train_state_cache),
        ("eval", eval_forecast_manifest, eval_state_cache),
    ):
        source_manifest = state_cache.get("source_manifest")
        if not isinstance(source_manifest, str) or (
            Path(source_manifest).resolve(strict=False)
            != raw_manifest.resolve(strict=False)
        ):
            raise ValueError(
                f"{label} state metadata source_manifest does not match its raw manifest"
            )
    if train_state_cache.get("backbone") != eval_state_cache.get("backbone"):
        raise ValueError(
            "train and evaluation state caches use different V-JEPA backbones "
            "or preprocessing contracts"
        )
    return {
        "train_forecast_manifest": str(train_forecast_manifest.resolve()),
        "train_forecast_manifest_sha256": file_sha256(train_forecast_manifest),
        "train_state_cache": train_state_cache,
        "eval_forecast_manifest": str(eval_forecast_manifest.resolve()),
        "eval_forecast_manifest_sha256": file_sha256(eval_forecast_manifest),
        "eval_state_cache": eval_state_cache,
        "qwen": _qwen_artifact_identity(qwen_model),
        "state_tensor_integrity_assumption": (
            "state manifests and metadata are hashed; referenced safetensors "
            "must remain immutable and read-only for the duration of the run"
        ),
    }


def _truncate_metric_rows(
    rows: list[dict[str, Any]], *, checkpoint_step: int, checkpoint_elapsed: float
) -> list[dict[str, Any]]:
    """Discard log events newer than the resumed checkpoint."""

    retained: list[dict[str, Any]] = []
    for row in rows:
        try:
            row_step = int(row["step"])
            row_elapsed = float(row["elapsed_seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        if row_step <= checkpoint_step and row_elapsed <= checkpoint_elapsed + 1.0e-6:
            retained.append(row)
    return retained


def _remove_stale_run_artifacts(
    run_dir: Path, *, checkpoint_step: int, checkpoint_elapsed: float
) -> None:
    """Remove generated result/evaluation artifacts ahead of a resumed checkpoint."""

    paths = [*run_dir.glob("evaluation*.json"), run_dir / "result.json"]
    for path in paths:
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            step = int(value["step"])
            elapsed = float(value["elapsed_seconds"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if step > checkpoint_step or elapsed > checkpoint_elapsed + 1.0e-6:
            path.unlink()


def next_interval_seconds(elapsed_seconds: float, interval_hours: float) -> float:
    if elapsed_seconds < 0 or interval_hours <= 0:
        raise ValueError("elapsed time must be nonnegative and interval positive")
    interval_seconds = interval_hours * 3600.0
    return (math.floor(elapsed_seconds / interval_seconds) + 1) * interval_seconds


def consume_interval(
    elapsed_seconds: float, next_due_seconds: float, interval_hours: float
) -> tuple[bool, float]:
    if interval_hours <= 0:
        raise ValueError("interval_hours must be positive")
    if elapsed_seconds < next_due_seconds:
        return False, next_due_seconds
    interval_seconds = interval_hours * 3600.0
    crossed = math.floor((elapsed_seconds - next_due_seconds) / interval_seconds) + 1
    return True, next_due_seconds + crossed * interval_seconds


def should_stop(
    *,
    step: int,
    elapsed_seconds: float,
    max_steps: int | None,
    max_duration_hours: float,
) -> bool:
    return (max_steps is not None and step >= max_steps) or (
        elapsed_seconds >= max_duration_hours * 3600.0
    )


class WallClockCosineScheduler:
    """Warm up and cosine-decay learning rates using cumulative job time.

    A step-count schedule is ill-defined for this run because online Qwen
    throughput can change with report length.  Cumulative elapsed time is
    already checkpointed, so this schedule is stable across exact resume.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        max_duration_hours: float,
        warmup_fraction: float,
        min_lr_ratio: float,
    ):
        self.optimizer = optimizer
        self.max_duration_seconds = max_duration_hours * 3600.0
        self.warmup_fraction = warmup_fraction
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.last_elapsed_seconds = 0.0

    def lr_multiplier(self, elapsed_seconds: float) -> float:
        progress = min(max(elapsed_seconds / self.max_duration_seconds, 0.0), 1.0)
        if self.warmup_fraction > 0 and progress < self.warmup_fraction:
            return progress / self.warmup_fraction
        cosine_progress = (
            (progress - self.warmup_fraction) / (1.0 - self.warmup_fraction)
            if self.warmup_fraction < 1.0
            else 1.0
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def step(self, elapsed_seconds: float) -> None:
        if elapsed_seconds < self.last_elapsed_seconds:
            raise ValueError("wall-clock scheduler elapsed time cannot go backwards")
        multiplier = self.lr_multiplier(elapsed_seconds)
        for group, base_lr in zip(
            self.optimizer.param_groups, self.base_lrs, strict=True
        ):
            group["lr"] = base_lr * multiplier
        self.last_elapsed_seconds = elapsed_seconds

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "wall-clock-cosine-v1",
            "max_duration_seconds": self.max_duration_seconds,
            "warmup_fraction": self.warmup_fraction,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": self.base_lrs,
            "last_elapsed_seconds": self.last_elapsed_seconds,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "schema_version": "wall-clock-cosine-v1",
            "max_duration_seconds": self.max_duration_seconds,
            "warmup_fraction": self.warmup_fraction,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": self.base_lrs,
        }
        for name, value in expected.items():
            if state.get(name) != value:
                raise ValueError(f"scheduler checkpoint field {name!r} differs")
        elapsed = float(state["last_elapsed_seconds"])
        self.last_elapsed_seconds = 0.0
        self.step(elapsed)


class SkipBatchSampler:
    """Skip sampler batches without loading/collating them during exact resume."""

    def __init__(self, batch_sampler: BatchSampler, skip_batches: int = 0):
        self.batch_sampler = batch_sampler
        self.skip_batches = skip_batches

    def __iter__(self):
        for batch_index, indices in enumerate(self.batch_sampler):
            if batch_index >= self.skip_batches:
                yield indices

    def __len__(self) -> int:
        return max(len(self.batch_sampler) - self.skip_batches, 0)


def distributed_context() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("serious online training requires CUDA")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group(backend="nccl", device_id=device)
    return rank, world_size, local_rank, device


def seed_everything(seed: int, rank: int) -> None:
    value = seed + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def synchronized_elapsed_seconds(
    *,
    previous_elapsed_seconds: float,
    session_start_monotonic: float,
    rank: int,
    world_size: int,
    device: torch.device,
) -> float:
    elapsed = (
        previous_elapsed_seconds + time.monotonic() - session_start_monotonic
        if rank == 0
        else 0.0
    )
    if world_size > 1:
        value = torch.tensor(elapsed, dtype=torch.float64, device=device)
        dist.broadcast(value, src=0)
        elapsed = float(value.item())
    return elapsed


def truncate_prompt_to_token_budget(
    tokenizer: Any,
    prompt: str,
    *,
    max_tokens: int,
    forbidden_special_token: str,
) -> str:
    """Right-truncate only report/prompt text, never image/query scaffolding."""

    if max_tokens <= 0:
        raise ValueError("prompt token budget must be positive")
    prompt = prompt.replace(
        forbidden_special_token, forbidden_special_token.replace("|", " | ")
    )
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return prompt
    return tokenizer.decode(
        token_ids[:max_tokens],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


class SeriousBatchCollator:
    """Build Qwen inputs solely from source image and regenerated current prompt."""

    def __init__(self, processor: Any, settings: QwenSettings, query_token_id: int):
        self.processor = processor
        self.settings = settings
        self.query_token_id = query_token_id
        # A 224px Qwen image contributes 49 merged visual placeholders.  The
        # remaining reserve covers chat scaffolding, labels, and boundary
        # effects.  Only prompt/report tokens are shortened; automatic sequence
        # truncation is disabled so image placeholders and all query markers
        # can never be silently dropped.
        self.max_prompt_tokens = settings.max_length - settings.query_tokens - 256
        if self.max_prompt_tokens <= 0:
            raise ValueError("Qwen max_length leaves no safe prompt budget")

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if not examples:
            raise ValueError("cannot collate an empty batch")
        suffix = self.settings.query_token * self.settings.query_tokens
        conversations: list[list[dict[str, Any]]] = []
        images: list[Image.Image] = []
        for example in examples:
            image = load_letterboxed_rgb(
                Path(example["source_image"]), self.settings.image_size
            )
            images.append(image)
            prompt = truncate_prompt_to_token_budget(
                self.processor.tokenizer,
                str(example["prompt"]),
                max_tokens=self.max_prompt_tokens,
                forbidden_special_token=self.settings.query_token,
            )
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {
                                "type": "text",
                                "text": (
                                    f"{prompt}\nLATENT TRANSITION QUERIES:\n{suffix}"
                                ),
                            },
                        ],
                    }
                ]
            )
        try:
            inputs = self.processor.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={
                    "text_kwargs": {
                        "padding": True,
                        "truncation": False,
                    },
                    "images_kwargs": {
                        "min_pixels": self.settings.image_size**2,
                        "max_pixels": self.settings.image_size**2,
                    },
                },
            )
        finally:
            for image in images:
                image.close()

        if "input_ids" not in inputs:
            raise ValueError("Qwen processor did not return input_ids")
        attention_mask = inputs.get("attention_mask")
        sequence_lengths = (
            attention_mask.sum(dim=1)
            if attention_mask is not None
            else torch.full(
                (inputs["input_ids"].shape[0],),
                inputs["input_ids"].shape[1],
            )
        )
        if int(sequence_lengths.max()) > self.settings.max_length:
            raise ValueError(
                "Qwen sequence exceeds max_length after prompt-only clipping; "
                "increase the fixed scaffolding reserve"
            )
        positions = batched_query_positions(
            inputs["input_ids"], self.query_token_id, self.settings.query_tokens
        )
        expected_grid = torch.tensor([1, 14, 14])
        grids = inputs.get("image_grid_thw")
        if (
            grids is None
            or grids.ndim != 2
            or grids.shape[0] != len(examples)
            or not torch.all(grids.cpu() == expected_grid)
        ):
            raise ValueError(
                f"expected one Qwen image grid [1, 14, 14] per example, got {grids}"
            )
        return {
            "dataset_index": torch.tensor(
                [int(example["dataset_index"]) for example in examples],
                dtype=torch.long,
            ),
            "transition_id": [str(example["transition_id"]) for example in examples],
            "source_state": torch.stack(
                [example["source_state"] for example in examples]
            ),
            "target_state": torch.stack(
                [example["target_state"] for example in examples]
            ),
            "query_positions": positions,
            "qwen_inputs": {
                key: value for key, value in inputs.items() if torch.is_tensor(value)
            },
        }


class SeriousForecastModel(nn.Module):
    """Online Qwen-LoRA conditioning followed by the VLA-JEPA predictor."""

    def __init__(
        self,
        *,
        qwen_model: str | Path,
        config: SeriousConfig,
        device: torch.device,
    ):
        super().__init__()
        self.config = config
        self.qwen_source = str(Path(qwen_model).resolve())
        self.processor = AutoProcessor.from_pretrained(
            qwen_model, local_files_only=True
        )
        self.processor.tokenizer.padding_side = "left"
        self.processor.tokenizer.truncation_side = "right"
        added = self.processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": [config.qwen.query_token]}
        )
        self.query_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            config.qwen.query_token
        )
        if self.query_token_id == self.processor.tokenizer.unk_token_id:
            raise RuntimeError("failed to register the latent query marker")

        self.qwen = Qwen3_5ForConditionalGeneration.from_pretrained(
            qwen_model,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        if added:
            self.qwen.resize_token_embeddings(
                len(self.processor.tokenizer), mean_resizing=False
            )
            with torch.no_grad():
                embeddings = self.qwen.get_input_embeddings().weight
                embeddings[self.query_token_id].copy_(
                    embeddings[self.processor.tokenizer.eos_token_id]
                )
        self.qwen.requires_grad_(False)
        self.qwen.config.use_cache = False
        self.qwen.config.text_config.use_cache = False

        base_text_model = self.qwen.model.language_model
        lora_config = LoraConfig(
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=list(config.lora.target_modules),
            bias="none",
        )
        self.qwen.model.language_model = get_peft_model(base_text_model, lora_config)
        if config.qwen.gradient_checkpointing:
            self.qwen.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        hidden_size = int(self.qwen.config.text_config.hidden_size)
        if hidden_size != config.predictor.action_dim:
            raise ValueError(
                f"Qwen hidden size {hidden_size} does not match predictor "
                f"action_dim {config.predictor.action_dim}"
            )
        base_embedding = (
            self.qwen.get_input_embeddings()
            .weight[self.processor.tokenizer.eos_token_id]
            .detach()
            .float()
        )
        initializer_range = float(self.qwen.config.text_config.initializer_range)
        initial_queries = base_embedding.unsqueeze(0).repeat(
            config.qwen.query_tokens, 1
        )
        initial_queries = initial_queries + torch.randn_like(initial_queries) * (
            0.1 * initializer_range
        )
        self.query_embeddings = nn.Parameter(initial_queries)
        self.predictor = VLAJEPAPredictor(config.predictor)
        self.to(device)
        self.qwen.model.visual.eval()

        lora_names = [
            name
            for name, parameter in self.qwen.model.language_model.named_parameters()
            if parameter.requires_grad
        ]
        if not lora_names or not all("lora_" in name for name in lora_names):
            raise RuntimeError(
                "Qwen trainable parameters must be a nonempty LoRA-only set; "
                f"got {lora_names[:8]}"
            )

    @property
    def text_model(self) -> nn.Module:
        return self.qwen.model.language_model

    def train(self, mode: bool = True) -> SeriousForecastModel:
        super().train(mode)
        # The image tower is conditioning-only and must remain deterministic.
        self.qwen.model.visual.eval()
        return self

    def encode_queries(
        self,
        qwen_inputs: dict[str, torch.Tensor],
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = qwen_inputs["input_ids"]
        batch_size = input_ids.shape[0]
        if tuple(query_positions.shape) != (
            batch_size,
            self.config.qwen.query_tokens,
        ):
            raise ValueError("query_positions shape does not match the Qwen batch")

        inputs_embeds = self.qwen.get_input_embeddings()(input_ids)
        inputs_embeds = inputs_embeds.clone()
        batch_indices = torch.arange(batch_size, device=input_ids.device)[:, None]
        learned_queries = self.query_embeddings.to(inputs_embeds.dtype)[None].expand(
            batch_size, -1, -1
        )
        inputs_embeds[batch_indices, query_positions] = learned_queries

        image_grid_thw = qwen_inputs.get("image_grid_thw")
        pixel_values = qwen_inputs.get("pixel_values")
        if pixel_values is None or image_grid_thw is None:
            raise ValueError("each serious-training Qwen batch must contain images")
        # The visual tower is a fixed current-image conditioner.  An explicit
        # no-grad boundary both documents that contract and prevents an
        # accidental activation graph if processor behavior changes later.
        with torch.no_grad():
            image_outputs = self.qwen.model.get_image_features(
                pixel_values, image_grid_thw, return_dict=True
            )
        image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        image_mask, _ = self.qwen.model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        position_ids = self.qwen.model.compute_3d_position_ids(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_grid_thw=image_grid_thw,
            attention_mask=qwen_inputs.get("attention_mask"),
            mm_token_type_ids=qwen_inputs.get("mm_token_type_ids"),
        )
        outputs = self.text_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=qwen_inputs.get("attention_mask"),
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        return hidden[batch_indices, query_positions]

    def forward(
        self,
        source_state: torch.Tensor,
        qwen_inputs: dict[str, torch.Tensor],
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        query_state = self.encode_queries(qwen_inputs, query_positions)
        return self.predictor(source_state, query_state)

    def trainable_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        lora = [
            parameter
            for parameter in self.text_model.parameters()
            if parameter.requires_grad
        ]
        predictor = [
            parameter
            for parameter in self.predictor.parameters()
            if parameter.requires_grad
        ]
        groups = {
            "lora": lora,
            "queries": [self.query_embeddings],
            "predictor": predictor,
        }
        parameter_ids = [
            id(parameter) for values in groups.values() for parameter in values
        ]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise RuntimeError("trainable parameter groups overlap")
        actual_ids = {
            id(parameter) for parameter in self.parameters() if parameter.requires_grad
        }
        if actual_ids != set(parameter_ids):
            raise RuntimeError(
                "some trainable parameters are missing from optimizer groups"
            )
        return groups

    def component_state_dict(self) -> dict[str, Any]:
        def cpu_tensors(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return {name: value.detach().cpu() for name, value in values.items()}

        return {
            "lora": cpu_tensors(get_peft_model_state_dict(self.text_model)),
            "query_embeddings": self.query_embeddings.detach().cpu(),
            "predictor": cpu_tensors(self.predictor.state_dict()),
        }

    def load_component_state_dict(self, state: dict[str, Any]) -> None:
        if set(state) != {"lora", "query_embeddings", "predictor"}:
            raise ValueError("checkpoint trainable component keys are invalid")
        set_peft_model_state_dict(self.text_model, state["lora"])
        if tuple(state["query_embeddings"].shape) != tuple(self.query_embeddings.shape):
            raise ValueError("checkpoint query embedding shape differs")
        with torch.no_grad():
            self.query_embeddings.copy_(state["query_embeddings"])
        self.predictor.load_state_dict(state["predictor"], strict=True)


def _move_batch_to_device(
    batch: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    return {
        **batch,
        "dataset_index": batch["dataset_index"].to(device, non_blocking=True),
        "source_state": batch["source_state"].to(device, non_blocking=True),
        "target_state": batch["target_state"].to(device, non_blocking=True),
        "query_positions": batch["query_positions"].to(device, non_blocking=True),
        "qwen_inputs": {
            name: value.to(device, non_blocking=True)
            for name, value in batch["qwen_inputs"].items()
        },
    }


def _capture_rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device),
    }


def _restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"], device)


def _atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    secure_directory(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(value, temporary_path)
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_checkpoint(
    path: Path,
    *,
    model: SeriousForecastModel,
    optimizer: torch.optim.Optimizer,
    scheduler: WallClockCosineScheduler,
    config: SeriousConfig,
    step: int,
    epoch: int,
    batch_in_epoch: int,
    elapsed_seconds: float,
    rank: int,
    world_size: int,
    device: torch.device,
    input_identity: dict[str, Any],
    last_evaluation: dict[str, Any] | None,
) -> None:
    local_rng = _capture_rng_state(device)
    rng_states: list[dict[str, Any]] | None = (
        [None] * world_size if rank == 0 else None  # type: ignore[list-item]
    )
    if world_size > 1:
        dist.gather_object(local_rng, rng_states, dst=0)
    else:
        rng_states = [local_rng]
    if rank == 0:
        _atomic_torch_save(
            path,
            {
                "schema_version": "mimic-vla-jepa-serious-v1",
                "step": step,
                "epoch": epoch,
                "batch_in_epoch": batch_in_epoch,
                "elapsed_seconds": elapsed_seconds,
                "world_size": world_size,
                "config": config.to_dict(),
                "input_identity": input_identity,
                "trainable_components": model.component_state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng_states": rng_states,
                "last_evaluation": last_evaluation,
            },
        )
    if world_size > 1:
        dist.barrier()


EVAL_VARIANTS = ("prediction", "copy_state", "zero_query", "shuffled_query")
EVAL_PAIRED = (
    "shuffled_minus_correct_l1",
    "zero_minus_correct_l1",
    "prediction_change_l1_shuffled",
    "prediction_change_l1_zero",
)


def _evaluation_batch_sums(
    *,
    prediction: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    zero_prediction: torch.Tensor,
    shuffled_prediction: torch.Tensor,
) -> torch.Tensor:
    variants = (prediction, source, zero_prediction, shuffled_prediction)
    values = [
        torch.tensor(float(target.shape[0]), dtype=torch.float64, device=target.device)
    ]
    l1_values: list[torch.Tensor] = []
    for value in variants:
        l1 = (value.float() - target.float()).abs().flatten(1).mean(dim=1)
        cosine = 1.0 - F.cosine_similarity(
            value.float().flatten(1), target.float().flatten(1), dim=-1
        )
        l1_values.append(l1)
        values.extend((l1.double().sum(), cosine.double().sum()))
    correct_l1, _, zero_l1, shuffled_l1 = l1_values
    values.extend(
        (
            (shuffled_l1 - correct_l1).double().sum(),
            (zero_l1 - correct_l1).double().sum(),
            (shuffled_prediction.float() - prediction.float())
            .abs()
            .flatten(1)
            .mean(dim=1)
            .double()
            .sum(),
            (zero_prediction.float() - prediction.float())
            .abs()
            .flatten(1)
            .mean(dim=1)
            .double()
            .sum(),
        )
    )
    return torch.stack(values)


def _finalize_evaluation(sums: torch.Tensor) -> dict[str, Any]:
    expected = 1 + 2 * len(EVAL_VARIANTS) + len(EVAL_PAIRED)
    if sums.ndim != 1 or sums.numel() != expected:
        raise ValueError(f"invalid evaluation sums shape: {tuple(sums.shape)}")
    examples = round(float(sums[0].item()))
    if examples < 2:
        raise ValueError("evaluation requires at least two examples")
    metrics: dict[str, dict[str, float]] = {}
    offset = 1
    for name in EVAL_VARIANTS:
        metrics[name] = {
            "l1": float(sums[offset].item() / examples),
            "cosine_distance": float(sums[offset + 1].item() / examples),
        }
        offset += 2
    metrics["prediction"]["copy_normalized_gain"] = 1.0 - (
        metrics["prediction"]["l1"] / max(metrics["copy_state"]["l1"], 1.0e-12)
    )
    paired = {
        name: float(sums[offset + index].item() / examples)
        for index, name in enumerate(EVAL_PAIRED)
    }
    return {"examples": examples, "metrics": metrics, "paired_query_effect": paired}


def stable_evaluation_indices(
    dataset: SeriousTransitionDataset, max_examples: int | None
) -> list[int]:
    """Select any capped evaluation subset without trusting manifest order."""

    indices = list(range(len(dataset.records)))
    if max_examples is not None and max_examples < len(indices):
        indices.sort(
            key=lambda index: (
                hashlib.sha256(
                    dataset.records[index].transition_id.encode("utf-8")
                ).digest(),
                dataset.records[index].transition_id,
            )
        )
        indices = indices[:max_examples]
    # Traversal and the cyclic shuffled-query mapping are independent of the
    # target-derived order that may have existed in an upstream manifest.
    indices.sort(key=lambda index: dataset.records[index].transition_id)
    return indices


def patient_disjoint_shuffle_map(
    dataset: SeriousTransitionDataset, selected_indices: list[int]
) -> dict[int, int]:
    """Build a deterministic cyclic condition shuffle with a different patient."""

    if len(selected_indices) < 2:
        raise ValueError("query shuffling requires at least two examples")
    orders = [
        selected_indices,
        sorted(
            selected_indices,
            key=lambda index: (
                dataset.records[index].patient_id,
                dataset.records[index].transition_id,
            ),
        ),
    ]
    for order in orders:
        for shift in range(1, len(order)):
            if all(
                dataset.records[index].patient_id
                != dataset.records[order[(position + shift) % len(order)]].patient_id
                for position, index in enumerate(order)
            ):
                mapping = {
                    index: order[(position + shift) % len(order)]
                    for position, index in enumerate(order)
                }
                if set(mapping) != set(selected_indices) or set(
                    mapping.values()
                ) != set(selected_indices):
                    raise AssertionError("patient-disjoint shuffle is not a bijection")
                return mapping
    raise ValueError(
        "evaluation selection has no patient-disjoint query permutation; "
        "one patient may dominate the selection"
    )


@torch.no_grad()
def evaluate(
    *,
    model: SeriousForecastModel,
    dataset: SeriousTransitionDataset,
    collator: SeriousBatchCollator,
    batch_size: int,
    max_examples: int | None,
    step: int,
    elapsed_seconds: float,
    rank: int,
    world_size: int,
    device: torch.device,
    output_path: Path,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    selected_indices = stable_evaluation_indices(dataset, max_examples)
    total_examples = len(selected_indices)
    if total_examples < 2:
        raise ValueError("evaluation split must have at least two joined examples")
    local_indices = selected_indices[rank::world_size]
    shuffled_index = patient_disjoint_shuffle_map(dataset, selected_indices)
    loader = DataLoader(
        Subset(dataset, local_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collator,
    )
    sums = torch.zeros(
        1 + 2 * len(EVAL_VARIANTS) + len(EVAL_PAIRED),
        dtype=torch.float64,
        device=device,
    )
    for cpu_batch in loader:
        original_indices = [int(value) for value in cpu_batch["dataset_index"]]
        shuffled_examples = [
            dataset[shuffled_index[index]] for index in original_indices
        ]
        shuffled_batch = _move_batch_to_device(collator(shuffled_examples), device)
        batch = _move_batch_to_device(cpu_batch, device)
        source = batch["source_state"]
        target = batch["target_state"]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            query = model.encode_queries(batch["qwen_inputs"], batch["query_positions"])
            shuffled_query = model.encode_queries(
                shuffled_batch["qwen_inputs"], shuffled_batch["query_positions"]
            )
            prediction = model.predictor(source, query)
            zero_prediction = model.predictor(source, torch.zeros_like(query))
            shuffled_prediction = model.predictor(source, shuffled_query)
        sums += _evaluation_batch_sums(
            prediction=prediction,
            source=source,
            target=target,
            zero_prediction=zero_prediction,
            shuffled_prediction=shuffled_prediction,
        )
    if world_size > 1:
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    result = {
        "schema_version": "mimic-vla-jepa-serious-evaluation-v1",
        "step": step,
        "elapsed_seconds": elapsed_seconds,
        "selection": (
            "all"
            if max_examples is None or max_examples >= len(dataset)
            else "stable_sha256_transition_id"
        ),
        **_finalize_evaluation(sums.cpu()),
    }
    if rank == 0:
        atomic_write_json(output_path, result)
    if world_size > 1:
        dist.barrier()
    if was_training:
        model.train()
    return result


def _optimizer_for_model(
    model: SeriousForecastModel, settings: SeriousTrainSettings
) -> torch.optim.Optimizer:
    groups = model.trainable_parameter_groups()
    return torch.optim.AdamW(
        [
            {
                "name": "predictor",
                "params": groups["predictor"],
                "lr": settings.predictor_learning_rate,
                "weight_decay": settings.weight_decay,
            },
            {
                "name": "qwen_lora",
                "params": groups["lora"],
                "lr": settings.lora_learning_rate,
                "weight_decay": settings.weight_decay,
            },
            {
                "name": "learned_queries",
                "params": groups["queries"],
                "lr": settings.query_learning_rate,
                "weight_decay": 0.0,
            },
        ],
        betas=settings.betas,
    )


def _parameter_summary(model: SeriousForecastModel) -> dict[str, int]:
    groups = model.trainable_parameter_groups()
    result = {
        name: sum(parameter.numel() for parameter in parameters)
        for name, parameters in groups.items()
    }
    result["total_trainable"] = sum(result.values())
    result["total_model"] = sum(parameter.numel() for parameter in model.parameters())
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max-steps must be positive")
        config = replace(
            config,
            train=replace(config.train, max_steps=args.max_steps),
        )
        validate_config(config)

    rank, world_size, _, device = distributed_context()
    try:
        if world_size != config.train.expected_world_size:
            raise ValueError(
                f"serious config requires {config.train.expected_world_size} ranks, "
                f"but torchrun started {world_size}"
            )
        seed_everything(config.train.seed, rank)
        project_root = Path(__file__).resolve().parents[2]
        checkpoint_dir = args.checkpoint_dir or (
            project_root / "checkpoints" / "trained" / args.run_dir.name
        )
        identity_holder: list[dict[str, Any] | None] = [None]
        if rank == 0:
            identity_holder[0] = build_input_identity(
                train_forecast_manifest=args.forecast_manifest,
                train_state_manifest=args.state_features,
                eval_forecast_manifest=args.eval_forecast_manifest,
                eval_state_manifest=args.eval_state_features,
                qwen_model=args.qwen_model,
            )
        if world_size > 1:
            dist.broadcast_object_list(identity_holder, src=0)
        input_identity = identity_holder[0]
        if input_identity is None:
            raise RuntimeError("rank zero did not broadcast the input identity")

        if args.resume is None:
            if args.run_dir.exists() and any(args.run_dir.iterdir()):
                raise FileExistsError(
                    f"refusing non-resume run in nonempty directory: {args.run_dir}"
                )
            if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
                raise FileExistsError(
                    "refusing non-resume run in nonempty checkpoint directory: "
                    f"{checkpoint_dir}"
                )
        if rank == 0:
            secure_directory(args.run_dir)
            secure_directory(checkpoint_dir)
        if world_size > 1:
            dist.barrier()

        train_dataset = SeriousTransitionDataset(
            args.forecast_manifest,
            args.state_features,
            split=config.train.train_split,
        )
        eval_dataset = SeriousTransitionDataset(
            args.eval_forecast_manifest,
            args.eval_state_features,
            split=config.train.eval_split,
        )
        model = SeriousForecastModel(
            qwen_model=args.qwen_model,
            config=config,
            device=device,
        )
        collator = SeriousBatchCollator(
            model.processor, config.qwen, model.query_token_id
        )
        optimizer = _optimizer_for_model(model, config.train)
        scheduler = WallClockCosineScheduler(
            optimizer,
            max_duration_hours=config.train.max_duration_hours,
            warmup_fraction=config.train.warmup_fraction,
            min_lr_ratio=config.train.min_lr_ratio,
        )

        step = 0
        start_epoch = 0
        start_batch_in_epoch = 0
        previous_elapsed_seconds = 0.0
        last_evaluation: dict[str, Any] | None = None
        resume_rng_state: dict[str, Any] | None = None
        if args.resume is not None:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
            if checkpoint.get("schema_version") != "mimic-vla-jepa-serious-v1":
                raise ValueError("resume checkpoint is not serious schema v1")
            if checkpoint.get("world_size") != world_size:
                raise ValueError("resume world size differs from checkpoint")
            if checkpoint.get("config") != config.to_dict():
                raise ValueError("resume config differs from checkpoint")
            if checkpoint.get("input_identity") != input_identity:
                raise ValueError("resume input manifests or Qwen model differ")
            rng_states = checkpoint.get("rng_states")
            if not isinstance(rng_states, list) or len(rng_states) != world_size:
                raise ValueError("checkpoint lacks one RNG state per rank")
            model.load_component_state_dict(checkpoint["trainable_components"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            step = int(checkpoint["step"])
            start_epoch = int(checkpoint["epoch"])
            start_batch_in_epoch = int(checkpoint["batch_in_epoch"])
            previous_elapsed_seconds = float(checkpoint["elapsed_seconds"])
            last_evaluation = checkpoint.get("last_evaluation")
            resume_rng_state = rng_states[rank]

        ddp_model: DistributedDataParallel | SeriousForecastModel
        if world_size > 1:
            ddp_model = DistributedDataParallel(
                model,
                device_ids=[device.index],
                output_device=device.index,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            ddp_model = model

        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config.train.seed,
            drop_last=False,
        )
        base_batch_sampler = BatchSampler(
            sampler,
            batch_size=config.train.batch_size,
            drop_last=False,
        )
        resumable_batch_sampler = SkipBatchSampler(
            base_batch_sampler, skip_batches=start_batch_in_epoch
        )
        loader_generator = torch.Generator()
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=resumable_batch_sampler,
            num_workers=0,
            pin_memory=True,
            collate_fn=collator,
            generator=loader_generator,
        )
        total_batches_per_epoch = len(base_batch_sampler)
        if total_batches_per_epoch == 0:
            raise ValueError("training split is empty on this rank")
        if not 0 <= start_batch_in_epoch < total_batches_per_epoch:
            raise ValueError("resume batch position is outside its sampler epoch")

        if resume_rng_state is not None:
            _restore_rng_state(resume_rng_state, device)

        metrics_path = args.run_dir / "metrics.jsonl"
        metric_rows: list[dict[str, Any]] = []
        if args.resume is not None and metrics_path.is_file():
            with metrics_path.open("r", encoding="utf-8") as handle:
                metric_rows = [json.loads(line) for line in handle if line.strip()]
            metric_rows = _truncate_metric_rows(
                metric_rows,
                checkpoint_step=step,
                checkpoint_elapsed=previous_elapsed_seconds,
            )
        if rank == 0:
            if args.resume is not None:
                _remove_stale_run_artifacts(
                    args.run_dir,
                    checkpoint_step=step,
                    checkpoint_elapsed=previous_elapsed_seconds,
                )
                atomic_write_text(metrics_path, jsonl_text(metric_rows))
            atomic_write_json(
                args.run_dir / "run.json",
                {
                    "schema_version": "mimic-vla-jepa-serious-run-v1",
                    "config": config.to_dict(),
                    "input_identity": input_identity,
                    "world_size": world_size,
                    "train_examples": len(train_dataset),
                    "eval_examples": len(eval_dataset),
                    "parameters": _parameter_summary(model),
                    "checkpoint_dir": str(checkpoint_dir.resolve()),
                    "resume": str(args.resume.resolve()) if args.resume else None,
                },
            )

        next_checkpoint = next_interval_seconds(
            previous_elapsed_seconds, config.train.checkpoint_interval_hours
        )
        next_eval = next_interval_seconds(
            previous_elapsed_seconds, config.train.eval_interval_hours
        )

        if config.train.eval_at_start and step == 0 and previous_elapsed_seconds == 0:
            elapsed = previous_elapsed_seconds
            last_evaluation = evaluate(
                model=model,
                dataset=eval_dataset,
                collator=collator,
                batch_size=config.train.eval_batch_size,
                max_examples=config.train.max_eval_examples,
                step=step,
                elapsed_seconds=elapsed,
                rank=rank,
                world_size=world_size,
                device=device,
                output_path=args.run_dir / "evaluation_step-000000_initial.json",
            )
            if rank == 0:
                metric_rows.append(
                    {"event": "evaluation", "phase": "initial", **last_evaluation}
                )
                atomic_write_text(metrics_path, jsonl_text(metric_rows))

        # The 24-hour clock measures active online-training wall time. Required
        # startup evaluation, periodic evaluation, and checkpoint I/O are
        # paused so an exact resume neither loses nor invents optimizer
        # exposure merely because setup or serialization took longer.
        session_start_monotonic = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch = start_epoch
        resume_epoch = start_epoch
        batch_in_epoch = start_batch_in_epoch
        micro_in_accumulation = 0
        accumulated_loss = torch.zeros((), dtype=torch.float64, device=device)
        initial_elapsed = synchronized_elapsed_seconds(
            previous_elapsed_seconds=previous_elapsed_seconds,
            session_start_monotonic=session_start_monotonic,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        stop = should_stop(
            step=step,
            elapsed_seconds=initial_elapsed,
            max_steps=config.train.max_steps,
            max_duration_hours=config.train.max_duration_hours,
        )
        last_eval_step = int(last_evaluation["step"]) if last_evaluation else -1

        while not stop:
            sampler.set_epoch(epoch)
            loader_generator.manual_seed(config.train.seed + epoch)
            skip_batches = start_batch_in_epoch if epoch == start_epoch else 0
            resumable_batch_sampler.skip_batches = skip_batches
            for current_batch, cpu_batch in enumerate(train_loader, start=skip_batches):
                batch = _move_batch_to_device(cpu_batch, device)
                will_step = (
                    micro_in_accumulation + 1
                    == config.train.gradient_accumulation_steps
                )
                sync_context = (
                    nullcontext()
                    if will_step or not isinstance(ddp_model, DistributedDataParallel)
                    else ddp_model.no_sync()
                )
                with sync_context:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        prediction = ddp_model(
                            batch["source_state"],
                            batch["qwen_inputs"],
                            batch["query_positions"],
                        )
                    loss = F.l1_loss(prediction.float(), batch["target_state"].float())
                    finite_loss = torch.tensor(
                        int(bool(torch.isfinite(loss))),
                        dtype=torch.int32,
                        device=device,
                    )
                    if world_size > 1:
                        dist.all_reduce(finite_loss, op=dist.ReduceOp.MIN)
                    if not bool(finite_loss):
                        raise FloatingPointError(
                            "non-finite loss detected on at least one rank"
                        )
                    (loss / config.train.gradient_accumulation_steps).backward()
                accumulated_loss += loss.detach().double()
                micro_in_accumulation += 1
                batch_in_epoch = current_batch + 1
                resume_epoch = epoch
                if batch_in_epoch >= total_batches_per_epoch:
                    batch_in_epoch = 0
                    resume_epoch = epoch + 1
                if not will_step:
                    continue

                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    config.train.grad_clip,
                    error_if_nonfinite=True,
                )
                elapsed = synchronized_elapsed_seconds(
                    previous_elapsed_seconds=previous_elapsed_seconds,
                    session_start_monotonic=session_start_monotonic,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                )
                scheduler.step(elapsed)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                micro_in_accumulation = 0
                mean_loss = accumulated_loss / config.train.gradient_accumulation_steps
                accumulated_loss.zero_()
                if world_size > 1:
                    dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
                    mean_loss /= world_size

                elapsed = synchronized_elapsed_seconds(
                    previous_elapsed_seconds=previous_elapsed_seconds,
                    session_start_monotonic=session_start_monotonic,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                )
                terminal_after_step = should_stop(
                    step=step,
                    elapsed_seconds=elapsed,
                    max_steps=config.train.max_steps,
                    max_duration_hours=config.train.max_duration_hours,
                )
                if step % config.train.log_every == 0 and rank == 0:
                    row = {
                        "event": "train",
                        "step": step,
                        "epoch": epoch,
                        "batch_in_epoch": batch_in_epoch,
                        "elapsed_seconds": elapsed,
                        "loss_l1": float(mean_loss.item()),
                        "gradient_norm": float(gradient_norm),
                        "learning_rates": {
                            str(group["name"]): float(group["lr"])
                            for group in optimizer.param_groups
                        },
                    }
                    metric_rows.append(row)
                    atomic_write_text(metrics_path, jsonl_text(metric_rows))
                    print(row, flush=True)

                eval_due = False
                checkpoint_due = False
                if not terminal_after_step:
                    eval_due, next_eval = consume_interval(
                        elapsed, next_eval, config.train.eval_interval_hours
                    )
                    checkpoint_due, next_checkpoint = consume_interval(
                        elapsed,
                        next_checkpoint,
                        config.train.checkpoint_interval_hours,
                    )
                if eval_due:
                    pause_started = time.monotonic()
                    last_evaluation = evaluate(
                        model=model,
                        dataset=eval_dataset,
                        collator=collator,
                        batch_size=config.train.eval_batch_size,
                        max_examples=config.train.max_eval_examples,
                        step=step,
                        elapsed_seconds=elapsed,
                        rank=rank,
                        world_size=world_size,
                        device=device,
                        output_path=(args.run_dir / f"evaluation_step-{step:06d}.json"),
                    )
                    last_eval_step = step
                    if rank == 0:
                        metric_rows.append({"event": "evaluation", **last_evaluation})
                        atomic_write_text(metrics_path, jsonl_text(metric_rows))
                        print(last_evaluation, flush=True)
                    session_start_monotonic += time.monotonic() - pause_started
                if checkpoint_due:
                    elapsed = synchronized_elapsed_seconds(
                        previous_elapsed_seconds=previous_elapsed_seconds,
                        session_start_monotonic=session_start_monotonic,
                        rank=rank,
                        world_size=world_size,
                        device=device,
                    )
                    pause_started = time.monotonic()
                    save_checkpoint(
                        checkpoint_dir / f"checkpoint_step-{step:06d}.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=config,
                        step=step,
                        epoch=resume_epoch,
                        batch_in_epoch=batch_in_epoch,
                        elapsed_seconds=elapsed,
                        rank=rank,
                        world_size=world_size,
                        device=device,
                        input_identity=input_identity,
                        last_evaluation=last_evaluation,
                    )
                    session_start_monotonic += time.monotonic() - pause_started
                elapsed = synchronized_elapsed_seconds(
                    previous_elapsed_seconds=previous_elapsed_seconds,
                    session_start_monotonic=session_start_monotonic,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                )
                stop = terminal_after_step
                if stop:
                    break
            if not stop:
                epoch += 1
                start_batch_in_epoch = 0

        elapsed = synchronized_elapsed_seconds(
            previous_elapsed_seconds=previous_elapsed_seconds,
            session_start_monotonic=session_start_monotonic,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        if last_eval_step != step:
            pause_started = time.monotonic()
            last_evaluation = evaluate(
                model=model,
                dataset=eval_dataset,
                collator=collator,
                batch_size=config.train.eval_batch_size,
                max_examples=config.train.max_eval_examples,
                step=step,
                elapsed_seconds=elapsed,
                rank=rank,
                world_size=world_size,
                device=device,
                output_path=args.run_dir / "evaluation_final.json",
            )
            if rank == 0:
                metric_rows.append(
                    {"event": "evaluation", "phase": "final", **last_evaluation}
                )
                atomic_write_text(metrics_path, jsonl_text(metric_rows))
            session_start_monotonic += time.monotonic() - pause_started
        elapsed = synchronized_elapsed_seconds(
            previous_elapsed_seconds=previous_elapsed_seconds,
            session_start_monotonic=session_start_monotonic,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        save_checkpoint(
            checkpoint_dir / "checkpoint_last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            step=step,
            epoch=resume_epoch,
            batch_in_epoch=batch_in_epoch,
            elapsed_seconds=elapsed,
            rank=rank,
            world_size=world_size,
            device=device,
            input_identity=input_identity,
            last_evaluation=last_evaluation,
        )
        if rank == 0:
            atomic_write_json(
                args.run_dir / "result.json",
                {
                    "schema_version": "mimic-vla-jepa-serious-result-v1",
                    "step": step,
                    "elapsed_seconds": elapsed,
                    "checkpoint": str(
                        (checkpoint_dir / "checkpoint_last.pt").resolve()
                    ),
                    "evaluation": last_evaluation,
                },
            )
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
