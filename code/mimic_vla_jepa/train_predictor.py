from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from .features import CachedFeatureDataset
from .io_utils import atomic_write_json, atomic_write_text, secure_directory
from .predictor import PredictorConfig, VLAJEPAPredictor


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    batch_size: int = 1
    max_steps: int | None = 5
    max_duration_hours: float | None = None
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-8
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    log_every: int = 1
    checkpoint_every: int = 0
    checkpoint_interval_hours: float | None = None
    eval_interval_hours: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the VLA-JEPA predictor from restricted local feature caches"
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Defaults to <project>/checkpoints/trained/<run-dir-name>",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-duration-hours", type=float)
    parser.add_argument("--checkpoint-interval-hours", type=float)
    parser.add_argument("--eval-interval-hours", type=float)
    parser.add_argument("--eval-features", type=Path)
    parser.add_argument("--eval-split", default="validate")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def load_config(path: Path) -> tuple[PredictorConfig, TrainConfig, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if (
        not isinstance(raw, dict)
        or not isinstance(raw.get("predictor"), dict)
        or not isinstance(raw.get("train"), dict)
    ):
        raise TypeError("config must contain predictor and train objects")
    predictor = PredictorConfig(**raw["predictor"])
    train_values = dict(raw["train"])
    if "betas" in train_values:
        train_values["betas"] = tuple(train_values["betas"])
    train = TrainConfig(**train_values)
    validate_train_config(train)
    return predictor, train, raw


def validate_train_config(config: TrainConfig) -> None:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.max_steps is not None and config.max_steps < 0:
        raise ValueError("max_steps must be non-negative or null")
    if config.max_duration_hours is not None and (
        not math.isfinite(config.max_duration_hours) or config.max_duration_hours <= 0
    ):
        raise ValueError("max_duration_hours must be positive or null")
    if config.max_steps is None and config.max_duration_hours is None:
        raise ValueError("at least one of max_steps or max_duration_hours is required")
    if config.log_every <= 0:
        raise ValueError("log_every must be positive")
    if config.checkpoint_every < 0:
        raise ValueError("checkpoint_every must be non-negative")
    for name, value in (
        ("checkpoint_interval_hours", config.checkpoint_interval_hours),
        ("eval_interval_hours", config.eval_interval_hours),
    ):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be positive or null")


def next_interval_seconds(
    elapsed_seconds: float, interval_hours: float | None
) -> float | None:
    """Return the first absolute interval boundary strictly after elapsed time."""

    if interval_hours is None:
        return None
    if not math.isfinite(interval_hours) or interval_hours <= 0:
        raise ValueError("interval_hours must be positive")
    interval_seconds = interval_hours * 3600.0
    completed = int(elapsed_seconds // interval_seconds)
    return (completed + 1) * interval_seconds


def interval_is_due(
    elapsed_seconds: float,
    next_due_seconds: float | None,
    interval_hours: float | None,
) -> tuple[bool, float | None]:
    """Consume all crossed boundaries and return the following boundary."""

    if next_due_seconds is None:
        return False, None
    if (
        interval_hours is None
        or not math.isfinite(interval_hours)
        or interval_hours <= 0
    ):
        raise ValueError("an enabled interval must be positive")
    if elapsed_seconds < next_due_seconds:
        return False, next_due_seconds
    interval_seconds = interval_hours * 3600.0
    crossed = int((elapsed_seconds - next_due_seconds) // interval_seconds) + 1
    return True, next_due_seconds + crossed * interval_seconds


def training_should_stop(
    *,
    step: int,
    elapsed_seconds: float,
    max_steps: int | None,
    max_duration_hours: float | None,
) -> bool:
    step_limit = max_steps is not None and step >= max_steps
    time_limit = (
        max_duration_hours is not None
        and elapsed_seconds >= max_duration_hours * 3600.0
    )
    return step_limit or time_limit


MUTABLE_RESUME_FIELDS = {
    "max_steps",
    "max_duration_hours",
    "log_every",
    "checkpoint_every",
    "checkpoint_interval_hours",
    "eval_interval_hours",
}


def validate_resume_train_config(
    checkpoint_train: dict[str, Any], current_train: TrainConfig
) -> None:
    """Keep optimization semantics fixed while allowing scheduling changes."""

    for field, value in asdict(current_train).items():
        if field in MUTABLE_RESUME_FIELDS:
            continue
        if checkpoint_train.get(field) != value:
            raise ValueError(
                f"resume train config field {field!r} does not match the checkpoint"
            )


def distributed_context() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
    return rank, world_size, local_rank


def seed_everything(seed: int, rank: int) -> None:
    value = seed + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    result = value.detach().float()
    if world_size > 1:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result /= world_size
    return result


def synchronized_elapsed_seconds(
    *,
    starting_elapsed_seconds: float,
    session_start_monotonic: float,
    rank: int,
    world_size: int,
    device: torch.device,
) -> float:
    """Use rank zero's monotonic clock as the authoritative job clock."""

    value = (
        starting_elapsed_seconds + time.monotonic() - session_start_monotonic
        if rank == 0
        else 0.0
    )
    if world_size > 1:
        tensor = torch.tensor(value, dtype=torch.float64, device=device)
        dist.broadcast(tensor, src=0)
        value = float(tensor.item())
    return value


EVALUATION_VARIANTS = (
    "prediction",
    "copy_state",
    "zero_query",
    "shuffled_query",
)
PAIRED_QUERY_METRICS = (
    "shuffled_minus_correct_l1",
    "zero_minus_correct_l1",
    "prediction_change_l1_shuffled",
    "prediction_change_l1_zero",
)


def evaluation_batch_sums(
    *,
    prediction: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    zero_prediction: torch.Tensor,
    shuffled_prediction: torch.Tensor,
) -> torch.Tensor:
    """Produce additive evaluation statistics, suitable for DDP reduction."""

    variants = (prediction, source, zero_prediction, shuffled_prediction)
    values: list[torch.Tensor] = [
        torch.tensor(float(target.shape[0]), dtype=torch.float64, device=target.device)
    ]
    per_variant_l1: list[torch.Tensor] = []
    for value in variants:
        per_l1 = (value.float() - target.float()).abs().flatten(1).mean(dim=1)
        per_cosine = 1.0 - F.cosine_similarity(
            value.float().flatten(1), target.float().flatten(1), dim=-1
        )
        per_variant_l1.append(per_l1)
        values.extend((per_l1.double().sum(), per_cosine.double().sum()))

    correct_l1, _, zero_l1, shuffled_l1 = per_variant_l1
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


def finalize_evaluation_sums(sums: torch.Tensor) -> dict[str, Any]:
    expected_values = 1 + 2 * len(EVALUATION_VARIANTS) + len(PAIRED_QUERY_METRICS)
    if sums.ndim != 1 or sums.numel() != expected_values:
        raise ValueError(
            f"expected {expected_values} evaluation sums, got {tuple(sums.shape)}"
        )
    examples = round(float(sums[0].item()))
    if examples <= 0:
        raise ValueError("evaluation requires at least one example")
    offset = 1
    metrics: dict[str, dict[str, float]] = {}
    for name in EVALUATION_VARIANTS:
        metrics[name] = {
            "l1": float(sums[offset].item() / examples),
            "cosine_distance": float(sums[offset + 1].item() / examples),
        }
        offset += 2
    metrics["prediction"]["copy_normalized_gain"] = 1.0 - (
        metrics["prediction"]["l1"] / max(metrics["copy_state"]["l1"], 1.0e-12)
    )
    paired_query_effect = {
        name: float(sums[offset + index].item() / examples)
        for index, name in enumerate(PAIRED_QUERY_METRICS)
    }
    return {
        "examples": examples,
        "metrics": metrics,
        "paired_query_effect": paired_query_effect,
    }


def evaluate_distributed(
    *,
    model: VLAJEPAPredictor | DistributedDataParallel,
    dataset: CachedFeatureDataset,
    batch_size: int,
    device: torch.device,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    """Evaluate each validation example exactly once across all ranks."""

    if len(dataset) < 2:
        raise ValueError("query shuffling requires at least two evaluation examples")
    if batch_size <= 0:
        raise ValueError("eval batch size must be positive")
    local_indices = list(range(rank, len(dataset), world_size))
    loader = DataLoader(
        Subset(dataset, local_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    index_by_id = {
        record.transition_id: index for index, record in enumerate(dataset.records)
    }
    bare_model = model.module if isinstance(model, DistributedDataParallel) else model
    was_training = bare_model.training
    bare_model.eval()
    total_sums = torch.zeros(
        1 + 2 * len(EVALUATION_VARIANTS) + len(PAIRED_QUERY_METRICS),
        dtype=torch.float64,
        device=device,
    )
    with torch.inference_mode():
        for batch in loader:
            source = batch["source_state"].to(
                device, dtype=torch.float32, non_blocking=True
            )
            target = batch["target_state"].to(
                device, dtype=torch.float32, non_blocking=True
            )
            query = batch["query_state"].to(
                device, dtype=torch.float32, non_blocking=True
            )
            shuffled_indices = [
                (index_by_id[transition_id] - 1) % len(dataset)
                for transition_id in batch["transition_id"]
            ]
            shuffled_query = torch.stack(
                [dataset[index]["query_state"] for index in shuffled_indices]
            ).to(device, dtype=torch.float32, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = bare_model(source, query)
                zero_prediction = bare_model(source, torch.zeros_like(query))
                shuffled_prediction = bare_model(source, shuffled_query)
            total_sums += evaluation_batch_sums(
                prediction=prediction,
                source=source,
                target=target,
                zero_prediction=zero_prediction,
                shuffled_prediction=shuffled_prediction,
            )
    if world_size > 1:
        dist.all_reduce(total_sums, op=dist.ReduceOp.SUM)
    bare_model.train(was_training)
    return finalize_evaluation_sums(total_sums.cpu())


def git_metadata(path: Path) -> dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None, "tracked_diff_sha256": None}
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=path, text=True
    )
    tracked_diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD", "--"], cwd=path
    )
    return {
        "revision": revision,
        "dirty": bool(status.strip()),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest()
        if tracked_diff
        else None,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])


def save_checkpoint(
    path: Path,
    model: VLAJEPAPredictor | DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    batch_in_epoch: int,
    rank: int,
    world_size: int,
    feature_manifest_sha256: str,
    predictor_config: PredictorConfig,
    train_config: TrainConfig,
    cumulative_elapsed_seconds: float,
) -> None:
    rank_rng_state = local_rng_state()
    if world_size > 1:
        gathered_rng_states = [None] * world_size if rank == 0 else None
        dist.gather_object(rank_rng_state, gathered_rng_states, dst=0)
    else:
        gathered_rng_states = [rank_rng_state]
    if rank != 0:
        return

    bare_model = model.module if isinstance(model, DistributedDataParallel) else model
    temporary_path = path.with_name(f".{path.name}.tmp")
    torch.save(
        {
            "schema_version": "mimic-vla-jepa-predictor-v3",
            "step": step,
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "cumulative_elapsed_seconds": cumulative_elapsed_seconds,
            "world_size": world_size,
            "feature_manifest_sha256": feature_manifest_sha256,
            "rng_states": gathered_rng_states,
            "model": bare_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "predictor_config": predictor_config.to_dict(),
            "train_config": asdict(train_config),
        },
        temporary_path,
    )
    os.chmod(temporary_path, 0o600)
    os.replace(temporary_path, path)


def append_jsonl_row(
    path: Path, rows: list[dict[str, Any]], row: dict[str, Any]
) -> None:
    rows.append(row)
    atomic_write_text(
        path,
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in rows),
    )


def alias_checkpoint(source: Path, destination: Path) -> None:
    """Atomically give an already-saved final state the conventional last name."""

    temporary_path = destination.with_name(f".{destination.name}.tmp")
    temporary_path.unlink(missing_ok=True)
    os.link(source, temporary_path)
    os.replace(temporary_path, destination)


def main() -> None:
    args = parse_args()
    code_root = Path(__file__).resolve().parents[1]
    project_root = code_root.parent
    checkpoint_dir = args.checkpoint_dir or (
        project_root / "checkpoints" / "trained" / args.run_dir.name
    )
    predictor_config, train_config, raw_config = load_config(args.config)
    train_overrides = {
        name: value
        for name, value in (
            ("max_steps", args.max_steps),
            ("max_duration_hours", args.max_duration_hours),
            ("checkpoint_interval_hours", args.checkpoint_interval_hours),
            ("eval_interval_hours", args.eval_interval_hours),
        )
        if value is not None
    }
    if train_overrides:
        train_config = TrainConfig(**{**asdict(train_config), **train_overrides})
    validate_train_config(train_config)
    if train_config.eval_interval_hours is not None and args.eval_features is None:
        raise ValueError("eval_interval_hours requires --eval-features")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")
    rank, world_size, local_rank = distributed_context()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("predictor training requires a CUDA GPU")
    torch.cuda.set_device(device)
    seed_everything(train_config.seed, rank)

    if args.run_dir.exists() and any(args.run_dir.iterdir()) and args.resume is None:
        if world_size > 1:
            dist.destroy_process_group()
        raise FileExistsError(
            f"refusing a non-resume run in nonempty directory: {args.run_dir}"
        )
    if (
        checkpoint_dir.exists()
        and any(checkpoint_dir.iterdir())
        and args.resume is None
    ):
        if world_size > 1:
            dist.destroy_process_group()
        raise FileExistsError(
            f"refusing a non-resume run in nonempty checkpoint directory: {checkpoint_dir}"
        )
    if rank == 0:
        secure_directory(args.run_dir)
        if args.checkpoint_dir is None:
            secure_directory(checkpoint_dir.parent)
        secure_directory(checkpoint_dir)
    if world_size > 1:
        dist.barrier()

    dataset = CachedFeatureDataset(args.features, split=args.split)
    eval_dataset = (
        CachedFeatureDataset(args.eval_features, split=args.eval_split)
        if args.eval_features is not None
        else None
    )
    if eval_dataset is not None and len(eval_dataset) < 2:
        raise ValueError("query shuffling requires at least two evaluation examples")
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=train_config.seed,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    model = VLAJEPAPredictor(predictor_config).to(device)
    if world_size > 1:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], broadcast_buffers=False
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=train_config.betas,
        weight_decay=train_config.weight_decay,
    )

    feature_manifest_sha256 = file_sha256(args.features)
    start_step = 0
    start_epoch = 0
    start_batch_in_epoch = 0
    starting_elapsed_seconds = 0.0
    resume_checkpoint_sha256 = None
    if args.resume is not None:
        resume_checkpoint_sha256 = file_sha256(args.resume)
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("schema_version") not in {
            "mimic-vla-jepa-predictor-v2",
            "mimic-vla-jepa-predictor-v3",
        }:
            raise ValueError(
                "exact resume requires a v2 or v3 checkpoint with sampler and RNG state"
            )
        if checkpoint.get("predictor_config") != predictor_config.to_dict():
            raise ValueError("resume predictor config does not match the checkpoint")
        if int(checkpoint.get("world_size", -1)) != world_size:
            raise ValueError("exact resume requires the original world size")
        if checkpoint.get("feature_manifest_sha256") != feature_manifest_sha256:
            raise ValueError("resume feature manifest does not match the checkpoint")
        checkpoint_train = dict(checkpoint.get("train_config", {}))
        validate_resume_train_config(checkpoint_train, train_config)
        bare_model = (
            model.module if isinstance(model, DistributedDataParallel) else model
        )
        bare_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        start_epoch = int(checkpoint["epoch"])
        start_batch_in_epoch = int(checkpoint["batch_in_epoch"])
        starting_elapsed_seconds = float(
            checkpoint.get("cumulative_elapsed_seconds", 0.0)
        )
        if starting_elapsed_seconds < 0:
            raise ValueError("checkpoint cumulative elapsed time is negative")
        rng_states = checkpoint.get("rng_states")
        if not isinstance(rng_states, list) or len(rng_states) != world_size:
            raise ValueError("checkpoint does not contain one RNG state per rank")
        restore_rng_state(rng_states[rank])

    if rank == 0:
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        atomic_write_json(
            args.run_dir / "run.json",
            {
                "schema_version": "mimic-vla-jepa-run-v1",
                "features": str(args.features.resolve()),
                "checkpoint_dir": str(checkpoint_dir.resolve()),
                "feature_manifest_sha256": feature_manifest_sha256,
                "config": raw_config,
                "resolved_train": asdict(train_config),
                "resolved_predictor": predictor_config.to_dict(),
                "dataset_records": len(dataset),
                "eval_features": str(args.eval_features.resolve())
                if args.eval_features
                else None,
                "eval_feature_manifest_sha256": file_sha256(args.eval_features)
                if args.eval_features
                else None,
                "eval_split": args.eval_split if args.eval_features else None,
                "eval_records": len(eval_dataset) if eval_dataset is not None else 0,
                "eval_batch_size": args.eval_batch_size,
                "world_size": world_size,
                "trainable_parameters": trainable,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "implementation_repository": git_metadata(code_root),
                "predictor_reference_repository": git_metadata(
                    code_root / "VLA-JEPA-reference"
                ),
                "resumed_from": {
                    "path": str(args.resume.resolve()),
                    "sha256_before_resume": resume_checkpoint_sha256,
                    "step": start_step,
                    "cumulative_elapsed_seconds": starting_elapsed_seconds,
                }
                if args.resume
                else None,
            },
        )

    model.train()
    log_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    metrics_path = args.run_dir / "metrics.jsonl"
    evaluations_path = args.run_dir / "evaluations.jsonl"
    if rank == 0 and args.resume is not None:
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as handle:
                log_rows = [json.loads(line) for line in handle if line.strip()]
        if evaluations_path.exists():
            with evaluations_path.open("r", encoding="utf-8") as handle:
                evaluation_rows = [json.loads(line) for line in handle if line.strip()]

    step = start_step
    epoch = start_epoch
    batch_in_epoch = start_batch_in_epoch
    next_epoch = epoch
    next_batch_in_epoch = batch_in_epoch
    session_start_monotonic = time.monotonic()

    def elapsed_now() -> float:
        return synchronized_elapsed_seconds(
            starting_elapsed_seconds=starting_elapsed_seconds,
            session_start_monotonic=session_start_monotonic,
            rank=rank,
            world_size=world_size,
            device=device,
        )

    def record_evaluation(stage: str, elapsed_seconds: float) -> None:
        assert eval_dataset is not None
        result = evaluate_distributed(
            model=model,
            dataset=eval_dataset,
            batch_size=args.eval_batch_size,
            device=device,
            rank=rank,
            world_size=world_size,
        )
        if rank == 0:
            row = {
                "schema_version": "mimic-vla-jepa-training-evaluation-v1",
                "stage": stage,
                "step": step,
                "epoch": epoch,
                "cumulative_elapsed_seconds": elapsed_seconds,
                "features": str(args.eval_features.resolve()),
                "split": args.eval_split,
                **result,
            }
            append_jsonl_row(evaluations_path, evaluation_rows, row)
            print(json.dumps({"evaluation": row}, sort_keys=True), flush=True)

    last_evaluated_step: int | None = None
    if eval_dataset is not None:
        already_evaluated = False
        if rank == 0:
            already_evaluated = any(
                int(row.get("step", -1)) == step for row in evaluation_rows
            )
        if world_size > 1:
            already_tensor = torch.tensor(
                int(already_evaluated), dtype=torch.int64, device=device
            )
            dist.broadcast(already_tensor, src=0)
            already_evaluated = bool(already_tensor.item())
        if not already_evaluated:
            record_evaluation("resume" if args.resume else "baseline", elapsed_now())
        last_evaluated_step = step

    elapsed_seconds = elapsed_now()
    next_eval_seconds = next_interval_seconds(
        elapsed_seconds, train_config.eval_interval_hours
    )
    next_checkpoint_seconds = next_interval_seconds(
        elapsed_seconds, train_config.checkpoint_interval_hours
    )
    stop_training = training_should_stop(
        step=step,
        elapsed_seconds=elapsed_seconds,
        max_steps=train_config.max_steps,
        max_duration_hours=train_config.max_duration_hours,
    )
    last_checkpoint_step: int | None = None
    last_checkpoint_path: Path | None = None

    while not stop_training:
        sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(loader):
            if batch_index < batch_in_epoch:
                continue
            elapsed_seconds = elapsed_now()
            if training_should_stop(
                step=step,
                elapsed_seconds=elapsed_seconds,
                max_steps=train_config.max_steps,
                max_duration_hours=train_config.max_duration_hours,
            ):
                stop_training = True
                break

            source = batch["source_state"].to(
                device, dtype=torch.float32, non_blocking=True
            )
            target = batch["target_state"].to(
                device, dtype=torch.float32, non_blocking=True
            )
            query = batch["query_state"].to(
                device, dtype=torch.float32, non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = model(source, query)
                loss = F.l1_loss(prediction, target, reduction="mean")
                copy_loss = F.l1_loss(source, target, reduction="mean")
                cosine = (
                    1.0
                    - F.cosine_similarity(
                        prediction.float().flatten(1), target.float().flatten(1), dim=-1
                    ).mean()
                )
                copy_cosine = (
                    1.0
                    - F.cosine_similarity(
                        source.float().flatten(1), target.float().flatten(1), dim=-1
                    ).mean()
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at step {step + 1}: {loss.item()}"
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config.grad_clip
            )
            optimizer.step()
            step += 1
            if batch_index + 1 == len(loader):
                next_epoch = epoch + 1
                next_batch_in_epoch = 0
            else:
                next_epoch = epoch
                next_batch_in_epoch = batch_index + 1

            mean_loss = reduce_mean(loss, world_size)
            mean_copy_loss = reduce_mean(copy_loss, world_size)
            mean_cosine = reduce_mean(cosine, world_size)
            mean_copy_cosine = reduce_mean(copy_cosine, world_size)
            mean_grad = reduce_mean(
                torch.as_tensor(grad_norm, device=device), world_size
            )
            elapsed_seconds = elapsed_now()
            stop_after_step = training_should_stop(
                step=step,
                elapsed_seconds=elapsed_seconds,
                max_steps=train_config.max_steps,
                max_duration_hours=train_config.max_duration_hours,
            )
            if rank == 0 and (step % train_config.log_every == 0 or stop_after_step):
                row = {
                    "step": step,
                    "epoch": epoch,
                    "cumulative_elapsed_seconds": elapsed_seconds,
                    "loss_l1": float(mean_loss.item()),
                    "copy_l1": float(mean_copy_loss.item()),
                    "copy_normalized_gain": float(
                        1.0 - mean_loss.item() / max(mean_copy_loss.item(), 1.0e-12)
                    ),
                    "cosine_distance": float(mean_cosine.item()),
                    "copy_cosine_distance": float(mean_copy_cosine.item()),
                    "grad_norm": float(mean_grad.item()),
                    "peak_memory_gib": torch.cuda.max_memory_allocated(device)
                    / (1024**3),
                }
                append_jsonl_row(metrics_path, log_rows, row)
                print(json.dumps(row, sort_keys=True), flush=True)

            time_checkpoint_due, next_checkpoint_seconds = interval_is_due(
                elapsed_seconds,
                next_checkpoint_seconds,
                train_config.checkpoint_interval_hours,
            )
            time_eval_due, next_eval_seconds = interval_is_due(
                elapsed_seconds,
                next_eval_seconds,
                train_config.eval_interval_hours,
            )
            step_checkpoint_due = bool(
                train_config.checkpoint_every
                and step % train_config.checkpoint_every == 0
            )

            # A final checkpoint/evaluation below represents this exact state,
            # so do not emit duplicate interval artifacts when a limit is hit.
            if stop_after_step:
                stop_training = True
                break

            if step_checkpoint_due or time_checkpoint_due:
                checkpoint_path = checkpoint_dir / f"checkpoint_step-{step:06d}.pt"
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    step,
                    next_epoch,
                    next_batch_in_epoch,
                    rank,
                    world_size,
                    feature_manifest_sha256,
                    predictor_config,
                    train_config,
                    elapsed_seconds,
                )
                last_checkpoint_step = step
                last_checkpoint_path = checkpoint_path

            if time_eval_due and eval_dataset is not None:
                record_evaluation("interval", elapsed_seconds)
                last_evaluated_step = step

            # Evaluation and checkpoint I/O count toward the requested wall
            # clock budget. Recheck before launching the next optimizer step.
            elapsed_seconds = elapsed_now()
            if training_should_stop(
                step=step,
                elapsed_seconds=elapsed_seconds,
                max_steps=train_config.max_steps,
                max_duration_hours=train_config.max_duration_hours,
            ):
                stop_training = True
                break

        if stop_training:
            break
        epoch += 1
        batch_in_epoch = 0

    final_elapsed_seconds = elapsed_now()
    if eval_dataset is not None and last_evaluated_step != step:
        record_evaluation("final", final_elapsed_seconds)
        last_evaluated_step = step
        final_elapsed_seconds = elapsed_now()

    checkpoint_last_path = checkpoint_dir / "checkpoint_last.pt"
    if last_checkpoint_step == step and last_checkpoint_path is not None:
        if rank == 0:
            alias_checkpoint(last_checkpoint_path, checkpoint_last_path)
    else:
        save_checkpoint(
            checkpoint_last_path,
            model,
            optimizer,
            step,
            next_epoch,
            next_batch_in_epoch,
            rank,
            world_size,
            feature_manifest_sha256,
            predictor_config,
            train_config,
            final_elapsed_seconds,
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
