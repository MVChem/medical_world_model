"""Train one shared, four-task checkpoint; smoke results are never table metrics.

Examples (from the project root)::

    python code/medworld_multitask/run.py --smoke --model qwen08b \
        --condition slots --out code/medworld_multitask/runs/08b_slots_smoke
    python code/medworld_multitask/run.py --steps 2400 --model qwen08b \
        --condition slots --out code/medworld_multitask/runs/08b_slots_train

``--steps`` is the total optimizer-step budget, including resumed steps. A task
gets one optimizer update every four steps. No task-specific checkpoint is used.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
from typing import Any

import numpy as np
import torch


TASKS = ("classification", "report", "segmentation", "sr")
FORMAT_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STOP_REQUESTED = False


def active_tasks(condition: str) -> tuple:
    return ("segmentation", "sr") if condition == "image_only" else TASKS


def add_donors(batch: dict, dataset, seed: int) -> dict:
    """Deterministic donors from the same task/split, including batch size one."""
    donors = []
    for identity, subject in zip(batch["ids"], batch["subject_ids"]):
        key = f"{seed}:{batch['task']}:{batch['split']}:{identity}".encode()
        start = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % len(dataset)
        donor = None
        for offset in range(len(dataset)):
            index = (start + offset) % len(dataset)
            if hasattr(dataset, "rows"):
                if str(dataset.rows[index]["subject_id"]) == str(subject):
                    continue
            candidate = dataset[index]
            if str(candidate["subject_id"]) != str(subject):
                donor = candidate
                break
        if donor is None:
            raise ValueError("Shuffled control needs at least two different patients in the split")
        if donor["task"] != batch["task"] or donor["split"] != batch["split"]:
            raise ValueError("Shuffled donor has a different task/split")
        donors.append(donor["image"])
    if len(donors) != len(batch["images"]):
        raise ValueError("Shuffled donor count differs from input batch size")
    batch["donor_images"] = donors
    return batch


def decoder_hashes(model) -> dict:
    result = {}
    for task in ("segmentation", "sr"):
        if not hasattr(model, task):
            continue
        digest = hashlib.sha256()
        for name, value in sorted(getattr(model, task).state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        result[task] = digest.hexdigest()
    return result


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_torch(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # CUDA kernels may remain nondeterministic; data ordering and RNG are pinned.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state() -> dict:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


class TaskStream:
    """Deterministic, resumable permutations, with no worker prefetch to rewind."""

    def __init__(self, dataset, collate, task: str, seed: int):
        self.dataset, self.collate, self.task, self.seed = dataset, collate, task, seed
        if not len(dataset):
            raise ValueError(f"Empty training dataset: {task}")
        self.epoch = self.position = 0
        self._permutation = None

    def _indices(self):
        if self._permutation is None:
            generator = torch.Generator().manual_seed(self.seed + self.epoch * 1_000_003)
            self._permutation = torch.randperm(len(self.dataset), generator=generator).tolist()
        return self._permutation

    def batch(self, batch_size: int):
        examples = []
        for _ in range(batch_size):
            if self.position == len(self.dataset):
                self.epoch += 1
                self.position = 0
                self._permutation = None
            examples.append(self.dataset[self._indices()[self.position]])
            self.position += 1
        return self.collate(self.task, examples)

    def state_dict(self) -> dict:
        return dict(epoch=self.epoch, position=self.position, seed=self.seed, size=len(self.dataset))

    def load_state_dict(self, state: dict) -> None:
        if state["seed"] != self.seed or state["size"] != len(self.dataset):
            raise ValueError(f"Resume sampler mismatch for {self.task}")
        if state["epoch"] < 0 or not 0 <= state["position"] <= len(self.dataset):
            raise ValueError(f"Invalid sampler position for {self.task}")
        self.epoch, self.position = state["epoch"], state["position"]
        self._permutation = None


def parameter_group(name: str) -> str:
    if "lora_" in name:
        return "lora"
    if "slot" in name or "queries" in name or "query_tokens" in name:
        return "slots"
    if any(part in name for part in ("head", "decoder", "upsample")):
        return "task_heads"
    if any(part in name for part in ("proj", "adapter", "fusion")):
        return "projections"
    return "other"


def trainable_parameters(model) -> dict:
    return {name: value for name, value in model.named_parameters() if value.requires_grad}


def compact_state(model) -> dict:
    return {name: value.detach().cpu().clone() for name, value in trainable_parameters(model).items()}


def restore_compact(model, weights: dict) -> None:
    parameters = trainable_parameters(model)
    if parameters.keys() != weights.keys():
        missing = sorted(parameters.keys() - weights.keys())
        extra = sorted(weights.keys() - parameters.keys())
        raise ValueError(f"Trainable checkpoint keys differ: missing={missing}, extra={extra}")
    with torch.no_grad():
        for name, value in parameters.items():
            saved = weights[name]
            if saved.shape != value.shape or saved.dtype != value.dtype:
                raise ValueError(f"Trainable checkpoint shape/dtype mismatch: {name}")
            value.copy_(saved.to(device=value.device))


def optimizer_for(model, learning_rate: float, lora_learning_rate: float):
    buckets = {}
    for name, value in trainable_parameters(model).items():
        buckets.setdefault(parameter_group(name), []).append(value)
    if not buckets:
        raise ValueError("The model has no trainable parameters")
    return torch.optim.AdamW([
        dict(params=values, name=name, lr=lora_learning_rate if name == "lora" else learning_rate)
        for name, values in sorted(buckets.items())
    ], betas=(0.9, 0.95), weight_decay=0.01)


def gradient_summary(model) -> dict:
    result = {}
    for name, value in trainable_parameters(model).items():
        group = result.setdefault(parameter_group(name), dict(parameters=0, with_grad=0, _norms=[]))
        group["parameters"] += value.numel()
        if value.grad is not None:
            group["with_grad"] += value.numel()
            group["_norms"].append(value.grad.detach().float().norm().square())
    for group in result.values():
        values = group.pop("_norms")
        norm = float(torch.stack(values).sum().sqrt()) if values else 0.0
        if not math.isfinite(norm):
            raise FloatingPointError("Non-finite parameter gradient")
        group["l2_norm"] = norm
        group["nonzero"] = norm > 0
    return result


def memory_usage(device: str) -> dict:
    if torch.device(device).type != "cuda":
        return dict(device=device)
    return dict(device=device, allocated_gib=torch.cuda.memory_allocated(device) / 2**30,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                reserved_gib=torch.cuda.memory_reserved(device) / 2**30)


def autocast_context(device: str):
    return torch.autocast("cuda", dtype=torch.bfloat16) if torch.device(device).type == "cuda" else nullcontext()


def finite_loss(loss, task: str) -> float:
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        raise ValueError(f"{task} loss must be a scalar tensor")
    result = float(loss.detach())
    if not math.isfinite(result):
        raise FloatingPointError(f"Non-finite {task} loss: {result}")
    return result


@torch.no_grad()
def validate(model, data, args) -> dict:
    """Per-task validation loss, never substituted for the paper's task metrics."""
    previous_rng, was_training = rng_state(), model.training
    model.eval()
    result = {}
    try:
        seed_all(args.seed + 70_001)
        for task in active_tasks(args.condition):
            if STOP_REQUESTED:
                break
            dataset = data.dataset(task, "validate")
            count = min(len(dataset), args.validation_batches * args.batch_size)
            if not count:
                raise ValueError(f"Empty validation dataset: {task}")
            # Fixed random subset avoids always selecting the same manifest prefix.
            generator = torch.Generator().manual_seed(args.seed + 80_003 + TASKS.index(task))
            indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
            total = 0.0
            for start in range(0, count, args.batch_size):
                chosen = indices[start:start + args.batch_size]
                batch = data.collate(task, [dataset[index] for index in chosen])
                if args.condition == "shuffled":
                    add_donors(batch, dataset, args.seed)
                with autocast_context(args.device):
                    total += finite_loss(model.loss(task, batch), task) * len(chosen)
            result[task] = dict(loss=total / count, samples=count, split="validate")
    finally:
        model.train(was_training)
        restore_rng(previous_rng)
    return result


def stop_handler(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(json.dumps(dict(event="stop_requested", signal=signum)), flush=True)


def configuration(args) -> dict:
    # Budget and output path may change on resume; optimization/data settings may not.
    return {key: getattr(args, key) for key in (
        "model", "condition", "seed", "batch_size", "accumulation", "learning_rate",
        "lora_learning_rate", "max_grad_norm", "dense_train_n", "smoke",
    )} | {"root": str(Path(args.root).resolve()), "tasks": list(active_tasks(args.condition))}


def prepare_output(args) -> Path:
    out = Path(args.out).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())) and not args.resume:
        raise FileExistsError(f"Output already exists; choose a fresh --out: {out}")
    if args.resume:
        source = Path(args.resume).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if out.exists() and any(out.iterdir()) and source.parent != out:
            raise FileExistsError("A populated --out may only resume its own checkpoint")
    out.mkdir(parents=True, exist_ok=True)
    return out


def source_hashes() -> dict:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))}


def run(args) -> int:
    global STOP_REQUESTED
    STOP_REQUESTED = False
    os.umask(0o077)
    out = prepare_output(args)
    config = configuration(args)
    tasks = active_tasks(args.condition)
    budget = len(tasks) if args.smoke else args.steps
    started, base_elapsed, step = time.monotonic(), 0.0, 0
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    model = optimizer = data = None
    streams, last_validation = {}, {}

    def status(state, **extra):
        record = dict(state=state, mode="smoke" if args.smoke else "training", pid=os.getpid(),
                      step=step, budget_steps=budget, task_updates={task: (step + len(tasks) - 1 - index) // len(tasks)
                      for index, task in enumerate(tasks)}, updated_unix=time.time(),
                      elapsed_seconds=base_elapsed + time.monotonic() - started,
                      table2_ready=False, pending=["official task metrics", "VQA", "MS-CXR grounding"],
                      **extra)
        atomic_json(out / "status.json", record)
        print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)

    def save_checkpoint(name="checkpoint_latest.pt"):
        payload = dict(format_version=FORMAT_VERSION, model_trainable=compact_state(model),
                       model_metadata=model.metadata, optimizer=optimizer.state_dict(),
                       step=step, elapsed_seconds=base_elapsed + time.monotonic() - started,
                       config=config, data_metadata=data.metadata,
                       streams={task: stream.state_dict() for task, stream in streams.items()},
                       rng=rng_state(), last_validation=last_validation,
                       smoke_only=args.smoke, table2_ready=False)
        atomic_torch(out / name, payload)
        return out / name

    try:
        status("initializing")
        seed_all(args.seed)
        from data import MultiTaskData
        from model import MultiTaskModel

        data = MultiTaskData(root=Path(args.root).resolve(), dense_train_n=args.dense_train_n)
        model = MultiTaskModel(model_id=args.model, condition=args.condition, device=args.device)
        if hasattr(model, "set_pos_weight"):
            model.set_pos_weight(data.pos_weight)
        initial_decoder_hashes = decoder_hashes(model)
        optimizer = optimizer_for(model, args.learning_rate, args.lora_learning_rate)
        streams = {task: TaskStream(data.dataset(task, "train"), data.collate, task,
                                    args.seed + 10_007 * (index + 1))
                   for index, task in enumerate(tasks)}
        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
            if checkpoint["format_version"] != FORMAT_VERSION or checkpoint["config"] != config:
                raise ValueError("Resume configuration differs (smoke checkpoints cannot become training runs)")
            if checkpoint["data_metadata"] != data.metadata or checkpoint["model_metadata"] != model.metadata:
                raise ValueError("Resume model/data provenance differs")
            restore_compact(model, checkpoint["model_trainable"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            for task, stream in streams.items():
                stream.load_state_dict(checkpoint["streams"][task])
            step, base_elapsed = checkpoint["step"], checkpoint["elapsed_seconds"]
            last_validation = checkpoint["last_validation"]
            restore_rng(checkpoint["rng"])
            del checkpoint
            if step > budget:
                raise ValueError(f"Checkpoint step {step} exceeds requested total budget {budget}")
        atomic_json(out / "config.json", config | {"budget_steps": budget})
        atomic_json(out / "provenance.json", dict(model=model.metadata, data=data.metadata,
                    source_hashes=source_hashes(), python=sys.version, torch=torch.__version__,
                    cuda=torch.version.cuda, effective_batch_size=args.batch_size * args.accumulation,
                    decoder_initial_sha256=initial_decoder_hashes,
                    shuffled_donors="same task/split, different patient, deterministic per recipient, independent of batch size" if args.condition == "shuffled" else None,
                    checkpoint_scope="all trainable parameters; frozen checkpoints referenced in model metadata",
                    note="Four-task shared checkpoint. Smoke/validation losses are not final Table 2 metrics."))
        atomic_json(out / "trainable_parameters.json", {
            name: dict(shape=list(value.shape), count=value.numel(), dtype=str(value.dtype),
                       group=parameter_group(name)) for name, value in trainable_parameters(model).items()
        })
        model.train()
        if torch.device(args.device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(args.device)
        status("running", memory=memory_usage(args.device))
        last_saved_step = step
        with (out / "metrics.jsonl").open("a", buffering=1) as history:
            while step < budget and not STOP_REQUESTED:
                if args.max_seconds and time.monotonic() - started >= args.max_seconds:
                    STOP_REQUESTED = True
                    break
                task, tick = tasks[step % len(tasks)], time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                total_loss = 0.0
                for _ in range(args.accumulation):
                    batch = streams[task].batch(args.batch_size)
                    if args.condition == "shuffled":
                        add_donors(batch, streams[task].dataset, args.seed)
                    with autocast_context(args.device):
                        loss = model.loss(task, batch)
                    total_loss += finite_loss(loss, task)
                    (loss / args.accumulation).backward()
                gradients = gradient_summary(model)
                branch_gradients = model.gradient_contract(task) if hasattr(model, "gradient_contract") else None
                if not any(group["nonzero"] for group in gradients.values()):
                    raise RuntimeError(f"No nonzero gradients for {task}")
                grad_norm = torch.nn.utils.clip_grad_norm_(list(trainable_parameters(model).values()),
                                                         args.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                step += 1
                event = dict(event="train_step", step=step, task=task,
                             loss=total_loss / args.accumulation,
                             effective_batch_size=args.batch_size * args.accumulation,
                             seconds=time.monotonic() - tick, gradients=gradients,
                             branch_gradients=branch_gradients,
                             gradient_norm_before_clip=float(grad_norm), memory=memory_usage(args.device),
                             smoke_only=args.smoke)
                history.write(json.dumps(event, allow_nan=False) + "\n")
                print(json.dumps(event, allow_nan=False), flush=True)
                if not args.smoke and not STOP_REQUESTED and (step % args.validate_every == 0 or step == budget):
                    last_validation = validate(model, data, args)
                    atomic_json(out / "validation_losses.json", dict(step=step, tasks=last_validation,
                                final_task_metrics=False, split="validate"))
                    history.write(json.dumps(dict(event="validation", step=step, tasks=last_validation)) + "\n")
                if step % args.save_every == 0 or STOP_REQUESTED:
                    save_checkpoint()
                    last_saved_step = step
                status("running", last_task=task, last_loss=event["loss"], memory=event["memory"])
        if last_saved_step != step or not (out / "checkpoint_latest.pt").exists():
            save_checkpoint()
        if STOP_REQUESTED:
            status("interrupted", checkpoint=str(out / "checkpoint_latest.pt"), memory=memory_usage(args.device))
            return 130
        if args.smoke:
            # Compare identical examples on both sides of checkpoint reload.
            model.eval()
            prediction_rng = rng_state()
            before = {}
            with torch.inference_mode():
                for task in tasks:
                    dataset = data.dataset(task, "train")
                    batch = data.collate(task, [dataset[0]])
                    if args.condition == "shuffled":
                        add_donors(batch, dataset, args.seed)
                    with autocast_context(args.device):
                        prediction = model.predict(task, batch, max_new_tokens=args.report_max_new_tokens) if task == "report" else model.predict(task, batch)
                    before[task] = prediction.detach().cpu().clone() if isinstance(prediction, torch.Tensor) else prediction
            # Exercise the same load path as resume, including strict keys and dtypes.
            checkpoint = torch.load(out / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
            restore_compact(model, checkpoint["model_trainable"])
            reloaded = compact_state(model)
            exact = all(torch.equal(reloaded[name], value) for name, value in checkpoint["model_trainable"].items())
            if not exact:
                raise RuntimeError("Saved trainable weights did not reload exactly")
            del reloaded, checkpoint
            restore_rng(prediction_rng)
            predictions = {}
            with torch.inference_mode():
                for task in tasks:
                    dataset = data.dataset(task, "train")
                    batch = data.collate(task, [dataset[0]])
                    if args.condition == "shuffled":
                        add_donors(batch, dataset, args.seed)
                    with autocast_context(args.device):
                        prediction = model.predict(task, batch, max_new_tokens=args.report_max_new_tokens) if task == "report" else model.predict(task, batch)
                    if task == "report":
                        if not isinstance(prediction, (list, tuple)) or not all(isinstance(item, str) for item in prediction):
                            raise TypeError("Report prediction must be a list of strings")
                        if list(prediction) != list(before[task]):
                            raise RuntimeError("Report output changed after checkpoint reload")
                        predictions[task] = dict(count=len(prediction), character_counts=[len(item) for item in prediction],
                                                 max_new_tokens=args.report_max_new_tokens, reload_output_equal=True)
                    else:
                        if not isinstance(prediction, torch.Tensor) or not torch.isfinite(prediction).all():
                            raise FloatingPointError(f"Invalid {task} prediction")
                        torch.testing.assert_close(prediction.cpu(), before[task], rtol=1e-5, atol=1e-6)
                        predictions[task] = dict(shape=list(prediction.shape), finite=True, reload_output_close=True)
            atomic_json(out / "smoke_checks.json", dict(smoke_only=True, table2_ready=False,
                        tasks=list(tasks), optimizer_steps=step, checkpoint_reload_exact=True,
                        decoder_initial_sha256=initial_decoder_hashes,
                        predictions=predictions, memory=memory_usage(args.device),
                        note="Execution checks only. Do not copy losses or predictions into Table 2."))
            status("smoke_complete", checkpoint_reload_exact=True, memory=memory_usage(args.device))
        else:
            status("budget_complete", validation=last_validation, memory=memory_usage(args.device))
        return 0
    except BaseException as error:
        # Do not serialize a partly completed optimizer update after an exception.
        status("failed", error_type=type(error).__name__, error=str(error),
               resume_checkpoint=str(out / "checkpoint_latest.pt") if (out / "checkpoint_latest.pt").exists() else None)
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="Four real-data optimizer steps plus save/reload and predictions")
    mode.add_argument("--steps", type=int, help="Total optimizer-step budget (includes resumed steps; four steps = one task cycle)")
    result.add_argument("--model", choices=("qwen08b", "qwen9b"), default="qwen08b")
    result.add_argument("--condition", choices=("slots", "full_tokens", "shuffled", "image_only"), default="slots")
    result.add_argument("--out", required=True, help="Fresh output directory; populated directories require --resume")
    result.add_argument("--resume", help="Trusted local checkpoint_latest.pt from this runner")
    result.add_argument("--root", default=str(PROJECT_ROOT))
    result.add_argument("--device", default="cuda")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--batch-size", type=int, default=1)
    result.add_argument("--accumulation", type=int, default=1)
    result.add_argument("--learning-rate", type=float, default=1e-4)
    result.add_argument("--lora-learning-rate", type=float, default=2e-5)
    result.add_argument("--max-grad-norm", type=float, default=1.0)
    result.add_argument("--dense-train-n", type=int, default=4096)
    result.add_argument("--save-every", type=int, default=40)
    result.add_argument("--validate-every", type=int, default=200)
    result.add_argument("--validation-batches", type=int, default=8)
    result.add_argument("--report-max-new-tokens", type=int, default=16)
    result.add_argument("--max-seconds", type=float, default=0, help="Optional wall-time limit for this invocation; save as interrupted")
    return result


def main(argv=None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    for name in ("steps", "batch_size", "accumulation", "dense_train_n", "save_every", "validate_every",
                 "validation_batches", "report_max_new_tokens", "learning_rate", "lora_learning_rate", "max_grad_norm"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            argument_parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if not 0 <= args.seed < 2**32:
        argument_parser.error("--seed must be in [0, 2**32)")
    if not math.isfinite(args.max_seconds) or args.max_seconds < 0:
        argument_parser.error("--max-seconds must be nonnegative and finite")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
