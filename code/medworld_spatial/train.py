"""Matched variants trained on the same on-demand batch; no feature cache."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch
from skimage.metrics import structural_similarity

from medworld.decoders import spatial_loss
from medworld.gpu import acquire_gpu
from medworld.runtime import atomic_json, seed_all
from .geometry import feature_loss, semantic_loss
from .model import SparseSpatialModel, reader_from_state
from .online import OnlineInputs

VARIANTS = ("image_only", "visual_slots", "slots", "featup", "featup_semantic", "image_only_featup")
TASKS = ("segmentation", "sr")


def training_position(step, tasks, batch_size):
    """Zero-based update -> task and its own deterministic sample offset."""
    return tasks[step % len(tasks)], (step // len(tasks)) * batch_size


def scores(prediction, target, valid, task):
    prediction, target, valid = prediction.detach().float().cpu(), target.cpu(), valid.cpu()
    if task == "segmentation":
        p = (prediction[:, :target.shape[1]].sigmoid() >= .5).float() * valid
        t = (target >= .5).float() * valid
        dice = (2 * (p * t).sum((2, 3)) + 1e-6) / (p.sum((2, 3)) + t.sum((2, 3)) + 1e-6)
        return [{"dice": float(row.mean()), "dice_per_organ": row.tolist()} for row in dice]
    prediction = prediction.clamp(0, 1)
    output = []
    for p, t, m in zip(prediction, target, valid):
        mse = float(((p - t).square() * m).sum() / m.sum())
        yy, xx = torch.where(m[0] > .5)
        region = (slice(int(yy.min()), int(yy.max()) + 1), slice(int(xx.min()), int(xx.max()) + 1))
        output.append({"psnr": float(-10 * np.log10(max(mse, 1e-12))),
            "ssim": float(structural_similarity(t[0].numpy()[region], p[0].numpy()[region], data_range=1.0))})
    return output


def save_export(path, result, batch, task, ablated=None):
    tensors = {key: value.detach().float().cpu().numpy() for key, value in result.items() if torch.is_tensor(value) and key != "feature"}
    tensors.update(input=batch["pixels"].cpu().numpy(), reference=batch["targets"].cpu().numpy(),
                   valid=batch["valid"].cpu().numpy(), teacher_semantic=batch["semantic_probabilities"].cpu().numpy())
    if ablated is not None:
        tensors["replacement_prediction"] = ablated["prediction"].float().cpu().numpy()
    np.savez_compressed(path, **tensors)
    atomic_json(path.with_suffix(".json"), {"ids": batch["ids"], "patients": batch["patients"], "task": task,
        "encoder_attention": "actual visual-slot softmax over native pre-merger image patches, raster reordered",
        "decoder_attention": "actual multihead pixel-query to state-slot attention, averaged over heads",
        "semantic_attention": "actual text-derived query to feature-field attention used in prediction",
        "teacher_semantic": "native VLM crop probabilities, NOT attention or lesion masks",
        "replacement": "same trained model, all state slots set to zero; sensitivity diagnostic, not matched retraining"})


def update(model, optimizer, batch, task, args, audit=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=batch["raw"].device.type, dtype=torch.bfloat16,
                        enabled=batch["raw"].is_cuda):
        result = model(batch, task)
        task_loss = spatial_loss(task, result["prediction"], batch["targets"], batch["valid"])
        consistency = (feature_loss(result["feature"], batch["teacher_features"], batch["thetas"], batch["valid"])
                       if model.variant in ("featup", "featup_semantic", "image_only_featup") else task_loss * 0)
        semantic, eligible = (semantic_loss(result["semantic_attention"], batch["semantic_probabilities"], batch["valid"])
                              if model.variant in ("featup_semantic", "image_only_featup") else (task_loss * 0, task_loss.detach() * 0))
        scale = args.sr_alignment_scale if task == "sr" else 1.0
        loss = task_loss + scale * (args.feature_weight * consistency + args.semantic_weight * semantic)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Nonfinite {model.variant}/{task} loss")
    if audit:
        task_query_grad = torch.autograd.grad(task_loss, model.reader.queries, retain_graph=True)[0]
    loss.backward()
    gradients = {}
    if audit:
        gradients = {name: float(p.grad.float().norm()) for name, p in model.named_parameters()
                     if p.grad is not None and ("queries" in name or "gate" in name or "semantic.query" in name)}
        gradients["task_loss_only.reader.queries"] = float(task_query_grad.norm())
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {"task": task, "loss": float(loss.detach()), "task_loss": float(task_loss.detach()),
            "feature_loss": float(consistency.detach()), "semantic_loss": float(semantic.detach()),
            "semantic_eligible_fraction": float(eligible.detach()), "alignment_scale": scale,
            "gradient_norm": float(norm)}, gradients


def contract(args, inputs):
    keys = ("variants", "tasks", "sr_scale", "seed", "steps", "batch_size", "learning_rate", "feature_weight", "semantic_weight",
            "sr_alignment_scale", "views", "selection_seed", "semantic_batch", "train_n", "val_n", "test_n", "human_n")
    return {"arguments": {key: getattr(args, key) for key in keys}, "input_protocol": inputs.protocol}


def save_group(path, models, optimizers, step, protocol):
    temporary = path.with_suffix(".tmp")
    torch.save({"models": {v: model.state_dict() for v, model in models.items()},
                "optimizers": {v: optimizer.state_dict() for v, optimizer in optimizers.items()},
                "step": step, "contract": protocol}, temporary)
    os.replace(temporary, path)


def restore_group(path, models, optimizers, protocol, device):
    saved = torch.load(path, weights_only=False, map_location=device)
    if saved["contract"] != protocol:
        raise ValueError("Resume requires identical inputs, teachers and experiment parameters")
    for variant, model in models.items():
        model.load_state_dict(saved["models"][variant], strict=True)
        optimizers[variant].load_state_dict(saved["optimizers"][variant])
    return saved["step"]


@torch.no_grad()
def evaluate(models, inputs, directories, split, batch_size, *, export=False, deadline=0, heartbeat=None):
    """One transient input batch shared across all models, also during evaluation."""
    for model in models.values():
        model.eval()
    metrics = {v: {} for v in models}
    complete = True
    for task in inputs.tasks:
        records = inputs.rows.get((task, split), [])
        if not records:
            continue
        values, losses = {v: [] for v in models}, {v: [] for v in models}
        for start in range(0, len(records), batch_size):
            if deadline and time.time() >= deadline:
                complete = False
                break
            batch = inputs.batch(task, split, range(start, min(start + batch_size, len(records))),
                                 alignment=False, semantic=export and start == 0)
            for variant, model in models.items():
                directory = directories[variant]
                directory.mkdir(parents=True, exist_ok=True)
                with torch.autocast(device_type=inputs.device, dtype=torch.bfloat16, enabled=inputs.device == "cuda"):
                    result = model(batch, task)
                loss = float(spatial_loss(task, result["prediction"], batch["targets"], batch["valid"]))
                losses[variant] += [loss] * len(batch["ids"])
                measured = scores(result["prediction"], batch["targets"], batch["valid"], task)
                values[variant].extend({"id": identity, "patient": patient, **row}
                    for identity, patient, row in zip(batch["ids"], batch["patients"], measured))
                if export and start == 0:
                    with torch.autocast(device_type=inputs.device, dtype=torch.bfloat16, enabled=inputs.device == "cuda"):
                        ablated = model(batch, task, replacement=torch.zeros_like(result["slots"]))
                    save_export(directory / f"{task}_attention.npz", result, batch, task, ablated)
                    del ablated
                del result
            del batch
            if heartbeat:
                heartbeat(split, task, min(start + batch_size, len(records)), len(records))
        for variant in models:
            rows = values[variant]
            directory = directories[variant]
            directory.mkdir(parents=True, exist_ok=True)
            metrics[variant][task] = {"n": len(rows), "expected_n": len(records), "complete": len(rows) == len(records)}
            if rows:
                scalar = [key for key, value in rows[0].items() if isinstance(value, (float, int))]
                metrics[variant][task].update(loss=float(np.mean(losses[variant])),
                    **{key: float(np.mean([row[key] for row in rows])) for key in scalar})
            with (directory / f"{task}.jsonl").open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
    for variant, directory in directories.items():
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "summary.json", metrics[variant])
    return metrics, complete


def main(args):
    os.umask(0o077)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise ValueError("Refusing to overwrite a finished seed group")
    stopping = []
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda signum, frame: stopping.append(signum))
    atomic_json(out / "status.json", {"state": "loading", "pid": os.getpid(), "heartbeat": time.time()})
    lock, device = acquire_gpu(args.gpu)
    torch.set_num_threads(args.threads)
    inputs = OnlineInputs(args, device)
    protocol = contract(args, inputs)
    atomic_json(out / "protocol.json", protocol)
    models, optimizers, directories, audits = {}, {}, {}, {}
    for variant in args.variants:
        seed_all(args.seed)
        models[variant] = SparseSpatialModel(reader_from_state(inputs.reader), inputs.text, variant).to(device)
        optimizers[variant] = torch.optim.AdamW(models[variant].parameters(), lr=args.learning_rate, weight_decay=.01)
        directories[variant] = args.jobs_root / f"{variant}_seed{args.seed}"
        directories[variant].mkdir(parents=True, exist_ok=True)
        atomic_json(directories[variant] / "protocol.json", {**protocol, "variant": variant,
                    "checkpoint": str(out / "last.pt"), "source_protocol": inputs.protocol})
        audits[variant] = {}
    first_step = restore_group(out / "last.pt", models, optimizers, protocol, device) if args.resume and (out / "last.pt").exists() else 0
    completed, start = first_step, time.time()
    need_alignment = any(v in ("featup", "featup_semantic", "image_only_featup") for v in models)
    need_semantic = any(v in ("featup_semantic", "image_only_featup") for v in models)
    print(json.dumps({"event": "training_started", "seed": args.seed, "step": completed,
                      "input_mode": "on_demand_no_disk_cache", "counts": inputs.protocol["counts"]}), flush=True)
    for step in range(first_step, args.steps):
        if stopping or (args.deadline and time.time() >= args.deadline - args.finish_reserve):
            break
        task, offset = training_position(step, args.tasks, args.batch_size)
        tick = time.time()
        batch = inputs.training_batch(task, offset, args.batch_size, args.seed,
                                      alignment=need_alignment, semantic=need_semantic)
        teacher_seconds = time.time() - tick
        records = {}
        for variant, model in models.items():
            row, gradients = update(model, optimizers[variant], batch, task, args, audit=step < len(args.tasks))
            if gradients:
                audits[variant][task] = gradients
                atomic_json(directories[variant] / "gradient_audit.json", audits[variant])
            row.update(step=step + 1, teacher_seconds=teacher_seconds, elapsed_seconds=time.time() - start)
            records[variant] = row
            with (directories[variant] / "metrics.jsonl").open("a") as log:
                log.write(json.dumps(row) + "\n")
            atomic_json(directories[variant] / "status.json", {"state": "training", **row, "heartbeat": time.time()})
        del batch
        completed = step + 1
        state = {"state": "training", "step": completed, "requested_steps": args.steps, "seed": args.seed,
                 "teacher_seconds": teacher_seconds, "batch_seconds": time.time() - tick,
                 "heartbeat": time.time(), "variants": list(models), "pid": os.getpid()}
        atomic_json(out / "status.json", state)
        if completed <= 2 or completed % 10 == 0:
            print(json.dumps(state), flush=True)
        if completed % args.save_every == 0 or completed == 2:
            save_group(out / "last.pt", models, optimizers, completed, protocol)
        if completed % args.validate_every == 0 and not stopping:
            evaluate(models, inputs, {v: p / f"validate_{completed:06d}" for v, p in directories.items()},
                     "validate", args.batch_size, deadline=args.deadline - args.finish_reserve if args.deadline else 0)
    save_group(out / "last.pt", models, optimizers, completed, protocol)
    if stopping:
        atomic_json(out / "status.json", {"state": "interrupted", "step": completed, "signals": stopping,
                                          "heartbeat": time.time(), "checkpoint": str(out / "last.pt")})
        print(json.dumps({"event": "interrupted", "step": completed, "signals": stopping}), flush=True)
        return
    summaries = {v: {"variant": v, "seed": args.seed, "tasks": args.tasks,
                    "steps": completed, "requested_steps": args.steps,
                    "complete_budget": completed == args.steps, "evaluations": {}, "evaluation_complete": True,
                    "checkpoint": str(out / "last.pt"), "input_mode": "on_demand_no_disk_cache"} for v in models}
    def heartbeat(split, task, done, total):
        atomic_json(out / "status.json", {"state": "evaluating", "step": completed, "split": split,
            "task": task, "evaluated": done, "total": total, "heartbeat": time.time()})
    for split in ("validate", "test", "human_test"):
        if not any(inputs.rows.get((task, split)) for task in args.tasks):
            continue
        measured, complete = evaluate(models, inputs, {v: p / split for v, p in directories.items()}, split,
                                      args.batch_size, export=True, deadline=args.deadline, heartbeat=heartbeat)
        for variant in models:
            summaries[variant]["evaluations"][split] = measured[variant]
            summaries[variant]["evaluation_complete"] &= complete
    # Exact reconstruction check uses one already selected validation case.
    reload_task = args.tasks[0]
    batch = inputs.batch(reload_task, "validate", [0], alignment=False, semantic=False)
    with torch.no_grad():
        saved = torch.load(out / "last.pt", weights_only=False, map_location="cpu")
        for variant, model in models.items():
            model.eval()
            reference = model(batch, reload_task)["prediction"]
            restored = SparseSpatialModel(reader_from_state(inputs.reader), inputs.text, variant).to(device).eval()
            restored.load_state_dict(saved["models"][variant])
            recovered = restored(batch, reload_task)["prediction"]
            torch.testing.assert_close(reference, recovered, rtol=0, atol=0)
            summaries[variant]["checkpoint_reload_max_error"] = float((reference - recovered).abs().max())
            del restored, reference, recovered
    for variant, summary in summaries.items():
        summary.update(elapsed_seconds=time.time() - start, completed_unix=time.time())
        atomic_json(directories[variant] / "summary.json", summary)
        atomic_json(directories[variant] / "status.json", {"state": "complete" if summary["complete_budget"] and summary["evaluation_complete"] else "budget_exhausted",
                    "step": completed, "heartbeat": time.time()})
    final_state = "complete" if all(s["complete_budget"] and s["evaluation_complete"] for s in summaries.values()) else "budget_exhausted"
    atomic_json(out / "summary.json", {"state": final_state, "step": completed, "seed": args.seed, "variants": list(models)})
    atomic_json(out / "status.json", {"state": final_state, "step": completed, "heartbeat": time.time()})
    if lock:
        lock.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "semantic-teacher", "out", "jobs-root"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    p.add_argument("--sr-scale", type=int, choices=(4, 8), default=4, help="SR enlargement per axis; pixel-count ratio is its square")
    p.add_argument("--gpu", default="auto")
    for name, value in (("steps", 4000), ("batch-size", 8), ("semantic-batch", 8), ("seed", 42),
                        ("selection-seed", 42), ("views", 2), ("train-n", 4096), ("val-n", 128),
                        ("test-n", 447), ("human-n", 138), ("save-every", 100), ("validate-every", 2000), ("threads", 4)):
        p.add_argument("--" + name, type=int, default=value)
    for name, value in (("learning-rate", 1e-4), ("feature-weight", .1), ("semantic-weight", .05),
                        ("sr-alignment-scale", .001), ("deadline", 0), ("finish-reserve", 1200)):
        p.add_argument("--" + name, type=float, default=value)
    p.add_argument("--resume", action="store_true")
    return p


if __name__ == "__main__":
    p = parser()
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.semantic_batch, args.save_every, args.validate_every,
           args.views, args.train_n, args.val_n, args.threads) <= 0 or len(set(args.variants)) != len(args.variants) or len(set(args.tasks)) != len(args.tasks):
        p.error("Counts must be positive; variants and tasks must be distinct")
    try:
        main(args)
    except Exception:
        args.out.mkdir(parents=True, exist_ok=True)
        atomic_json(args.out / "failure.json", {"traceback": traceback.format_exc(), "time": time.time(), "pid": os.getpid()})
        raise
