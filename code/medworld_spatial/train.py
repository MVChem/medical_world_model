"""Matched two-task spatial pilots, with fixed steps, deadline and real exports."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from skimage.metrics import structural_similarity

from medworld.datasets.current import _sha256
from medworld.decoders import spatial_loss
from medworld.gpu import acquire_gpu
from medworld.runtime import atomic_json, seed_all
from .geometry import feature_loss, semantic_loss
from .model import SparseSpatialModel, reader_from_state


class Cache:
    def __init__(self, root, workers=4):
        self.root = Path(root)
        self.rows, self.protocols = {}, {}
        self.pool = ThreadPoolExecutor(max_workers=workers)
        for task in ("segmentation", "sr"):
            directory = self.root / task
            completed = json.loads((directory / "complete.json").read_text())
            for name, expected in completed["sha256"].items():
                if _sha256(directory / name) != expected:
                    raise ValueError(f"Cache metadata changed: {task}/{name}")
            records = json.loads((directory / "records.json").read_text())
            self.protocols[task] = json.loads((directory / "protocol.json").read_text())
            for split in {r["split"] for r in records}:
                self.rows[task, split] = [r for r in records if r["split"] == split]
        if self.protocols["segmentation"]["data_fingerprint"] != self.protocols["sr"]["data_fingerprint"]:
            raise ValueError("Task caches do not share the patient protocol")
        for name in ("reader_init.pt", "text_embeddings.pt"):
            # Save format ZIP metadata can differ; compare the actual tensors below.
            left = torch.load(self.root / "segmentation" / name, weights_only=True)
            right = torch.load(self.root / "sr" / name, weights_only=True)
            if name == "text_embeddings.pt":
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            else:
                for key in left["state_dict"]:
                    torch.testing.assert_close(left["state_dict"][key], right["state_dict"][key], rtol=0, atol=0)
        self.reader = torch.load(self.root / "segmentation/reader_init.pt", weights_only=True)
        self.text = torch.load(self.root / "segmentation/text_embeddings.pt", weights_only=True)
        self.permutations = {}

    def batch(self, task, split, indices, device):
        records = [self.rows[task, split][i] for i in indices]
        samples = list(self.pool.map(lambda r: torch.load(self.root / task / r["path"], weights_only=True), records))
        grid = tuple(samples[0]["grid"])
        if any(tuple(row["grid"]) != grid for row in samples):
            raise ValueError("Cannot mix native feature grids")
        result = {key: torch.stack([row[key] for row in samples]).to(device)
                  for key, value in samples[0].items() if torch.is_tensor(value)}
        result.update(grid=grid, ids=[r["id"] for r in records], patients=[r["patient"] for r in records])
        return result

    def training_batch(self, task, offset, batch_size, seed, device):
        size = len(self.rows[task, "train"])
        indices = []
        for position in range(offset, offset + batch_size):
            epoch, within = divmod(position, size)
            key = (task, epoch, seed)
            if key not in self.permutations:
                entropy = int.from_bytes(hashlib.sha256(f"{seed}:{task}:{epoch}".encode()).digest()[:8], "little")
                self.permutations[key] = np.random.default_rng(entropy).permutation(size)
                self.permutations = {k: v for k, v in self.permutations.items() if k[0] != task or k == key}
            indices.append(int(self.permutations[key][within]))
        return self.batch(task, "train", indices, device)


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
    tensors = {key: value.detach().float().cpu().numpy() for key, value in result.items() if torch.is_tensor(value)}
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


@torch.no_grad()
def evaluate(model, cache, out, device, split, batch_size, export=False):
    model.eval()
    metrics = {}
    out.mkdir(parents=True, exist_ok=True)
    for task in ("segmentation", "sr"):
        records = cache.rows.get((task, split), [])
        if not records:
            continue
        values, losses = [], []
        for start in range(0, len(records), batch_size):
            batch = cache.batch(task, split, range(start, min(start + batch_size, len(records))), device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                result = model(batch, task)
            loss = float(spatial_loss(task, result["prediction"], batch["targets"], batch["valid"]))
            losses += [loss] * len(batch["ids"])
            measurements = scores(result["prediction"], batch["targets"], batch["valid"], task)
            for identity, patient, measured in zip(batch["ids"], batch["patients"], measurements):
                values.append({"id": identity, "patient": patient, **measured})
            if export and start == 0:
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                    ablated = model(batch, task, replacement=torch.zeros_like(result["slots"]))
                save_export(out / f"{task}_attention.npz", result, batch, task, ablated)
        scalar = [key for key, value in values[0].items() if isinstance(value, (float, int))]
        metrics[task] = {"n": len(values), "loss": float(np.mean(losses)),
                         **{key: float(np.mean([r[key] for r in values])) for key in scalar}}
        with (out / f"{task}.jsonl").open("w") as handle:
            for row in values:
                handle.write(json.dumps(row) + "\n")
    atomic_json(out / "summary.json", metrics)
    return metrics


def checkpoint(path, model, optimizer, step, args):
    temporary = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                "arguments": vars(args), "variant": args.variant, "seed": args.seed}, temporary)
    os.replace(temporary, path)


def main(args):
    os.umask(0o077)
    lock, device = acquire_gpu(args.gpu)
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    cache = Cache(args.cache, args.threads)
    model = SparseSpatialModel(reader_from_state(cache.reader), cache.text, args.variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise ValueError("Refusing to overwrite completed training")
    atomic_json(out / "protocol.json", {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "cache_protocols": cache.protocols, "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "targets": "original dense task targets; extra alignment uses frozen teacher features/crop scores only",
        "evaluation": "fixed final update, no test-set checkpoint or ROI selection"})
    start, first_step = time.time(), 0
    if args.resume and (out / "last.pt").exists():
        saved = torch.load(out / "last.pt", weights_only=False, map_location=device)
        for name in ("variant", "seed", "steps", "batch_size", "learning_rate", "feature_weight", "semantic_weight", "sr_alignment_scale"):
            if saved["arguments"][name] != getattr(args, name):
                raise ValueError(f"Resume configuration changed: {name}")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        first_step = saved["step"]
    gradient_audit = {}
    completed = first_step
    with (out / "metrics.jsonl").open("a", buffering=1) as log:
        for step in range(first_step, args.steps):
            if args.deadline and time.time() >= args.deadline - args.finish_reserve:
                break
            task = ("segmentation", "sr")[step % 2]
            offset = (step // 2) * args.batch_size
            batch = cache.training_batch(task, offset, args.batch_size, args.seed, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                result = model(batch, task)
                task_loss = spatial_loss(task, result["prediction"], batch["targets"], batch["valid"])
                consistency = feature_loss(result["feature"], batch["teacher_features"], batch["thetas"], batch["valid"]) if args.variant in ("featup", "featup_semantic", "image_only_featup") else task_loss * 0
                semantic, eligible = semantic_loss(result["semantic_attention"], batch["semantic_probabilities"], batch["valid"]) if args.variant in ("featup_semantic", "image_only_featup") else (task_loss * 0, task_loss.detach() * 0)
                alignment_scale = args.sr_alignment_scale if task == "sr" else 1.0
                loss = task_loss + alignment_scale * (args.feature_weight * consistency + args.semantic_weight * semantic)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at {step}")
            if step < 2:
                task_query_grad = torch.autograd.grad(task_loss, model.reader.queries, retain_graph=True)[0]
            loss.backward()
            if step < 2:
                gradient_audit[task] = {name: float(parameter.grad.float().norm()) for name, parameter in model.named_parameters()
                    if parameter.grad is not None and ("queries" in name or "gate" in name or "semantic.query" in name)}
                gradient_audit[task]["task_loss_only.reader.queries"] = float(task_query_grad.norm())
                atomic_json(out / "gradient_audit.json", gradient_audit)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            completed = step + 1
            row = {"step": completed, "task": task, "loss": float(loss.detach()), "task_loss": float(task_loss.detach()),
                   "feature_loss": float(consistency.detach()), "semantic_loss": float(semantic.detach()),
                   "semantic_eligible_fraction": float(eligible.detach()), "alignment_scale": alignment_scale,
                   "elapsed_seconds": time.time() - start}
            log.write(json.dumps(row) + "\n")
            if completed % 50 == 0 or completed == 1:
                atomic_json(out / "status.json", {"state": "training", **row, "heartbeat": time.time()})
                print(args.variant, args.seed, row, flush=True)
            if completed % args.save_every == 0:
                checkpoint(out / "last.pt", model, optimizer, completed, args)
            if completed % args.validate_every == 0:
                evaluate(model, cache, out / f"validate_{completed:06d}", device, "validate", args.batch_size)
    checkpoint(out / "last.pt", model, optimizer, completed, args)
    summary = {"variant": args.variant, "seed": args.seed, "steps": completed,
               "requested_steps": args.steps, "complete_budget": completed == args.steps, "evaluations": {}}
    atomic_json(out / "status.json", {"state": "evaluating", "step": completed, "heartbeat": time.time()})
    for split in ("validate", "test", "human_test"):
        summary["evaluations"][split] = evaluate(model, cache, out / split, device, split, args.batch_size, export=True)
    # Export-mode parity and checkpoint reconstruction are tested on real data.
    batch = cache.batch("segmentation", "validate", [0], device)
    model.eval()
    with torch.no_grad():
        reference = model(batch, "segmentation")["prediction"]
        restored = SparseSpatialModel(reader_from_state(cache.reader), cache.text, args.variant).to(device).eval()
        restored.load_state_dict(torch.load(out / "last.pt", weights_only=False, map_location=device)["model"])
        recovered = restored(batch, "segmentation")["prediction"]
        torch.testing.assert_close(reference, recovered, rtol=0, atol=0)
    summary.update(elapsed_seconds=time.time() - start, checkpoint_reload_max_error=float((reference - recovered).abs().max()),
                   completed_unix=time.time())
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "status.json", {"state": "complete", "step": completed, "heartbeat": time.time()})
    if lock:
        lock.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--variant", choices=("image_only", "visual_slots", "slots", "featup", "featup_semantic", "image_only_featup"), required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--feature-weight", type=float, default=.1)
    p.add_argument("--semantic-weight", type=float, default=.05)
    p.add_argument("--sr-alignment-scale", type=float, default=.001,
                   help="Scale extra losses to SR MSE units; do not let feature loss swamp pixel MSE")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--validate-every", type=int, default=2000)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--deadline", type=float, default=0)
    p.add_argument("--finish-reserve", type=float, default=900)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.save_every, args.validate_every, args.threads) <= 0:
        p.error("Counts must be positive")
    main(args)
