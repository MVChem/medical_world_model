"""Optimizer-boundary checkpoints, reproducible streams and loss validation."""
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from . import FORMAT_VERSION
from .datasets import TASKS
from .datasets.current import _sha256


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def source_fingerprint(cfg):
    root = Path(cfg["qwen"])
    paths = sorted(p for p in root.iterdir() if p.is_file() and
                   (p.suffix in (".json", ".safetensors", ".model") or p.name == "merges.txt"))
    if not paths or not any(p.suffix == ".safetensors" for p in paths):
        raise ValueError("Expected local Qwen weights and configuration")
    hashes = {"qwen/" + p.name: _sha256(p) for p in paths}
    hashes["jepa"] = _sha256(Path(cfg["jepa"]))
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def optimizer_for(model, cfg):
    lora, other = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (lora if "lora_" in name else other).append(parameter)
    return torch.optim.AdamW([{"params": lora, "lr": cfg["lora_learning_rate"]},
                             {"params": other, "lr": cfg["learning_rate"]}], weight_decay=.01)


def save_checkpoint(path, model, optimizer, progress, data_fingerprint, weights_fingerprint, rng_payload=None):
    path = Path(path)
    state = {"format_version": FORMAT_VERSION, "config": model.cfg, "metadata": model.metadata,
             "model": model.compact_state(), "optimizer": optimizer.state_dict(),
             "progress": dict(progress), "rng": rng_state() if rng_payload is None else rng_payload,
             "data_fingerprint": data_fingerprint, "weights_fingerprint": weights_fingerprint}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def read_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    required = {"format_version", "config", "metadata", "model", "optimizer", "progress", "rng",
                "data_fingerprint", "weights_fingerprint"}
    if set(state) != required or state["format_version"] != FORMAT_VERSION:
        raise ValueError("Not a supported unified MedWorld checkpoint")
    return state


def load_model(path, device="cuda"):
    from .model import MedWorld
    state = read_checkpoint(path)
    if source_fingerprint(state["config"]) != state["weights_fingerprint"]:
        raise ValueError("Pretrained weights/configuration changed since this checkpoint")
    seed_all(state["config"]["seed"])
    model = MedWorld(state["config"], device)
    model.restore(state["model"])
    return model.eval(), state


@torch.no_grad()
def validate(model, data, samples):
    rng, training = rng_state(), model.training
    model.eval()
    metrics = {}
    try:
        for task in (*TASKS, *(("temporal",) if model.target is not None else ())):
            count = min(samples, len(data.rows(task, "validate")))
            if not count:
                continue
            values = []
            for index in range(count):
                batch = data.batch(task, "validate", [index])
                loss, _ = (model.temporal_loss(batch) if task == "temporal" else model.current_loss(task, batch))
                values.append(float(loss))
            metrics[task + "_loss"] = float(np.mean(values))
        return metrics
    finally:
        restore_rng(rng)
        model.train(training)


class Trainer:
    def __init__(self, model, data, out, weights_fingerprint):
        self.model, self.data, self.out = model, data, Path(out)
        self.cfg, self.weights_fingerprint = model.cfg, weights_fingerprint
        self.progress = {"stage": "stage1", "step": 0, "stage_complete": False,
                         "offsets": {task: 0 for task in (*TASKS, "temporal")}, "replay_index": 0}
        self.optimizer = optimizer_for(model, self.cfg)
        self.stop = False

    def request_stop(self, *_args):
        self.stop = True

    def install_signals(self):
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)

    def resume(self, saved):
        if saved["progress"].get("world_size", 1) != 1:
            raise ValueError("Resume a distributed checkpoint with medworld.distributed_train")
        if saved["config"] != self.cfg or saved["data_fingerprint"] != self.data.fingerprint:
            raise ValueError("Resume requires identical configuration and filtered data")
        if saved["weights_fingerprint"] != self.weights_fingerprint:
            raise ValueError("Pretrained weights changed")
        self.model.restore(saved["model"])
        self.progress = saved["progress"]
        self.optimizer.load_state_dict(saved["optimizer"])
        if self.progress["stage"] == "stage2":
            if self.model.target is None or int(self.model.target.updates) != self.progress["step"]:
                raise ValueError("EMA update count differs from completed optimizer steps")
        restore_rng(saved["rng"])

    def enter_stage2(self):
        if self.progress["stage"] != "stage1" or not self.progress["stage_complete"]:
            raise ValueError("Stage 2 must start from a completed Stage 1 checkpoint")
        self.model.begin_stage2()
        self.progress.update(stage="stage2", step=0, stage_complete=False, replay_index=0)
        self.optimizer = optimizer_for(self.model, self.cfg)

    def save(self, name="last.pt"):
        save_checkpoint(self.out / name, self.model, self.optimizer, self.progress,
                        self.data.fingerprint, self.weights_fingerprint)

    def batch(self, task):
        offset = self.progress["offsets"][task]
        batch = self.data.training_batch(task, offset, self.cfg["batch_size"], self.cfg["seed"])
        self.progress["offsets"][task] += self.cfg["batch_size"]
        return batch

    def run_stage(self):
        if self.progress["stage_complete"]:
            return True
        stage = self.progress["stage"]
        budget = self.cfg[stage + "_steps"]
        accumulation = self.cfg[stage + "_accumulation"]
        self.model.train()
        while self.progress["step"] < budget and not self.stop:
            device = next(self.model.parameters()).device
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.monotonic()
            step = self.progress["step"]
            task = TASKS[step % len(TASKS)] if stage == "stage1" else "temporal"
            self.optimizer.zero_grad(set_to_none=True)
            values = {}
            for _ in range(accumulation):
                batch = self.batch(task)
                loss, parts = (self.model.temporal_loss(batch) if task == "temporal"
                               else self.model.current_loss(task, batch))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite {task} loss")
                (loss / accumulation).backward()
                for key, value in parts.items():
                    values[key] = values.get(key, 0.) + float(value) / accumulation
            replay_every = self.cfg["replay_every"]
            if stage == "stage2" and replay_every and (step + 1) % replay_every == 0:
                replay_task = TASKS[self.progress["replay_index"] % len(TASKS)]
                self.progress["replay_index"] += 1
                for _ in range(accumulation):
                    loss, _ = self.model.current_loss(replay_task, self.batch(replay_task))
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite replay loss")
                    (self.cfg["replay_weight"] * loss / accumulation).backward()
                    key = "replay_" + replay_task
                    values[key] = values.get(key, 0.) + float(loss.detach()) / accumulation
            norm = torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],
                                                  self.cfg["max_grad_norm"], error_if_nonfinite=True)
            self.optimizer.step()
            # Exactly once per optimizer update, including updates with replay.
            if stage == "stage2":
                self.model.target.update(self.model.encoder, self.cfg["ema_momentum"])
            self.progress["step"] = step + 1
            self.progress["stage_complete"] = step + 1 == budget
            record = {"stage": stage, "step": step + 1, "task": task, **values,
                      "grad_norm": float(norm), "seconds": time.monotonic() - start,
                      "ema_updates": 0 if self.model.target is None else int(self.model.target.updates)}
            if device.type == "cuda":
                record.update(cuda_step_peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3,
                              cuda_step_peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024**3)
            print(json.dumps(record), flush=True)
            with (self.out / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if (step + 1) % self.cfg["validate_every"] == 0 or self.progress["stage_complete"]:
                metrics = validate(self.model, self.data, self.cfg["validation_samples"])
                atomic_json(self.out / f"validation_{stage}.json", {"step": step + 1, **metrics})
                print(json.dumps({"validation": stage, **metrics}), flush=True)
            if (step + 1) % self.cfg["save_every"] == 0 or self.stop or self.progress["stage_complete"]:
                self.save()
        self.save(stage + ".pt" if self.progress["stage_complete"] else "last.pt")
        return self.progress["stage_complete"]
