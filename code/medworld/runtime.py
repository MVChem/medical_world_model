"""Trained-weight checkpoints, reproducible streams and loss validation."""
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from .batching import BatchPrefetch
from . import FORMAT_VERSION, WEIGHTS_ONLY_RESUME_ERROR
from .datasets.protocol import _sha256
from .architecture import RAW_INPUT, uses_slot_branch, uses_temporal
from .evaluation.protocol import training_tasks


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
    if not uses_slot_branch(cfg):
        return hashlib.sha256(b"medworld/raw_input_v1/no-pretrained-assets").hexdigest()
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


def save_checkpoint(path, model, progress, data_fingerprint, weights_fingerprint):
    """Save learned online/EMA weights and provenance, without training-resume state."""
    path = Path(path)
    summary = {"step": progress["step"], "complete": progress["complete"],
               "world_size": progress.get("world_size", 1),
               "task_samples": {task: progress["offsets"][task] for task in training_tasks(model.cfg)}}
    state = {"format_version": FORMAT_VERSION, "config": model.cfg, "metadata": model.metadata,
             "model": model.compact_state(), "progress": summary,
             "data_fingerprint": data_fingerprint, "weights_fingerprint": weights_fingerprint}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def read_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    required = {"format_version", "config", "metadata", "model", "progress",
                "data_fingerprint", "weights_fingerprint"}
    if not isinstance(state, dict) or set(state) != required or state["format_version"] != FORMAT_VERSION:
        raise ValueError("Not a supported MedWorld trained-weights checkpoint")
    if not isinstance(state["config"], dict) or state["config"].get("architecture") != RAW_INPUT:
        raise ValueError("Checkpoint must explicitly declare the supported raw-input architecture")
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
        for task in training_tasks(model.cfg):
            count = min(samples, len(data.rows(task, "validate")))
            if not count:
                continue
            values = []
            for index in range(count):
                batch = data.batch(task, "validate", [index])
                loss, _ = model(task, batch)
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
        self.progress = {"step": 0, "complete": False,
                         "offsets": {task: 0 for task in (*training_tasks(self.cfg), "temporal")}}
        self.optimizer = optimizer_for(model, self.cfg)
        self.stop = False

    def request_stop(self, *_args):
        self.stop = True

    def install_signals(self):
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)

    def resume(self, saved):
        raise ValueError(WEIGHTS_ONLY_RESUME_ERROR)

    def save(self, name="last.pt"):
        save_checkpoint(self.out / name, self.model, self.progress,
                        self.data.fingerprint, self.weights_fingerprint)

    def run(self):
        if self.progress["complete"]:
            return True
        self.model.train()
        stream = BatchPrefetch(self.data, self.cfg, self.progress, rank=0, world_size=1)
        try:
            while self.progress["step"] < self.cfg["steps"] and not self.stop:
                device = next(self.model.parameters()).device
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                start = time.monotonic()
                task, batches, marker = stream.next()
                self.optimizer.zero_grad(set_to_none=True)
                values = {}
                for batch, temporal_batch in batches:
                    loss, parts = self.model(task, batch, temporal_batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite joint loss")
                    (loss / len(batches)).backward()
                    for key, value in parts.items():
                        values[key] = values.get(key, 0.) + float(value) / len(batches)
                    del loss, parts, batch, temporal_batch
                norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.cfg["max_grad_norm"], error_if_nonfinite=True)
                self.optimizer.step()
                if uses_temporal(self.cfg):
                    self.model.target.update(self.model.encoder, self.cfg["ema_momentum"])
                self.progress.update(marker)
                self.progress["step"] += 1
                self.progress["complete"] = self.progress["step"] == self.cfg["steps"]
                record = {"step": self.progress["step"], "task": task, **values,
                          "grad_norm": float(norm), "seconds": time.monotonic() - start,
                          "ema_updates": int(self.model.target.updates) if uses_temporal(self.cfg) else 0,
                          "current_global_batch": self.cfg.get("task_batch_sizes", {}).get(task, self.cfg["batch_size"]) * len(batches),
                          "temporal_global_batch": (self.cfg.get("task_batch_sizes", {}).get("temporal", self.cfg["batch_size"]) * len(batches)
                                                    if uses_temporal(self.cfg) else 0)}
                if device.type == "cuda":
                    record.update(cuda_step_peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3,
                                  cuda_step_peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024**3)
                print(json.dumps(record), flush=True)
                with (self.out / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                del batches
                if self.progress["step"] % self.cfg["validate_every"] == 0 or self.progress["complete"]:
                    metrics = validate(self.model, self.data, self.cfg["validation_samples"])
                    atomic_json(self.out / "validation.json", {"step": self.progress["step"], **metrics})
                if self.progress["step"] % self.cfg["save_every"] == 0 or self.stop or self.progress["complete"]:
                    self.save()
        finally:
            stream.close()
        self.save("final.pt" if self.progress["complete"] else "last.pt")
        return self.progress["complete"]
