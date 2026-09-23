"""Synchronous multi-GPU training with patient-safe sampling and wall-clock budgets."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import os
import random
from pathlib import Path
import signal
import time
import traceback

import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .batching import BatchPrefetch, training_finished
from . import WEIGHTS_ONLY_RESUME_ERROR
from .config import load_config
from .architecture import require_reviewed_data, uses_temporal
from .datasets import UnifiedData
from .evaluation.protocol import training_tasks
from .model import MedWorld
from .runtime import (atomic_json, optimizer_for,
                      save_checkpoint, seed_all, source_fingerprint)


def rank_rng_state(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": [torch.cuda.get_rng_state(device)], "local_cuda_only": True}


def restore_rank_rng(state, device):
    if not state.get("local_cuda_only") or len(state["cuda"]) != 1:
        raise ValueError("Expected one CUDA RNG state for this rank")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"][0], device)


class DistributedTrainer:
    def __init__(self, model, data, out, weights_fingerprint, rank, world_size):
        self.model, self.data, self.out = model, data, Path(out)
        self.cfg, self.weights_fingerprint = model.cfg, weights_fingerprint
        self.rank, self.world_size, self.device = rank, world_size, model.device
        self.progress = {"step": 0, "complete": False,
                         "offsets": {task: 0 for task in (*training_tasks(self.cfg), "temporal")},
                         "world_size": world_size, "started_unix": None, "deadline_unix": None}
        self.optimizer = optimizer_for(model, self.cfg)
        self.ddp = None
        self.stop = False

    def request_stop(self, *_args):
        self.stop = True

    def wrap(self):
        self.ddp = DistributedDataParallel(self.model, device_ids=[self.device.index],
                                          output_device=self.device.index, broadcast_buffers=False,
                                          find_unused_parameters=True, gradient_as_bucket_view=True)

    def initialize_clock(self):
        if self.progress["started_unix"] is not None:
            return
        value = [time.time() if self.rank == 0 else None]
        dist.broadcast_object_list(value, src=0)
        start = value[0]
        self.progress.update(started_unix=start, deadline_unix=start + self.cfg.get("total_hours", 0) * 3600)
        if self.rank == 0:
            self.write_status("running")

    def write_status(self, state, **extra):
        if self.rank == 0:
            atomic_json(self.out / "status.json", {**self.progress, "state": state,
                        "heartbeat_unix": time.time(), "stopped": self.stop, **extra})

    def resume(self, saved):
        raise ValueError(WEIGHTS_ONLY_RESUME_ERROR)

    def parameter_spread(self):
        # Two moments per trainable tensor detect divergence without copying the
        # complete model between ranks. Shared tensors are enumerated once.
        moments = torch.stack([value for p in self.model.parameters() if p.requires_grad
                               for value in (p.detach().double().sum(), p.detach().double().square().sum())])
        maximum, minimum = moments.clone(), moments.clone()
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        spread = float((maximum - minimum).abs().max())
        if spread > 1e-7:
            raise RuntimeError(f"DDP replicas diverged: parameter moment spread {spread}")
        return spread

    def save(self, name="last.pt"):
        self.parameter_spread()
        if self.rank == 0:
            save_checkpoint(self.out / name, self.model, self.progress,
                            self.data.fingerprint, self.weights_fingerprint)
        dist.barrier()

    @torch.no_grad()
    def validate(self):
        saved_rng, training = rank_rng_state(self.device), self.model.training
        self.model.eval()
        metrics = {}
        try:
            for task in training_tasks(self.cfg):
                count = min(self.cfg["validation_samples"], len(self.data.rows(task, "validate")))
                values = torch.zeros(2, device=self.device, dtype=torch.float64)
                for index in range(self.rank, count, self.world_size):
                    loss, _ = self.model(task, self.data.batch(task, "validate", [index]))
                    values[0] += loss.detach().double()
                    values[1] += 1
                dist.all_reduce(values)
                if values[1] > 0:
                    metrics[task + "_loss"] = float(values[0] / values[1])
            if self.rank == 0:
                record = {"step": self.progress["step"], **metrics}
                atomic_json(self.out / "validation.json", record)
                with (self.out / "validation.jsonl").open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps({"validation": record}), flush=True)
            return metrics
        finally:
            restore_rank_rng(saved_rng, self.device)
            self.model.train(training)

    def run(self):
        if self.progress["complete"]:
            return True
        self.model.train()
        prepare = self.model.prepare_batch if self.cfg.get("prefetch_preprocessing") else None
        stream = BatchPrefetch(self.data, self.cfg, self.progress, self.rank, self.world_size, prepare=prepare)
        try:
            while True:
                flags = torch.tensor([int(self.stop), int(training_finished(self.progress, self.cfg, time.time()))],
                                     device=self.device, dtype=torch.int32)
                dist.all_reduce(flags, op=dist.ReduceOp.MAX)
                stop, finished = flags.tolist()
                if stop or finished:
                    self.stop = bool(stop)
                    self.progress["complete"] = bool(finished)
                    break
                torch.cuda.reset_peak_memory_stats(self.device)
                profiler = None
                if self.progress["step"] + 1 == int(os.environ.get("MEDWORLD_PROFILE_STEP", "0")):
                    import cProfile
                    profiler = cProfile.Profile()
                    profiler.enable()
                start = time.monotonic()
                task, batches, marker = stream.next()
                data_seconds = time.monotonic() - start
                self.optimizer.zero_grad(set_to_none=True)
                parts_sum = {}
                for micro, (batch, temporal_batch) in enumerate(batches):
                    with self.ddp.no_sync() if micro + 1 < len(batches) else nullcontext():
                        loss, parts = self.ddp(task, batch, temporal_batch)
                        finite = torch.isfinite(loss).to(torch.int32)
                        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                        if not finite:
                            raise FloatingPointError(f"Nonfinite {task} loss on a rank")
                        (loss / len(batches)).backward()
                    for key, value in parts.items():
                        parts_sum[key] = parts_sum.get(key, 0.) + value.detach().float() / len(batches)
                    del loss, parts, batch, temporal_batch
                norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], self.cfg["max_grad_norm"],
                    error_if_nonfinite=True)
                self.optimizer.step()
                if uses_temporal(self.cfg):
                    self.model.target.update(self.model.encoder, self.cfg["ema_momentum"])
                self.progress.update(marker)
                self.progress["step"] += 1
                names = sorted(parts_sum)
                values = torch.stack([parts_sum[name] for name in names])
                dist.all_reduce(values)
                values /= self.world_size
                torch.cuda.synchronize(self.device)
                seconds = time.monotonic() - start
                if profiler is not None:
                    profiler.disable()
                    profiler.dump_stats(str(self.out / f"rank{self.rank}_step{self.progress['step']}.prof"))
                rank_stats = torch.tensor([seconds, data_seconds, torch.cuda.max_memory_allocated(self.device) / 1024**3,
                                           torch.cuda.max_memory_reserved(self.device) / 1024**3], device=self.device)
                statistics = [torch.empty_like(rank_stats) for _ in range(self.world_size)]
                dist.all_gather(statistics, rank_stats)
                sizes = self.cfg.get("task_batch_sizes", {})
                global_batch = sizes.get(task, self.cfg["batch_size"]) * len(batches) * self.world_size
                record = {"step": self.progress["step"], "task": task,
                          **dict(zip(names, values.cpu().tolist())), "grad_norm": float(norm),
                          "seconds": max(float(x[0]) for x in statistics), "current_global_batch": global_batch,
                          "temporal_global_batch": (sizes.get("temporal", self.cfg["batch_size"]) * len(batches) * self.world_size
                                                    if uses_temporal(self.cfg) else 0),
                          "ranks": [dict(zip(("seconds", "data_wait_seconds", "peak_allocated_gib", "peak_reserved_gib"),
                                             x.cpu().tolist())) for x in statistics],
                          "ema_updates": int(self.model.target.updates) if uses_temporal(self.cfg) else 0,
                          "wall_unix": time.time()}
                if self.rank == 0:
                    with (self.out / "metrics.jsonl").open("a") as handle:
                        handle.write(json.dumps(record) + "\n")
                    print(json.dumps(record), flush=True)
                    self.write_status("running", last_update=record)
                del batches, parts_sum, values
                if self.progress["step"] % self.cfg["validate_every"] == 0:
                    metrics = self.validate()
                    tasks = training_tasks(self.cfg)
                    score = sum(metrics[t + "_loss"] for t in tasks) / len(tasks)
                    if score < self.progress.get("best_validation", float("inf")):
                        self.progress["best_validation"] = score
                        self.save("best.pt")
                if self.progress["step"] % self.cfg["save_every"] == 0:
                    self.save()
        finally:
            stream.close()
        # At a wall-clock boundary, retain the most recent periodic validation;
        # save promptly instead of extending the user's requested run budget.
        if not self.cfg.get("total_hours", 0) and self.progress["complete"]:
            self.validate()
        self.save()
        if self.progress["complete"]:
            self.save("final.pt")
        self.write_status("stopped" if self.stop else "complete")
        return self.progress["complete"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", help="Unavailable: checkpoints contain trained weights only")
    args = parser.parse_args()
    if args.resume:
        parser.error(WEIGHTS_ONLY_RESUME_ERROR)
    rank, local_rank, world_size = (int(os.environ[key]) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if world_size < 2:
        parser.error("Launch with torchrun on at least two GPUs")
    allowed = sorted(os.sched_getaffinity(0))
    threads = int(os.environ.get("OMP_NUM_THREADS", "8"))
    # Keep CPU cores available to image workers independently of the
    # intra-op thread count; small tensor operations suffer from oversubscription.
    cores = int(os.environ.get("MEDWORLD_CPU_CORES_PER_RANK", "8"))
    selection = allowed[local_rank * cores:(local_rank + 1) * cores]
    if selection:
        os.sched_setaffinity(0, selection)
    torch.set_num_threads(threads)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=15))
    out = Path(args.out).resolve()
    if not (out / "launch.json").exists():
        raise ValueError("Use the GPU-locking medworld.launch_distributed entry")
    (out / f"rank{rank}.pid").write_text(str(os.getpid()) + "\n")
    try:
        cfg = load_config(args.config)
        seed_all(cfg["seed"])
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        data = UnifiedData(cfg)
        require_reviewed_data(cfg, data)
        fingerprints = [None] * world_size
        dist.all_gather_object(fingerprints, data.fingerprint)
        if len(set(fingerprints)) != 1:
            raise ValueError("Ranks loaded different datasets")
        weights = [source_fingerprint(cfg) if rank == 0 else None]
        dist.broadcast_object_list(weights, src=0)
        model = MedWorld(cfg, device)
        model.pos_weight.copy_(data.current.pos_weight.to(device))
        trainer = DistributedTrainer(model, data, out, weights[0], rank, world_size)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
            signal.signal(sig, trainer.request_stop)
        trainer.wrap()
        seed_all(cfg["seed"] + rank)
        if rank == 0:
            atomic_json(out / "config.json", cfg)
            atomic_json(out / "data_protocol.json", data.metadata)
            atomic_json(out / "model.json", {**model.metadata, "world_size": world_size,
                        "precision": "BF16 autocast with FP32 trainable parameters" if cfg.get("amp") else "BF16 frozen base, FP32 adapters"})
        dist.barrier()
        trainer.initialize_clock()
        trainer.run()
        trainer.write_status("stopped" if trainer.stop else "complete", replica_parameter_spread=trainer.parameter_spread())
        dist.barrier()
    except Exception as error:
        # A failed rank cannot join collective teardown while peers are still
        # waiting in a training collective. Exit so torchrun can stop the group.
        traceback.print_exc()
        atomic_json(out / f"rank{rank}_error.json", {
            "rank": rank, "error": repr(error), "wall_unix": time.time(),
            "traceback": traceback.format_exc(),
        })
        import sys
        sys.stderr.flush()
        os._exit(1)
    else:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
