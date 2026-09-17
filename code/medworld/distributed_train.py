"""Synchronous multi-GPU training with patient-safe sampling and wall-clock budgets."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import timedelta
import gc
import json
import os
import random
from pathlib import Path
import signal
import time

import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .config import load_config
from .datasets import TASKS, UnifiedData
from .model import MedWorld
from .runtime import (atomic_json, optimizer_for, read_checkpoint,
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


def rank_slice(offset, batch_size, rank, world_size):
    if not 0 <= rank < world_size or offset < 0 or batch_size < 1:
        raise ValueError("Invalid distributed stream coordinates")
    return offset + rank * batch_size, offset + world_size * batch_size


def phase_finished(progress, cfg, now):
    if cfg.get("total_hours", 0) > 0:
        if now >= progress["deadline_unix"]:
            return True
        return (progress["stage"] == "stage1" and progress["step"] >= 4
                and progress["step"] % 4 == 0 and now >= progress["stage1_deadline_unix"])
    return progress["step"] >= cfg[progress["stage"] + "_steps"]


class BatchPrefetch:
    """Plan ahead without advancing the committed checkpoint cursor."""
    def __init__(self, data, cfg, progress, rank, world_size):
        self.data, self.cfg, self.rank, self.world_size = data, cfg, rank, world_size
        self.planned = dict(progress["offsets"])
        self.stage, self.step, self.replay = progress["stage"], progress["step"], progress["replay_index"]
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"data-rank{rank}")
        self.queue = deque()
        for _ in range(cfg.get("prefetch_batches", 2)):
            self._submit()

    def _request(self, task):
        size = self.cfg.get("task_batch_sizes", {}).get(task, self.cfg["batch_size"])
        offset, following = rank_slice(self.planned[task], size, self.rank, self.world_size)
        self.planned[task] = following
        return task, offset, size

    def _submit(self):
        task = TASKS[self.step % len(TASKS)] if self.stage == "stage1" else "temporal"
        replay_task = None
        if self.stage == "stage2" and self.cfg["replay_every"] and (self.step + 1) % self.cfg["replay_every"] == 0:
            replay_task = TASKS[self.replay % len(TASKS)]
            self.replay += 1
        requests = [(self._request(task), self._request(replay_task) if replay_task else None)
                    for _ in range(self.cfg[self.stage + "_accumulation"])]
        marker = {"offsets": dict(self.planned), "replay_index": self.replay}
        def load():
            def batch(request):
                return self.data.training_batch(*request, self.cfg["seed"]) if request else None
            return task, replay_task, [(batch(main), batch(replay)) for main, replay in requests], marker
        self.queue.append(self.pool.submit(load))
        self.step += 1

    def next(self):
        result = self.queue.popleft().result()
        self._submit()
        return result

    def close(self):
        for future in self.queue:
            future.cancel()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.queue.clear()


class DistributedTrainer:
    def __init__(self, model, data, out, weights_fingerprint, rank, world_size):
        self.model, self.data, self.out = model, data, Path(out)
        self.cfg, self.weights_fingerprint = model.cfg, weights_fingerprint
        self.rank, self.world_size, self.device = rank, world_size, model.device
        self.progress = {"stage": "stage1", "step": 0, "stage_complete": False,
                         "offsets": {task: 0 for task in (*TASKS, "temporal")}, "replay_index": 0,
                         "world_size": world_size, "started_unix": None, "deadline_unix": None,
                         "stage1_deadline_unix": None}
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
        self.progress.update(started_unix=start, deadline_unix=start + self.cfg.get("total_hours", 0) * 3600,
                             stage1_deadline_unix=start + self.cfg.get("stage1_hours", 0) * 3600)
        if self.rank == 0:
            self.write_status("running")

    def write_status(self, state, **extra):
        if self.rank == 0:
            atomic_json(self.out / "status.json", {**self.progress, "state": state,
                        "heartbeat_unix": time.time(), "stopped": self.stop, **extra})

    def resume(self, saved):
        if saved["config"] != self.cfg or saved["data_fingerprint"] != self.data.fingerprint:
            raise ValueError("Distributed resume requires identical config and data")
        if saved["weights_fingerprint"] != self.weights_fingerprint:
            raise ValueError("Pretrained weights changed")
        if saved["progress"].get("world_size") != self.world_size:
            raise ValueError("Resume must retain the same number of ranks")
        if len(saved["rng"].get("per_rank", [])) != self.world_size:
            raise ValueError("Per-rank RNG states missing")
        self.model.restore(saved["model"])
        self.optimizer.load_state_dict(saved["optimizer"])
        self.progress = dict(saved["progress"])
        if self.progress["stage"] == "stage2" and int(self.model.target.updates) != self.progress["step"]:
            raise ValueError("EMA count differs from optimizer updates")
        self.wrap()
        restore_rank_rng(saved["rng"]["per_rank"][self.rank], self.device)

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
        local = rank_rng_state(self.device)
        states = [None] * self.world_size
        dist.all_gather_object(states, local)
        if self.rank == 0:
            save_checkpoint(self.out / name, self.model, self.optimizer, self.progress,
                            self.data.fingerprint, self.weights_fingerprint,
                            rng_payload={**states[0], "per_rank": states, "world_size": self.world_size})
        dist.barrier()

    @torch.no_grad()
    def validate(self):
        saved_rng, training = rank_rng_state(self.device), self.model.training
        self.model.eval()
        metrics = {}
        try:
            for task in (*TASKS, *(("temporal",) if self.model.target is not None else ())):
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
                record = {"stage": self.progress["stage"], "step": self.progress["step"], **metrics}
                atomic_json(self.out / f"validation_{self.progress['stage']}.json", record)
                with (self.out / "validation.jsonl").open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps({"validation": record}), flush=True)
            return metrics
        finally:
            restore_rank_rng(saved_rng, self.device)
            self.model.train(training)

    def enter_stage2(self):
        if self.progress["stage"] != "stage1" or not self.progress["stage_complete"]:
            raise ValueError("Complete Stage 1 before initializing the EMA target")
        self.ddp = None
        gc.collect()
        self.model.begin_stage2()
        self.optimizer = optimizer_for(self.model, self.cfg)
        self.progress.update(stage="stage2", step=0, stage_complete=False, replay_index=0)
        self.progress.pop("best_validation", None)
        self.wrap()

    def run_stage(self):
        if self.progress["stage_complete"]:
            return True
        self.model.train()
        stream = BatchPrefetch(self.data, self.cfg, self.progress, self.rank, self.world_size)
        try:
            while True:
                flags = torch.tensor([int(self.stop), int(phase_finished(self.progress, self.cfg, time.time()))],
                                     device=self.device, dtype=torch.int32)
                dist.all_reduce(flags, op=dist.ReduceOp.MAX)
                stop, finished = flags.tolist()
                if stop or finished:
                    self.stop = bool(stop)
                    self.progress["stage_complete"] = bool(finished)
                    break
                torch.cuda.reset_peak_memory_stats(self.device)
                start = time.monotonic()
                task, replay_task, batches, marker = stream.next()
                data_seconds = time.monotonic() - start
                self.optimizer.zero_grad(set_to_none=True)
                parts_sum = {}
                for micro, (batch, replay_batch) in enumerate(batches):
                    with self.ddp.no_sync() if micro + 1 < len(batches) else nullcontext():
                        loss, parts = self.ddp(task, batch, replay_task, replay_batch)
                        finite = torch.isfinite(loss).to(torch.int32)
                        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                        if not finite:
                            raise FloatingPointError(f"Nonfinite {task} loss on a rank")
                        (loss / len(batches)).backward()
                    for key, value in parts.items():
                        parts_sum[key] = parts_sum.get(key, 0.) + value.detach().float() / len(batches)
                    del loss, parts, batch, replay_batch
                norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], self.cfg["max_grad_norm"],
                    error_if_nonfinite=True)
                self.optimizer.step()
                if self.progress["stage"] == "stage2":
                    self.model.target.update(self.model.encoder, self.cfg["ema_momentum"])
                self.progress.update(marker)
                self.progress["step"] += 1
                names = sorted(parts_sum)
                values = torch.stack([parts_sum[name] for name in names])
                dist.all_reduce(values)
                values /= self.world_size
                torch.cuda.synchronize(self.device)
                seconds = time.monotonic() - start
                rank_stats = torch.tensor([seconds, data_seconds, torch.cuda.max_memory_allocated(self.device) / 1024**3,
                                           torch.cuda.max_memory_reserved(self.device) / 1024**3], device=self.device)
                statistics = [torch.empty_like(rank_stats) for _ in range(self.world_size)]
                dist.all_gather(statistics, rank_stats)
                sizes = self.cfg.get("task_batch_sizes", {})
                global_batch = sizes.get(task, self.cfg["batch_size"]) * len(batches) * self.world_size
                record = {"stage": self.progress["stage"], "step": self.progress["step"], "task": task,
                          **dict(zip(names, values.cpu().tolist())), "grad_norm": float(norm),
                          "seconds": max(float(x[0]) for x in statistics), "global_batch": global_batch,
                          "ranks": [dict(zip(("seconds", "data_wait_seconds", "peak_allocated_gib", "peak_reserved_gib"),
                                             x.cpu().tolist())) for x in statistics],
                          "ema_updates": 0 if self.model.target is None else int(self.model.target.updates),
                          "wall_unix": time.time()}
                if self.rank == 0:
                    with (self.out / "metrics.jsonl").open("a") as handle:
                        handle.write(json.dumps(record) + "\n")
                    print(json.dumps(record), flush=True)
                    self.write_status("running", last_update=record)
                del batches, parts_sum, values
                if self.progress["step"] % self.cfg["validate_every"] == 0:
                    metrics = self.validate()
                    score = metrics.get("temporal_loss", sum(metrics.values()) / max(1, len(metrics)))
                    if score < self.progress.get("best_validation", float("inf")):
                        self.progress["best_validation"] = score
                        self.save("best_" + self.progress["stage"] + ".pt")
                if self.progress["step"] % self.cfg["save_every"] == 0:
                    self.save()
        finally:
            stream.close()
        # At a wall-clock boundary, retain the most recent periodic validation;
        # save promptly instead of extending the user's requested run budget.
        if not self.cfg.get("total_hours", 0) and self.progress["stage_complete"]:
            self.validate()
        self.save()
        if self.progress["stage_complete"]:
            self.save(self.progress["stage"] + ".pt")
        self.write_status("stopped" if self.stop else "phase_complete")
        return self.progress["stage_complete"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    if args.resume and args.config:
        parser.error("Resume uses its checkpoint configuration")
    rank, local_rank, world_size = (int(os.environ[key]) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if world_size < 2:
        parser.error("Launch with torchrun on at least two GPUs")
    allowed = sorted(os.sched_getaffinity(0))
    threads = int(os.environ.get("OMP_NUM_THREADS", "8"))
    selection = allowed[local_rank * threads:(local_rank + 1) * threads]
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
        saved = read_checkpoint(args.resume) if args.resume else None
        cfg = saved["config"] if saved else load_config(args.config)
        seed_all(cfg["seed"])
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        data = UnifiedData(cfg)
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
        if saved:
            trainer.resume(saved)
        else:
            trainer.wrap()
            seed_all(cfg["seed"] + rank)
        if rank == 0:
            atomic_json(out / "config.json", cfg)
            atomic_json(out / "data_protocol.json", data.metadata)
            atomic_json(out / "model.json", {**model.metadata, "world_size": world_size,
                        "precision": "BF16 autocast with FP32 trainable parameters" if cfg.get("amp") else "BF16 frozen base, FP32 adapters"})
        dist.barrier()
        trainer.initialize_clock()
        if trainer.progress["stage"] == "stage1":
            complete = trainer.run_stage()
            if complete and not trainer.stop:
                if not cfg.get("total_hours", 0) or time.time() < trainer.progress["deadline_unix"]:
                    trainer.enter_stage2()
                    trainer.save()  # crash recovery immediately after the stage transition
        if trainer.progress["stage"] == "stage2" and not trainer.stop:
            trainer.run_stage()
        trainer.write_status("stopped" if trainer.stop else "complete", replica_parameter_spread=trainer.parameter_spread())
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
