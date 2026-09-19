"""Bounded real-data capacity/throughput probe; never writes trained weights."""
import argparse
import gc
import json
from pathlib import Path
import time

from .gpu import acquire_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--config", help="Probe the actual training architecture/configuration")
    parser.add_argument("--replay-tasks", default="", help="Temporal replay tasks, one per repeat cyclically")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batches", default="4,8,16,32")
    parser.add_argument("--tasks", default="classification,report,segmentation,sr,temporal")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-memory-gib", type=float, default=20)
    parser.add_argument("--image-workers", type=int, default=8)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        parser.error("Benchmark output already exists")
    lock, device = acquire_gpu(args.gpu)
    import torch
    from .config import load_config
    from .datasets import UnifiedData
    from .model import MedWorld
    from .runtime import optimizer_for, seed_all
    cfg = load_config(args.config, overrides={"amp": True, "ce_chunk_tokens": 128, "image_workers": args.image_workers})
    seed_all(cfg["seed"])
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    data = UnifiedData(cfg)
    model = MedWorld(cfg, device).train()
    model.begin_stage2()
    model.pos_weight.copy_(data.current.pos_weight.to(device))
    optimizer = optimizer_for(model, cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / (out.stem + "_config.json")).write_text(json.dumps(cfg, indent=2) + "\n")
    replay_tasks = [t for t in args.replay_tasks.split(",") if t]
    for task in args.tasks.split(","):
        for size in map(int, args.batches.split(",")):
            last_peak = 0
            for repetition in range(args.repeats):
                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                start = time.monotonic()
                batch = data.training_batch(task, repetition * size, size, cfg["seed"])
                replay_task = replay_tasks[repetition % len(replay_tasks)] if task == "temporal" and replay_tasks else None
                replay_batch = (data.training_batch(replay_task, repetition * size,
                                cfg.get("replay_batch_sizes", {}).get(replay_task, cfg.get("task_batch_sizes", {}).get(replay_task, size)), cfg["seed"])
                                if replay_task else None)
                loaded = time.monotonic()
                try:
                    loss, parts = model(task, batch, replay_task, replay_batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite benchmark loss")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], cfg["max_grad_norm"], error_if_nonfinite=True)
                    optimizer.step()
                    if task == "temporal":
                        model.target.update(model.encoder, cfg["ema_momentum"])
                    torch.cuda.synchronize()
                    seconds = time.monotonic() - start
                    last_peak = torch.cuda.max_memory_allocated() / 1024**3
                    record = {"task": task, "batch_size": size, "repetition": repetition, "replay_task": replay_task,
                              "seconds": seconds, "data_seconds": loaded - start,
                              "samples_per_second": size / seconds, "loss": float(loss.detach()),
                              "peak_allocated_gib": last_peak,
                              "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3}
                    del loss, parts, norm, batch, replay_batch
                except torch.cuda.OutOfMemoryError:
                    record = {"task": task, "batch_size": size, "repetition": repetition, "oom": True}
                    last_peak = float("inf")
                    model.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                with out.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
                if record.get("oom"):
                    return
                if last_peak > args.max_memory_gib:
                    break
            if last_peak > args.max_memory_gib:
                break
    lock.close()


if __name__ == "__main__":
    main()
