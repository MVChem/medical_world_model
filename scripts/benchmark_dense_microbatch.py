"""Compare dense microbatch throughput while keeping effective batch 8."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/medworld_dense_baselines"))
from common import atomic
from frozen_slots_train import FrozenSlotHead, SlotCorpus, autocast, objective


def main(args):
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    uuid = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    with (Path("/tmp") / f"medworld-frozen-slots-{uuid}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        results = []
        for task in ("segmentation", "sr"):
            corpus = SlotCorpus(args.run, args.run, "image_only", task, "image_only", "cuda")
            ids = np.random.default_rng(20260913).permutation(corpus.pool["train"]).tolist()
            # Interleaved repetitions expose cache/warmup effects.
            for microbatch in (4, 8, 8, 4):
                torch.manual_seed(20260913)
                model = FrozenSlotHead(task).cuda()
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
                torch.cuda.reset_peak_memory_stats()
                elapsed = []
                for step in range(args.steps + 10):
                    chunk = ids[(step * 8) % (len(ids) - 8):(step * 8) % (len(ids) - 8) + 8]
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    for position in range(0, 8, microbatch):
                        ii = chunk[position:position + microbatch]
                        x, slots, target, mask = corpus.batch(ii)
                        with autocast(corpus.device):
                            prediction = model(x, slots)
                        loss = objective(task, prediction, target, mask)
                        if not torch.isfinite(loss):
                            raise ValueError("nonfinite benchmark loss")
                        (loss * len(ii) / 8).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    torch.cuda.synchronize()
                    if step >= 10:
                        elapsed.append(time.perf_counter() - started)
                record = dict(task=task, effective_batch=8, microbatch=microbatch, steps=args.steps,
                              mean_step_seconds=float(np.mean(elapsed)), median_step_seconds=float(np.median(elapsed)),
                              images_per_second=float(8 / np.mean(elapsed)),
                              peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024 ** 2)
                results.append(record)
                print(json.dumps(record), flush=True)
                del optimizer, model
                torch.cuda.empty_cache()
        atomic(args.run / "microbatch_benchmark.json", dict(gpu=gpu, results=results,
               scope="real expanded-cohort images and targets; matched decoder; effective batch unchanged",
               selected_microbatch=8))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    main(parser.parse_args())
