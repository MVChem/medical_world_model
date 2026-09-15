"""Measure resident-model slot extraction on real inputs, without task decoders."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from data import MultiTaskData
from model import MultiTaskModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen08b", "qwen9b"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.repeats < 3:
        raise ValueError("at least three timings are required")
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    info = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=uuid,name",
                                    "--format=csv,noheader"], text=True).strip()
    uid, name = [value.strip() for value in info.split(",", 1)]
    active = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-compute-apps=pid",
                                      "--format=csv,noheader"], text=True).strip()
    if active:
        raise RuntimeError("selected GPU has another compute process")
    with Path(f"/tmp/medworld-frozen-slots-{uid}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4)
        torch.manual_seed(42)
        data = MultiTaskData()
        start = time.perf_counter()
        model = MultiTaskModel(args.model, "slots", "cuda").eval()
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - start
        results = []
        for task, label in [("classification", "full_8_slots"), ("segmentation", "visual_4_hr"),
                            ("sr", "visual_4_lr")]:
            dataset = data.dataset(task, "train")
            # A dataset example may expose one PIL as 'image'; collate is the
            # canonical model interface and avoids relying on private formats.
            batches = [data.collate(task, [dataset[i]]) for i in range(args.repeats)]
            def encode(batch):
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    return model.state(task, batch)
            for batch in batches[:3]:
                encode(batch)
            torch.cuda.synchronize()
            timings = []
            for batch in batches:
                torch.cuda.synchronize()
                started = time.perf_counter()
                state = encode(batch)
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - started)
            result = dict(branch=label, batch_size=1, images=args.repeats, warmup_images=3,
                          median_ms=float(np.median(timings) * 1000),
                          mean_ms=float(np.mean(timings) * 1000),
                          p90_ms=float(np.percentile(timings, 90) * 1000),
                          mean_images_per_second=1 / float(np.mean(timings)),
                          state_shape=list(state.shape), dtype=str(state.dtype),
                          state_bytes=state.numel() * state.element_size())
            results.append(result)
            print(json.dumps(result), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dict(model=args.model, gpu=name, physical_gpu=gpu,
            model_load_seconds=load_seconds, timings=results,
            precision="BF16 autocast with FP32 readout parameters",
            scope="image preprocessing, H2D and encoder/readouts; synchronized wall time; resident model; no task decoder, backward, or data read",
            initialization="pretrained model with new state readouts; latency probe, not a quality evaluation"), indent=2) + "\n")


if __name__ == "__main__":
    main()
