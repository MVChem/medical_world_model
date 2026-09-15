"""Check image isolation and benchmark native vision batches on actual inputs."""
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
from common import atomic, load_model_spec
from frozen_slots_extract import branch_image, load_vision


def main(args):
    torch.set_num_threads(4)
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    uuid = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    with (Path("/tmp") / f"medworld-frozen-slots-{uuid}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        images = np.load(args.run / "data/images.npy", mmap_mode="r")
        lr_images = np.load(args.run / "data/lr_images.npy", mmap_mode="r")
        extractor, metadata = load_vision(load_model_spec(args.model))
        results, equivalence, references = [], [], {}
        for branch in ("hr", "lr"):
            batch = [branch_image(branch, i, images, lr_images) for i in range(4)]
            singles = np.stack([extractor(image) for image in batch]).astype(np.float32)
            references[branch] = singles
            multiple = extractor.batch(batch).astype(np.float32)
            # CUDA BF16 may choose different GEMM kernels at different batches.
            relative_l2 = float(np.linalg.norm(multiple - singles) / np.linalg.norm(singles))
            cosine = float((multiple * singles).sum() / np.linalg.norm(multiple) / np.linalg.norm(singles))
            record = dict(branch=branch, max_absolute_error=float(np.abs(multiple - singles).max()),
                          relative_l2=relative_l2, cosine=cosine, tolerance_relative_l2=.02, tolerance_cosine=.999)
            if relative_l2 >= .02 or cosine < .999:
                raise AssertionError(f"batch image-equivalence check failed: {record}")
            # The first image must not depend on neighboring images in its batch.
            changed = extractor.batch([batch[0], batch[0], batch[0], batch[0]])[0].astype(np.float32)
            isolation_error = float(np.linalg.norm(changed - multiple[0]) / np.linalg.norm(multiple[0]))
            if isolation_error >= .002:
                raise AssertionError(f"cross-image contamination: {isolation_error}")
            record["neighbor_change_relative_l2"] = isolation_error
            equivalence.append(record)
            print(json.dumps(record), flush=True)
        samples = [branch_image("hr", i, images, lr_images) for i in range(64)]
        for batch_size in [int(v) for v in args.batch_sizes.split(",")]:
            torch.cuda.reset_peak_memory_stats()
            try:
                extractor.batch(samples[:batch_size])
                timings = []
                for _ in range(3):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    actual = extractor.batch(samples[:batch_size])
                    torch.cuda.synchronize()
                    timings.append(time.perf_counter() - start)
                n = min(4, batch_size)
                difference = actual[:n].astype(np.float32) - references["hr"][:n]
                relative_l2 = float(np.linalg.norm(difference) / np.linalg.norm(references["hr"][:n]))
                if relative_l2 >= .02:
                    raise AssertionError(f"batch {batch_size} changed image slots beyond BF16 tolerance: {relative_l2}")
                result = dict(batch_size=batch_size, mean_seconds=float(np.mean(timings)),
                              reference_relative_l2=relative_l2,
                              images_per_second=float(batch_size / np.mean(timings)),
                              peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024 ** 2)
            except torch.OutOfMemoryError:
                result = dict(batch_size=batch_size, status="out_of_memory")
                torch.cuda.empty_cache()
            results.append(result)
            print(json.dumps(result), flush=True)
        atomic(args.run / f"extract_batch_benchmark_{args.model}.json",
               dict(model=args.model, gpu=gpu, equivalence=equivalence, results=results,
                    vision_parameters=metadata["vision_parameters"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-sizes", default="1,8,16,32,64")
    main(parser.parse_args())
