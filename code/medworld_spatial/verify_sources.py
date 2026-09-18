"""Stream reconstructed pixels into historical NPY hashes without writing arrays."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import torch

from medworld.config import PROJECT
from medworld.runtime import atomic_json
from .online import source_canvas, low_resolution, human_target


def npy_digest(shape):
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {"descr": "|u1", "fortran_order": False, "shape": shape})
    return hashlib.sha256(header.getvalue())


def verify(out):
    torch.set_num_threads(2)
    directory = PROJECT / "code/medworld_dense_baselines/runs/frozen_slots_20260913/data"
    rows = [json.loads(line) for line in (directory / "observations.jsonl").read_text().splitlines()]
    manifest = json.loads((directory / "manifest.json").read_text())
    digests = {"image_sha256": npy_digest((len(rows), 512, 512)),
               "lr_image_sha256": npy_digest((len(rows), 128, 128)),
               "human_mask_sha256": npy_digest((138, 2, 256, 256))}
    start = time.time()
    # Bounded batches: never retain the full decoded dataset, even in RAM.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for offset in range(0, len(rows), 16):
            for row, hr in zip(rows[offset:offset + 16], pool.map(source_canvas, rows[offset:offset + 16])):
                digests["image_sha256"].update((hr[0] * 255).round().byte().numpy().tobytes())
                digests["lr_image_sha256"].update((low_resolution(hr)[0] * 255).round().byte().numpy().tobytes())
                if row["kind"] == "montgomery":
                    digests["human_mask_sha256"].update(human_target(row).byte().numpy().tobytes())
            if offset % 512 == 0:
                print(json.dumps({"verified": offset, "total": len(rows), "elapsed": time.time() - start}), flush=True)
    actual = {key: value.hexdigest() for key, value in digests.items()}
    matches = {key: value == manifest[key] for key, value in actual.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out, {"records": len(rows), "actual": actual, "expected": {k: manifest[k] for k in actual},
                     "matches": matches, "elapsed_seconds": time.time() - start, "arrays_written": 0})
    if not all(matches.values()):
        raise ValueError(f"Reconstructed pixels differ from original prepared arrays: {matches}")
    print(json.dumps({"matches": matches, "arrays_written": 0}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    verify(p.parse_args().out)
