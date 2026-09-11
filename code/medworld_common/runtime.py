"""Filesystem and reproducibility utilities shared by experiments."""

import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    os.chmod(temp, 0o600)
    os.replace(temp, path)


def atomic_torch(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(data, temp)
    os.chmod(temp, 0o600)
    os.replace(temp, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)


def load_rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_rows(path, rows):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    with temp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.chmod(temp, 0o600)
    temp.replace(path)
