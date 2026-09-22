"""Decode original MRI and expert-reviewed masks with a bounded RAM cache."""
from functools import lru_cache
import hashlib
from pathlib import Path
from threading import RLock

import nibabel as nib
import numpy as np
from PIL import Image
import torch

_VOLUME_LOCK = RLock()


@lru_cache(maxsize=8)
def _volume(path, size, mtime_ns, is_mask, expected_sha256=None):
    # File metadata participates in the cache key, preventing stale arrays after
    # an explicitly replaced source. No cache is written to disk.
    if expected_sha256 is not None:
        digest = hashlib.sha256()
        with Path(path).open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("MRI source fingerprint differs from the prepared manifest")
    volume = nib.as_closest_canonical(nib.load(path))
    if len(volume.shape) != 3:
        raise ValueError("MRI must be a three-dimensional volume")
    values = np.asanyarray(volume.dataobj)
    if not np.isfinite(values).all():
        raise ValueError("MRI contains nonfinite voxels")
    if is_mask:
        if not np.isin(values, [0, 1, 2, 3, 4]).all():
            raise ValueError("MRI target labels must be integers 0..4")
        return values.astype(np.uint8), volume.affine
    return values.astype(np.float32), volume.affine


def _load(path, is_mask=False, expected_sha256=None):
    stat = Path(path).stat()
    # lru_cache alone can decompress the same missing key concurrently. Serialize
    # cache fills so the threaded sampler retains at most eight decoded arrays.
    with _VOLUME_LOCK:
        return _volume(str(path), stat.st_size, stat.st_mtime_ns, is_mask, expected_sha256)


def _slice(values, record):
    shape, z = record["canonical_shape"], record["slice_index"]
    if list(values.shape) != shape:
        raise ValueError("MRI volume shape differs from the prepared manifest")
    if type(z) is not int or not 0 <= z < values.shape[2]:
        raise ValueError("MRI axial slice index is out of bounds")
    # Canonical x increases towards R, y towards A. In the displayed plane,
    # anterior is at the top and left is at the left (neurological convention).
    return np.rot90(values[:, :, z])


def mri_source_canvas(record):
    """Return float 1x512x512 image; windowing never reads a target mask."""
    values, _ = _load(record["image"], expected_sha256=record.get("t1ce_sha256"))
    lo, hi = record["normalization"]
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError("MRI intensity window must contain increasing finite values")
    plane = _slice(values, record)
    pixels = np.rint(np.clip((plane - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    y, x, height, width = record["box"]
    resized = np.asarray(Image.fromarray(pixels).resize((width, height), Image.Resampling.BICUBIC))
    canvas = np.zeros((512, 512), dtype=np.uint8)
    canvas[y:y + height, x:x + width] = resized
    return torch.from_numpy(canvas)[None].float() / 255


def mri_target(record):
    """Return global six-channel targets; MRI occupies channels 2, 3, 4, 5."""
    values, affine = _load(record["mask_file"], is_mask=True, expected_sha256=record.get("mask_sha256"))
    image, image_affine = _load(record["image"], expected_sha256=record.get("t1ce_sha256"))
    if values.shape != image.shape or not np.allclose(affine, image_affine, atol=1e-4, rtol=1e-5):
        raise ValueError("MRI source and target grids differ")
    if record["channels"] != [2, 3, 4, 5]:
        raise ValueError("MRI targets require the NETC/SNFH/ET/RC channel mapping")
    plane = _slice(values, record)
    y, x, height, width = [int(value) // 2 for value in record["box"]]
    labels = np.asarray(Image.fromarray(plane).resize((width, height), Image.Resampling.NEAREST))
    target = torch.zeros(6, 256, 256)
    for label, channel in enumerate(record["channels"], start=1):
        target[channel, y:y + height, x:x + width] = torch.from_numpy(labels == label)
    return target
