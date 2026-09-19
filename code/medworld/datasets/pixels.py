"""Reconstruct historical canvases from source files without disk caches."""
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

def source_canvas(record):
    """Reproduce the frozen cohort's uint8 canvas using its recorded box."""
    y, x, height, width = record["box"]
    with Image.open(record["image"]) as image:
        resized = np.asarray(image.convert("L").resize((width, height), Image.Resampling.BICUBIC))
    canvas = np.zeros((512, 512), dtype=np.uint8)
    canvas[y:y + height, x:x + width] = resized
    return torch.from_numpy(canvas)[None].float() / 255


def low_resolution(hr, scale=4):
    # Preserve the previous uint8 rounding, including antialiasing and clamping.
    if scale != 4:
        raise ValueError("Unified current SR uses scale 4")
    lr = F.interpolate(hr[None], scale_factor=1 / scale, mode="bicubic", align_corners=False, antialias=True)[0]
    return (lr.clamp(0, 1) * 255).round().byte().float() / 255


def human_target(record):
    y, x, height, width = [n // 2 for n in record["box"]]
    target = torch.zeros(2, 256, 256)
    for organ, path in enumerate(record["masks"]):
        with Image.open(path) as image:
            values = np.array(image.convert("L").resize((width, height), Image.Resampling.NEAREST), copy=True)
        target[organ, y:y + height, x:x + width] = torch.from_numpy(values > 0)
    return target
