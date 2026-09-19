"""Shared baseline spatial decoder and task-loss dispatch."""
import torch
from torch import nn
import torch.nn.functional as F
from ... import STATE_WIDTH
from . import block, positional_encoding
from ..segmentation.loss import segmentation_loss
from ..super_resolution.loss import super_resolution_loss

class SpatialHead(nn.Module):
    def __init__(self, task):
        super().__init__()
        if task not in ("segmentation", "sr"):
            raise ValueError(task)
        self.task = task
        self.stem = nn.Sequential(block(1, 32, 2), block(32, 64, 2))
        self.slot_normalize = nn.LayerNorm(STATE_WIDTH, elementwise_affine=False)
        self.slot_project = nn.Sequential(nn.Linear(STATE_WIDTH, 64), nn.GELU())
        self.query_normalize = nn.LayerNorm(64)
        self.attention = nn.MultiheadAttention(64, 4, batch_first=True, dropout=0)
        self.register_buffer("image_position", positional_encoding(32, 32))
        self.register_buffer("depth_position", positional_encoding(1, 4))
        self.fuse = block(128, 64)
        layers, channels = [], 64
        for width in [48, 32, 16] + ([8] if task == "sr" else []):
            layers.extend([nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), block(channels, width)])
            channels = width
        layers.append(nn.Conv2d(channels, 1 if task == "sr" else 3, 1))
        self.decode = nn.Sequential(*layers)

    def forward(self, image, slots):
        if tuple(slots.shape[1:]) != (4, STATE_WIDTH):
            raise ValueError("Spatial tasks read exactly the four visual slots")
        z = F.interpolate(self.stem(image), (32, 32), mode="bilinear", align_corners=False)
        query = self.query_normalize(z.flatten(2).transpose(1, 2) + self.image_position)
        values = self.slot_project(self.slot_normalize(slots.float())) + self.depth_position
        attended, _ = self.attention(query, values, values, need_weights=False)
        attended = attended.transpose(1, 2).reshape(len(image), 64, 32, 32)
        prediction = self.decode(self.fuse(torch.cat([z, attended], 1)))
        if self.task == "sr":
            prediction = F.interpolate(image, scale_factor=4, mode="bicubic", align_corners=False) + .1 * prediction
        return prediction


def spatial_loss(task, prediction, target, mask):
    if task == "sr":
        return super_resolution_loss(prediction, target, mask)
    return segmentation_loss(prediction, target, mask)
