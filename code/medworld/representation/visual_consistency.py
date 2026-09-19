"""Training-only reconstruction of a spatial feature field from four visual slots."""
import torch
from torch import nn
import torch.nn.functional as F
from .targets import consistency_loss


class SlotSpatialReconstruction(nn.Module):
    """No image input: all sample-dependent evidence must pass through slots."""
    def __init__(self, width=64):
        super().__init__()
        coordinates = torch.linspace(-1, 1, 32)
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        self.register_buffer("coordinates", torch.stack((xx, yy), -1).reshape(1, 1024, 2), persistent=False)
        self.position = nn.Sequential(nn.Linear(2, width), nn.GELU(), nn.Linear(width, width))
        self.project = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, width))
        self.depth = nn.Parameter(torch.randn(1, 4, width) * .02)
        self.read = nn.MultiheadAttention(width, 4, batch_first=True, dropout=0)
        self.refine = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU(), nn.Conv2d(width, width, 1))

    def forward(self, slots):
        if slots.ndim != 3 or tuple(slots.shape[1:]) != (4, 1024):
            raise ValueError("Expected four visual slots [B,4,1024]")
        values = self.project(slots.float())
        queries = self.position(self.coordinates).expand(len(slots), -1, -1)
        field, _ = self.read(queries, values + self.depth, values, need_weights=True)
        field = field.transpose(1, 2).reshape(len(slots), -1, 32, 32)
        return self.refine(F.interpolate(field, (64, 64), mode="bilinear", align_corners=False))

    def loss(self, slots, pixels, valid, ids, teacher, processor, cfg):
        return consistency_loss(self(slots), pixels, valid, ids, teacher, processor, cfg)
