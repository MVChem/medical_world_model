"""Shared state validation and image-decoder building blocks."""
import math
import torch
from torch import nn
from ... import STATE_WIDTH

def validate_state(state):
    if (not isinstance(state, torch.Tensor) or state.ndim != 3 or state.shape[0] < 1
            or tuple(state.shape[1:]) != (8, STATE_WIDTH) or not state.is_floating_point()
            or not torch.isfinite(state).all()):
        raise ValueError("Expected finite floating state [B,8,1024]")

def block(a, b, stride=1):
    return nn.Sequential(nn.Conv2d(a, b, 3, stride=stride, padding=1), nn.GroupNorm(4, b), nn.GELU())


def positional_encoding(height, width, channels=64):
    frequency = torch.exp(-math.log(10000) * torch.arange(channels // 4) / (channels // 4))
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    xx, yy = x.flatten()[:, None] * frequency, y.flatten()[:, None] * frequency
    return torch.cat([xx.sin(), xx.cos(), yy.sin(), yy.cos()], -1)[None]
