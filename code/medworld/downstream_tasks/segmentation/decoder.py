"""Spatial output head on the same Transformer-decoded image tokens."""
import torch.nn.functional as F
from torch import nn
from ..common import block


class SegmentationHead(nn.Module):
    def __init__(self, width=256, channels=6):
        super().__init__()
        self.decode = nn.Sequential(block(width, 64), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                    block(64, 32), nn.Conv2d(32, channels, 1))

    def forward(self, image_features):
        size = int(image_features.shape[1] ** .5)
        image = image_features.transpose(1, 2).reshape(len(image_features), -1, size, size)
        return self.decode(F.interpolate(image, (128, 128), mode='bilinear', align_corners=False))
