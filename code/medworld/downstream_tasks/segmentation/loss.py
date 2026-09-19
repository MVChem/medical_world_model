"""Organ segmentation loss and per-image Dice metrics."""
import numpy as np
import torch.nn.functional as F


def segmentation_loss(prediction, target, mask):
    prediction = prediction[:, :target.shape[1]].float()
    bce = F.binary_cross_entropy_with_logits(prediction, target, reduction="none")
    bce = ((bce * mask).sum((1, 2, 3)) / (mask.sum((1, 2, 3)) * target.shape[1]).clamp_min(1)).mean()
    p, t = prediction.sigmoid() * mask, (target > .5).float() * mask
    dice = (2 * (p * t).sum((2, 3)) + 1) / (p.sum((2, 3)) + t.sum((2, 3)) + 1)
    return bce + 1 - dice.mean()
