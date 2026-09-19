"""Organ segmentation loss and per-image Dice metrics."""
import numpy as np
import torch.nn.functional as F


def segmentation_metrics(prediction, target, mask):
    p = (prediction[:, :target.shape[1]].sigmoid() > .5).float() * mask
    t = (target > .5).float() * mask
    dice = ((2 * (p * t).sum((2, 3)) + 1e-6) /
            (p.sum((2, 3)) + t.sum((2, 3)) + 1e-6))[0].tolist()
    return {"dice_per_organ": dice, "mean_dice": float(np.mean(dice))}
