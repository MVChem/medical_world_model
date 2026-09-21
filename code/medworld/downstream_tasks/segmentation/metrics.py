"""Per-image, per-organ binary Dice and intersection-over-union metrics."""
import numpy as np


def segmentation_metrics(prediction, target, mask):
    p = (prediction[:, :target.shape[1]].sigmoid() > .5).float() * mask
    t = (target > .5).float() * mask
    dice = ((2 * (p * t).sum((2, 3)) + 1e-6) /
            (p.sum((2, 3)) + t.sum((2, 3)) + 1e-6))[0].tolist()
    intersection = (p * t).sum((2, 3))
    union = p.sum((2, 3)) + t.sum((2, 3)) - intersection
    # Match Dice's existing convention: two empty masks score 1.
    iou = ((intersection + 1e-6) / (union + 1e-6))[0].tolist()
    return {"dice_per_organ": dice, "mean_dice": float(np.mean(dice)),
            "iou_per_organ": iou, "mean_iou": float(np.mean(iou))}
