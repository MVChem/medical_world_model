"""Organ segmentation loss and per-image Dice metrics."""
import torch.nn.functional as F


def segmentation_loss(prediction, target, mask):
    prediction = prediction.float()
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Segmentation predictions, targets and channel masks must align")
    active = mask.sum((2, 3)) > 0
    if not active.any(1).all():
        raise ValueError("Every segmentation sample must have a supervised channel")
    bce = F.binary_cross_entropy_with_logits(prediction, target, reduction="none")
    bce = ((bce * mask).sum((1, 2, 3)) / mask.sum((1, 2, 3)).clamp_min(1)).mean()
    p, t = prediction.sigmoid() * mask, (target > .5).float() * mask
    dice = (2 * (p * t).sum((2, 3)) + 1) / (p.sum((2, 3)) + t.sum((2, 3)) + 1)
    dice = ((dice * active).sum(1) / active.sum(1)).mean()
    return bce + 1 - dice
