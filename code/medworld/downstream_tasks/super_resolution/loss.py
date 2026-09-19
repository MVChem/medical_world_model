"""Super-resolution reconstruction loss and per-image PSNR/SSIM metrics."""
import torch


def super_resolution_loss(prediction, target, mask):
    return (((prediction.float() - target).square() * mask).sum((1, 2, 3)) /
            mask.sum((1, 2, 3)).clamp_min(1)).mean()
