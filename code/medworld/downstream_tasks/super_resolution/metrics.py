"""Super-resolution reconstruction loss and per-image PSNR/SSIM metrics."""
import torch


def super_resolution_metrics(prediction, target, mask):
    import numpy as np
    from skimage.metrics import structural_similarity
    p = prediction.clamp(0, 1)
    mse = float(((p - target).square() * mask).sum() / mask.sum())
    yy, xx = torch.where(mask[0, 0] > 0)
    region = (slice(int(yy.min()), int(yy.max()) + 1), slice(int(xx.min()), int(xx.max()) + 1))
    return {"psnr": float(-10 * np.log10(max(mse, 1e-12))),
             "ssim": float(structural_similarity(target[0, 0].numpy()[region],
                           p[0, 0].numpy()[region], data_range=1.0))}
