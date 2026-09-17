"""Explicit Qwen patch order and shared image/feature view transforms."""
import torch
import torch.nn.functional as F


def patch_grid(tokens, height, width, merge=2):
    """[..., H*W, C] merge-block order -> [..., C, H, W], before merger.

    Qwen image processor orders [H//m, W//m, m_y, m_x], not raster.
    Only single-frame images are supported by this experiment.
    """
    if height % merge or width % merge or tokens.shape[-2] != height * width:
        raise ValueError("Patch count/grid/merge mismatch")
    prefix, channels = tokens.shape[:-2], tokens.shape[-1]
    x = tokens.reshape(-1, height // merge, width // merge, merge, merge, channels)
    return x.permute(0, 5, 1, 3, 2, 4).reshape(*prefix, channels, height, width)


def view(x, theta, size=None):
    """Same normalized affine coordinates for every feature/image resolution."""
    size = size or x.shape[-2:]
    theta = theta.to(device=x.device, dtype=x.dtype)
    if theta.ndim == 2:
        theta = theta[None].expand(len(x), -1, -1)
    grid = F.affine_grid(theta, (len(x), x.shape[1], *size), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def jitter(generator):
    # Zoom into a translated crop. No left/right flips for anatomical prompts.
    scale = float(torch.empty(1).uniform_(.75, .95, generator=generator))
    shift = torch.empty(2).uniform_(-(1 - scale), 1 - scale, generator=generator)
    return torch.tensor([[scale, 0, shift[0]], [0, scale, shift[1]]])


def feature_loss(field, targets, thetas, valid):
    """D(T(F_high)) = frozen_encoder(T(image)); no moving target or HR input."""
    losses = []
    for j in range(targets.shape[1]):
        transformed = view(field.float(), thetas[:, j])
        predicted = F.adaptive_avg_pool2d(transformed, targets.shape[-2:])
        support = F.adaptive_avg_pool2d(view(valid.float(), thetas[:, j], field.shape[-2:]), targets.shape[-2:])
        support = (support > .99).float()
        target = targets[:, j].detach().float()
        error = (F.normalize(predicted, dim=1) - F.normalize(target, dim=1)).square().sum(1, keepdim=True)
        losses.append((error * support).sum() / support.sum().clamp_min(1))
    return torch.stack(losses).mean()


def semantic_loss(attention, probabilities, valid, min_spread=.08):
    """Weak 2x2 crop targets; no claim of pixel labels or teacher attention.

    A phrase is eligible only if VLM crop scores vary and contain positive
    evidence. Compare probability mass in valid quadrants, not per-pixel masks.
    """
    support = F.interpolate(valid.float(), attention.shape[-2:], mode="area")
    mass = F.adaptive_avg_pool2d(attention * support, (2, 2)).flatten(2)
    mass = mass / mass.sum(-1, keepdim=True).clamp_min(1e-8)
    scores = probabilities.detach().float().flatten(2)
    crop_valid = F.adaptive_avg_pool2d(valid.float(), (2, 2)).flatten(2) > .2
    eligible = ((scores.amax(-1) - scores.amin(-1) >= min_spread)
                & (scores.amax(-1) >= .55) & crop_valid.all(-1))
    target = scores.clamp(.001, .999).logit().softmax(-1)
    kl = (target * (target.clamp_min(1e-8).log() - mass.clamp_min(1e-8).log())).sum(-1)
    loss = (kl * eligible).sum() / eligible.sum().clamp_min(1)
    return loss, eligible.float().mean()
