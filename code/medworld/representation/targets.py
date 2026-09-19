"""Frozen native-vision targets and online crop consistency. No task decoder dependency."""
import hashlib
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from ..adaptation import capture_depths, shared_frozen_copy

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



class FrozenFeatureTeacher(nn.Module):
    """Independent eval module tree sharing only immutable pretrained weights.

    Construct BEFORE adapting the online vision encoder. No moving online/EMA
    parameters or task targets enter the teacher. Projection is reproducible
    from a private seed, so compact checkpoints need no extra frozen tensors.
    """
    def __init__(self, vision, width=64):
        super().__init__()
        self.vision = shared_frozen_copy(vision)
        self.requires_grad_(False).eval()
        generator = torch.Generator().manual_seed(9137)
        projection = torch.randn(vision.config.hidden_size, width, generator=generator)
        self.register_buffer("projection", torch.linalg.qr(projection, mode="reduced").Q)

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, pixels, processor, vision_pixels):
        arrays = pixels.detach().float().clamp(0, 1).mul(255).round().byte().cpu().numpy()
        images = [Image.fromarray(x[0]).convert("RGB") for x in arrays]
        inputs = processor.image_processor(images=images, return_tensors="pt",
                                          min_pixels=vision_pixels**2, max_pixels=vision_pixels**2)
        device = self.projection.device
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        grid = inputs["image_grid_thw"]
        if not torch.equal(grid, grid[:1].expand_as(grid)) or int(grid[0, 0]) != 1:
            raise ValueError("Teacher requires equally sized single-frame images")
        with capture_depths(self.vision.blocks) as captured:
            self.vision(hidden_states=inputs["pixel_values"].to(next(self.vision.parameters()).dtype),
                        grid_thw=grid)
        raw = captured[3].reshape(len(pixels), -1, captured[3].shape[-1]).float()
        features = F.layer_norm(raw, (raw.shape[-1],)) @ self.projection
        return patch_grid(features, int(grid[0, 1]), int(grid[0, 2]), self.vision.config.spatial_merge_size)


def consistency_loss(field, pixels, valid, ids, teacher, processor, cfg):
    """Identity plus deterministic input-only crops, computed and discarded online."""
    identity = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    targets, transforms = [], []
    for j in range(cfg.get("visual_consistency_views", 2) + 1):
        matrices = []
        for sample_id in ids:
            if j == 0:
                matrices.append(identity)
            else:
                seed = int.from_bytes(hashlib.sha256(
                    f"{cfg['seed']}:{sample_id}:{j}".encode()).digest()[:8], "little")
                matrices.append(jitter(torch.Generator().manual_seed(seed)))
        theta = torch.stack(matrices).to(pixels.device)
        # Preserve original uint8 input for identity; affine views use the same
        # normalized coordinates as the high-resolution feature field and mask.
        target = teacher(pixels if j == 0 else view(pixels, theta), processor, cfg["vision_pixels"])
        targets.append(target)
        transforms.append(theta)
    return feature_loss(field, torch.stack(targets, 1), torch.stack(transforms, 1), valid)
