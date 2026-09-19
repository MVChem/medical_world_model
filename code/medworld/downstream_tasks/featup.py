"""FeatUp-inspired guided decoding and fixed native-vision feature consistency.

Adapted from the local spatial pilot; not a reproduction of FeatUp's JBU.
Only input pixels and four differentiable visual slots enter the decoder.
"""
import hashlib

from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from ..adaptation import capture_depths, shared_frozen_copy
from .common import block, positional_encoding

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



class ReadSlots(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.slot_norm = nn.LayerNorm(1024)
        self.project = nn.Linear(1024, width)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True, dropout=0)
        self.gate = nn.Parameter(torch.tensor(.1))

    def forward(self, feature, slots):
        b, c, h, w = feature.shape
        positions = positional_encoding(h, w, c).to(feature)
        query = self.query_norm(feature.flatten(2).transpose(1, 2) + positions)
        values = self.project(self.slot_norm(slots))
        value_positions = positional_encoding(1, slots.shape[1], c).to(values)
        # Always use the same attention implementation when exporting / training.
        output, weights = self.attention(query, values + value_positions, values,
                                         need_weights=True, average_attn_weights=False)
        feature = feature + self.gate.tanh() * output.transpose(1, 2).reshape(b, c, h, w)
        return feature, weights.mean(1).transpose(1, 2).reshape(b, -1, h, w)


class GuidedUpsample(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.kernel = nn.Sequential(block(1, 16), nn.Conv2d(16, 9, 1))
        self.refine = block(width, width)

    def forward(self, field, image):
        size = tuple(2 * n for n in field.shape[-2:])
        guidance = F.interpolate(image, size, mode="bilinear", align_corners=False)
        logits = self.kernel(guidance)
        # Mask off-image neighbors so padding cannot attenuate a constant field.
        valid = F.unfold(torch.ones_like(guidance), 3, padding=1).reshape(len(image), 9, *size)
        weights = logits.masked_fill(valid == 0, -torch.inf).softmax(1)
        up = F.interpolate(field, size, mode="bilinear", align_corners=False)
        neighbors = F.unfold(up, 3, padding=1).reshape(len(image), up.shape[1], 9, *size)
        return self.refine((neighbors * weights[:, None]).sum(2)), weights



class FeatUpSpatialHead(nn.Module):
    def __init__(self, task, width=64):
        super().__init__()
        if task not in ("segmentation", "sr"):
            raise ValueError(task)
        self.task = task
        self.stem = nn.Sequential(block(1, 32, 2), block(32, width, 2))
        self.read32, self.read64 = ReadSlots(width), ReadSlots(width)
        self.upsample = GuidedUpsample(width)
        self.feature = nn.Conv2d(width, width, 1)
        widths = (32, 16) if task == "segmentation" else (32, 16, 8)
        layers, previous = [], width
        for channels in widths:
            layers += [nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                       block(previous, channels)]
            previous = channels
        self.decode = nn.Sequential(*layers, nn.Conv2d(previous, 3 if task == "segmentation" else 1, 1))

    def forward(self, image, slots, *, return_features=False):
        if tuple(slots.shape[1:]) != (4, 1024):
            raise ValueError("FeatUp spatial tasks require four visual slots")
        field = F.adaptive_avg_pool2d(self.stem(image), (32, 32))
        field, _ = self.read32(field, slots)
        field, _ = self.upsample(field, image)
        field, _ = self.read64(field, slots)
        prediction = self.decode(field)
        if self.task == "sr":
            prediction = F.interpolate(image, size=prediction.shape[-2:], mode="bicubic",
                                       align_corners=False) + .1 * prediction
        if return_features:
            return prediction, self.feature(field)
        return prediction


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
    for j in range(cfg.get("featup_views", 2) + 1):
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
