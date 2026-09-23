"""One observation encoder for current tasks, forecasting and retrodiction."""
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
import torch.nn.functional as F

from . import STATE_WIDTH
from .adaptation import LANGUAGE_TARGETS, adapt_selected, capture_depths


class FrozenJEPA(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        from .third_party.vjepa2.vision_transformer import vit_base
        self.backbone = vit_base(img_size=(384, 384), patch_size=16, num_frames=64,
                                 tubelet_size=2, use_sdpa=True, use_rope=True,
                                 img_temporal_dim_size=1, interpolate_rope=True)
        saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)["ema_encoder"]
        weights = {k.replace("module.", "").replace("backbone.", ""): v for k, v in saved.items()}
        self.backbone.load_state_dict(weights, strict=True)
        self.requires_grad_(False).eval().to(dtype=torch.bfloat16)

    def train(self, mode=True):
        return super().train(False)

    @staticmethod
    def prepare_pixels(images):
        arrays = [np.array(ImageOps.pad(im.convert("RGB"), (384, 384),
                          method=Image.Resampling.BICUBIC, color="black"), copy=True) for im in images]
        return torch.stack([torch.from_numpy(x).permute(2, 0, 1) for x in arrays])

    @torch.no_grad()
    def forward(self, images, prepared_pixels=None):
        device = next(self.parameters()).device
        pixels = self.prepare_pixels(images) if prepared_pixels is None else prepared_pixels
        pixels = pixels.to(device).float() / 255
        mean = pixels.new_tensor([.485, .456, .406])[None, :, None, None]
        std = pixels.new_tensor([.229, .224, .225])[None, :, None, None]
        # V-JEPA RoPE computes q/k in FP32 internally; autocast makes the SDPA
        # q/k/v dtypes consistent with the frozen BF16 value projection.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            features = self.backbone(((pixels - mean) / std).unsqueeze(2).to(next(self.parameters()).dtype))
        if tuple(features.shape[1:]) != (576, 768):
            raise ValueError(f"Unexpected JEPA feature shape: {features.shape}")
        grid = features.transpose(1, 2).reshape(-1, 768, 24, 24)
        return F.adaptive_avg_pool2d(grid, (8, 8)).flatten(2).transpose(1, 2)


class StateEncoder(nn.Module):
    def __init__(self, language, vision, cfg):
        super().__init__()
        self.language, self.vision = language, vision
        hidden, visual_width = language.get_input_embeddings().weight.shape[1], vision.config.hidden_size
        adapt_selected(language, language.layers, LANGUAGE_TARGETS, cfg)
        adapt_selected(vision, vision.blocks, {"qkv", "proj"}, cfg)
        self.jepa_adapter = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden),
                                         nn.GELU(), nn.Linear(hidden, hidden))
        self.jepa_positions = nn.Parameter(torch.randn(1, 64, hidden) * .02)
        self.slot_queries = nn.Parameter(torch.randn(8, STATE_WIDTH) * .02)
        self.language_query = nn.Linear(STATE_WIDTH, hidden)
        self.fusion_readouts = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, STATE_WIDTH)) for _ in range(4)])
        self.visual_readouts = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(visual_width), nn.Linear(visual_width, STATE_WIDTH)) for _ in range(4)])
        self.visual_norm = nn.LayerNorm(STATE_WIDTH)

    def forward(self, image_inputs, text_ids, text_mask, features):
        grid = image_inputs["image_grid_thw"]
        batch = len(grid)
        if not batch or not torch.equal(grid, grid[:1].expand_as(grid)):
            raise ValueError("Use one equally prepared square image per observation")
        with capture_depths(self.vision.blocks) as captured:
            self.vision(hidden_states=image_inputs["pixel_values"].to(next(self.vision.parameters()).dtype),
                        grid_thw=grid)
        visual = []
        for j in range(4):
            raw = captured[j].reshape(batch, -1, captured[j].shape[-1])
            values = self.visual_readouts[j](raw.float())
            query = self.slot_queries[4 + j]
            weights = ((values * query).sum(-1) / math.sqrt(STATE_WIDTH)).softmax(-1)
            visual.append(self.visual_norm((weights[..., None] * values).sum(1) + query))
        if features is None or features.shape != (batch, 64, 768) or text_ids is None or text_mask is None:
            raise ValueError("Full state requires aligned JEPA features and observation text")
        embed = self.language.get_input_embeddings()
        prefix = (self.jepa_adapter(features.float()) + self.jepa_positions).to(embed.weight.dtype)
        queries = self.language_query(self.slot_queries[:4])[None].expand(batch, -1, -1).to(embed.weight.dtype)
        sequence = torch.cat([prefix, embed(text_ids), queries], 1)
        mask = torch.cat([text_mask.new_ones(prefix.shape[:2]), text_mask,
                          text_mask.new_ones(queries.shape[:2])], 1)
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        with capture_depths(self.language.layers) as captured:
            self.language(inputs_embeds=sequence, attention_mask=mask, position_ids=positions,
                          use_cache=False, return_dict=True)
        fusion = [self.fusion_readouts[j](captured[j][:, -4 + j].float()) for j in range(4)]
        return torch.stack(fusion + visual, 1)
