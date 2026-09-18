"""Sparse slots -> image-guided feature field -> actual dense predictions.

All reported attention tensors participate in the forward path. This is an
experimental adaptation of FeatUp's consistency idea, not its JBU implementation.
"""
import copy
import math

import torch
from torch import nn
import torch.nn.functional as F

from medworld.decoders import block, positional_encoding
from .geometry import patch_grid


def reader_from_state(state):
    width = state["raw_width"]
    layers = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1024)) for _ in range(4)])
    reader = SlotReader(layers, state["queries"], nn.LayerNorm(1024), state["merge"])
    reader.load_state_dict(state["state_dict"], strict=True)
    return reader


class SlotReader(nn.Module):
    def __init__(self, readouts, queries, norm, merge=2):
        super().__init__()
        self.readouts = copy.deepcopy(readouts).requires_grad_(True)
        self.queries = nn.Parameter(queries.detach().float().clone())
        self.norm = copy.deepcopy(norm).requires_grad_(True)
        self.merge = merge

    def forward(self, raw, grid):
        slots, maps = [], []
        for j, readout in enumerate(self.readouts):
            values = readout(raw[:, j].float())
            weights = ((values * self.queries[j]).sum(-1) / math.sqrt(values.shape[-1])).softmax(-1)
            slots.append(self.norm((values * weights[..., None]).sum(1) + self.queries[j]))
            maps.append(patch_grid(weights[..., None], *grid, self.merge).squeeze(1))
        return torch.stack(slots, 1), torch.stack(maps, 1)


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


class SemanticReadout(nn.Module):
    def __init__(self, text_embeddings, width):
        super().__init__()
        self.register_buffer("text_embeddings", text_embeddings.detach().float().clone())
        self.query = nn.Linear(text_embeddings.shape[-1], width)
        self.keys = nn.Conv2d(width, width, 1)
        self.values = nn.Conv2d(width, width, 1)
        self.output = nn.Conv2d(width, width, 1)
        self.gate = nn.Parameter(torch.tensor(.1))

    def forward(self, field, valid):
        b, c, h, w = field.shape
        query = self.query(self.text_embeddings)
        keys = self.keys(field).flatten(2)
        logits = torch.einsum("kc,bcn->bkn", query, keys) / math.sqrt(c)
        support = F.interpolate(valid.float(), (h, w), mode="area").flatten(2) > .5
        if not support.any(-1).all():
            raise ValueError("No valid image pixels for semantic attention")
        attention = logits.masked_fill(~support, -torch.inf).softmax(-1)
        concept_values = attention @ self.values(field).flatten(2).transpose(1, 2)
        # Reuse these actual attentions to send concept summaries back to pixels.
        routing = attention / attention.sum(1, keepdim=True).clamp_min(1e-8)
        context = (routing.transpose(1, 2) @ concept_values).transpose(1, 2).reshape(b, c, h, w)
        return field + self.gate.tanh() * self.output(context), attention.reshape(b, -1, h, w)


class SparseSpatialModel(nn.Module):
    def __init__(self, reader, text_embeddings, variant="featup_semantic", width=64):
        super().__init__()
        if variant not in ("image_only", "visual_slots", "slots", "featup", "featup_semantic", "image_only_featup"):
            raise ValueError(variant)
        self.variant, self.reader = variant, reader
        self.stem = nn.Sequential(block(1, 32, 2), block(32, width, 2))
        self.read32, self.read64 = ReadSlots(width), ReadSlots(width)
        self.upsample = GuidedUpsample(width)
        self.semantic = SemanticReadout(text_embeddings, width)
        self.feature = nn.Conv2d(width, width, 1)
        self.heads = nn.ModuleDict()
        for task, output, widths in (("segmentation", 3, (32, 16)), ("sr", 1, (32, 16, 8))):
            layers, previous = [], width
            for channels in widths:
                layers += [nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), block(previous, channels)]
                previous = channels
            self.heads[task] = nn.Sequential(*layers, nn.Conv2d(previous, output, 1))

    def forward(self, batch, task, *, replacement=None):
        image, valid = batch["pixels"], batch["valid"]
        visual, encoder_attention = self.reader(batch["raw"], batch["grid"])
        # Fusion slots are frozen image-only JEPA -> adapter -> language states.
        slots = torch.cat([batch["fusion"].float(), visual], 1)
        if self.variant == "visual_slots":
            slots = torch.cat([torch.zeros_like(slots[:, :4]), slots[:, 4:]], 1)
        elif self.variant in ("image_only", "image_only_featup"):
            slots = slots * 0
        if replacement is not None:
            slots = replacement.to(slots)
        field = F.adaptive_avg_pool2d(self.stem(image), (32, 32))
        field, attention32 = self.read32(field, slots)
        field, local_attention = self.upsample(field, image)
        field, attention64 = self.read64(field, slots)
        field, semantic_attention = self.semantic(field, valid)
        prediction = self.heads[task](field)
        if task == "sr":
            # The head defines the 512 canvas; neither HR pixels nor targets are read.
            prediction = F.interpolate(image, size=prediction.shape[-2:], mode="bicubic", align_corners=False) + .1 * prediction
        return {"prediction": prediction, "feature": self.feature(field), "slots": slots,
                "encoder_attention": encoder_attention, "decoder_attention32": attention32,
                "decoder_attention64": attention64, "semantic_attention": semantic_attention,
                "local_attention": local_attention}
