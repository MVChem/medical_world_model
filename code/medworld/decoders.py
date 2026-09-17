"""Differentiable task readouts; report decoding consumes only state tokens."""
import math
from numbers import Integral

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from . import STATE_WIDTH


def validate_state(state):
    if (not isinstance(state, torch.Tensor) or state.ndim != 3 or state.shape[0] < 1
            or tuple(state.shape[1:]) != (8, STATE_WIDTH) or not state.is_floating_point()
            or not torch.isfinite(state).all()):
        raise ValueError("Expected finite floating state [B,8,1024]")


def chunked_ce(states, targets, weight, chunk=32):
    valid = targets != -100
    states, targets = states[valid], targets[valid]
    if not len(targets):
        return states.sum() * 0
    def loss_fn(h, y):
        return F.cross_entropy(F.linear(h.to(weight.dtype), weight).float(), y, reduction="sum")
    total = states.new_zeros((), dtype=torch.float32)
    for start in range(0, len(targets), chunk):
        args = states[start:start + chunk], targets[start:start + chunk]
        total = total + (checkpoint(loss_fn, *args, use_reentrant=False)
                         if torch.is_grad_enabled() else loss_fn(*args))
    return total / len(targets)


class ReportDecoder(nn.Module):
    def __init__(self, language, output_head, tokenizer, cfg):
        super().__init__()
        self.language, self.output_head, self.tokenizer = language, output_head, tokenizer
        self.report_tokens = cfg["report_tokens"]
        self.ce_chunk_tokens = cfg.get("ce_chunk_tokens", 32)
        hidden = language.get_input_embeddings().weight.shape[1]
        self.projection = nn.Sequential(nn.LayerNorm(STATE_WIDTH), nn.Linear(STATE_WIDTH, hidden))
        prompt = tokenizer.apply_chat_template([{"role": "user", "content":
            "Describe the supplied chest radiograph state. Write only FINDINGS and IMPRESSION."}],
            tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
        self.register_buffer("prompt_ids", torch.tensor(prompt, dtype=torch.long), persistent=False)

    def prefix(self, state):
        validate_state(state)
        embedding = self.language.get_input_embeddings()
        state = state.to(device=embedding.weight.device)
        slots = self.projection(state.float()).to(embedding.weight.dtype)
        prompt = embedding(self.prompt_ids)[None].expand(len(state), -1, -1)
        return torch.cat([slots, prompt], 1)

    def targets(self, reports, device):
        rows = [self.tokenizer.encode(report, add_special_tokens=False)[:self.report_tokens - 1]
                + [self.tokenizer.eos_token_id] for report in reports]
        ids = torch.full((len(rows), max(map(len, rows))), self.tokenizer.pad_token_id,
                         device=device, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(rows):
            ids[i, :len(row)] = torch.tensor(row, device=device)
            mask[i, :len(row)] = 1
        return ids, mask

    def loss(self, state, reports):
        if len(reports) != len(state) or any(not isinstance(r, str) or not r.strip() for r in reports):
            raise ValueError("One nonempty report target is required per state")
        prefix = self.prefix(state)
        ids, valid = self.targets(reports, prefix.device)
        sequence = torch.cat([prefix, self.language.get_input_embeddings()(ids)], 1)
        mask = torch.cat([valid.new_ones(prefix.shape[:2]), valid], 1)
        positions = torch.arange(sequence.shape[1], device=sequence.device)[None].expand(len(state), -1)
        hidden = self.language(inputs_embeds=sequence, attention_mask=mask, position_ids=positions,
                               use_cache=False, return_dict=True).last_hidden_state
        return chunked_ce(hidden[:, prefix.shape[1] - 1:-1], ids.masked_fill(~valid.bool(), -100),
                          self.output_head.weight, chunk=self.ce_chunk_tokens)

    @torch.no_grad()
    def generate(self, state, max_new_tokens=384):
        if self.training:
            raise ValueError("Call model.eval() before text generation")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, Integral) or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        prefix = self.prefix(state)
        mask = torch.ones(prefix.shape[:2], device=prefix.device, dtype=torch.long)
        positions = torch.arange(prefix.shape[1], device=prefix.device)[None].expand(len(prefix), -1)
        output = self.language(inputs_embeds=prefix, attention_mask=mask, position_ids=positions,
                               use_cache=True, return_dict=True)
        ended = torch.zeros(len(prefix), device=prefix.device, dtype=torch.bool)
        generated = []
        for index in range(max_new_tokens):
            weight = self.output_head.weight
            token = F.linear(output.last_hidden_state[:, -1].to(weight.dtype), weight).argmax(-1)
            token = torch.where(ended, self.tokenizer.pad_token_id, token)
            generated.append(token)
            ended |= token == self.tokenizer.eos_token_id
            if ended.all() or index + 1 == max_new_tokens:
                break
            mask = torch.cat([mask, mask.new_ones((len(mask), 1))], 1)
            positions = (mask.sum(-1) - 1)[:, None]
            output = self.language(input_ids=token[:, None], attention_mask=mask, position_ids=positions,
                                   past_key_values=output.past_key_values, use_cache=True, return_dict=True)
        return self.tokenizer.batch_decode(torch.stack(generated, 1).cpu(), skip_special_tokens=True)


class ClassificationHead(nn.Module):
    def __init__(self, findings=13):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(STATE_WIDTH), nn.Linear(STATE_WIDTH, 128))
        self.queries = nn.Parameter(torch.randn(findings, 128) * .02)
        self.attention = nn.MultiheadAttention(128, 4, batch_first=True, dropout=0)
        self.output = nn.Linear(128, 1)

    def forward(self, state):
        values = self.project(state.float())
        queries = self.queries[None].expand(len(state), -1, -1)
        readout, _ = self.attention(queries, values, values, need_weights=False)
        return self.output(queries + readout).squeeze(-1)


def block(a, b, stride=1):
    return nn.Sequential(nn.Conv2d(a, b, 3, stride=stride, padding=1), nn.GroupNorm(4, b), nn.GELU())


def positional_encoding(height, width, channels=64):
    frequency = torch.exp(-math.log(10000) * torch.arange(channels // 4) / (channels // 4))
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    xx, yy = x.flatten()[:, None] * frequency, y.flatten()[:, None] * frequency
    return torch.cat([xx.sin(), xx.cos(), yy.sin(), yy.cos()], -1)[None]


class SpatialHead(nn.Module):
    def __init__(self, task):
        super().__init__()
        if task not in ("segmentation", "sr"):
            raise ValueError(task)
        self.task = task
        self.stem = nn.Sequential(block(1, 32, 2), block(32, 64, 2))
        self.slot_normalize = nn.LayerNorm(STATE_WIDTH, elementwise_affine=False)
        self.slot_project = nn.Sequential(nn.Linear(STATE_WIDTH, 64), nn.GELU())
        self.query_normalize = nn.LayerNorm(64)
        self.attention = nn.MultiheadAttention(64, 4, batch_first=True, dropout=0)
        self.register_buffer("image_position", positional_encoding(32, 32))
        self.register_buffer("depth_position", positional_encoding(1, 4))
        self.fuse = block(128, 64)
        layers, channels = [], 64
        for width in [48, 32, 16] + ([8] if task == "sr" else []):
            layers.extend([nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), block(channels, width)])
            channels = width
        layers.append(nn.Conv2d(channels, 1 if task == "sr" else 3, 1))
        self.decode = nn.Sequential(*layers)

    def forward(self, image, slots):
        if tuple(slots.shape[1:]) != (4, STATE_WIDTH):
            raise ValueError("Spatial tasks read exactly the four visual slots")
        z = F.interpolate(self.stem(image), (32, 32), mode="bilinear", align_corners=False)
        query = self.query_normalize(z.flatten(2).transpose(1, 2) + self.image_position)
        values = self.slot_project(self.slot_normalize(slots.float())) + self.depth_position
        attended, _ = self.attention(query, values, values, need_weights=False)
        attended = attended.transpose(1, 2).reshape(len(image), 64, 32, 32)
        prediction = self.decode(self.fuse(torch.cat([z, attended], 1)))
        if self.task == "sr":
            prediction = F.interpolate(image, scale_factor=4, mode="bicubic", align_corners=False) + .1 * prediction
        return prediction


def spatial_loss(task, prediction, target, mask):
    if task == "sr":
        return (((prediction.float() - target).square() * mask).sum((1, 2, 3)) /
                mask.sum((1, 2, 3)).clamp_min(1)).mean()
    prediction = prediction[:, :target.shape[1]].float()
    bce = F.binary_cross_entropy_with_logits(prediction, target, reduction="none")
    bce = ((bce * mask).sum((1, 2, 3)) / (mask.sum((1, 2, 3)) * target.shape[1]).clamp_min(1)).mean()
    p, t = prediction.sigmoid() * mask, (target > .5).float() * mask
    dice = (2 * (p * t).sum((2, 3)) + 1) / (p.sum((2, 3)) + t.sum((2, 3)) + 1)
    return bce + 1 - dice.mean()
