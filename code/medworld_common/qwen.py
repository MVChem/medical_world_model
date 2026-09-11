"""Shared Qwen loading, adaptation, state encoding and text readout."""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from peft import LoraConfig, get_peft_model
from transformers import Qwen3_5ForConditionalGeneration

LORA_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
]


def load_qwen(cfg):
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        cfg["qwen"],
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.requires_grad_(False)
    return model


def adapt(text_model, cfg):
    return get_peft_model(
        text_model,
        LoraConfig(
            r=cfg["lora_rank"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=0.0,
            target_modules=LORA_MODULES,
            bias="none",
        ),
    )


def hidden(model, embeds, mask, *, use_cache=False, past=None):
    positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
    if past is not None:
        positions = positions[:, -embeds.shape[1] :]
    return model(
        inputs_embeds=embeds,
        attention_mask=mask,
        position_ids=positions,
        use_cache=use_cache,
        past_key_values=past,
        return_dict=True,
    )


def masked_bce(logits, labels):
    valid = (labels == 0) | (labels == 1)
    losses = F.binary_cross_entropy_with_logits(
        logits.float(), labels.clamp(0, 1).float(), reduction="none"
    )
    return (losses * valid).sum() / valid.sum().clamp_min(1)


def chunked_ce(states, targets, weight, chunk=32):
    """Exact vocabulary CE without retaining full sequence x 248K logits."""
    valid = targets != -100
    states, targets = states[valid], targets[valid]
    if len(targets) == 0:
        return states.sum() * 0
    total = states.new_zeros((), dtype=torch.float32)

    def loss_fn(h, y):
        return F.cross_entropy(
            F.linear(h.to(weight.dtype), weight).float(), y, reduction="sum"
        )

    for i in range(0, len(targets), chunk):
        h, y = states[i : i + chunk], targets[i : i + chunk]
        total = total + (
            checkpoint(loss_fn, h, y, use_reentrant=False)
            if torch.is_grad_enabled()
            else loss_fn(h, y)
        )
    return total / len(targets)


class StateEncoder(nn.Module):
    def __init__(self, backbone, cfg, width):
        super().__init__()
        self.backbone = backbone
        self.adapter = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, width), nn.GELU(), nn.Linear(width, width)
        )
        self.slots = nn.Parameter(torch.randn(cfg["slots"], width) * 0.02)
        self.visual_position = nn.Parameter(
            torch.randn(1, cfg["visual_grid"] ** 2, width) * 0.02
        )

    def forward(self, features, text_ids, text_mask):
        embedding = self.backbone.get_input_embeddings()
        visual = self.adapter(features.float()) + self.visual_position
        text = embedding(text_ids)
        slots = self.slots[None].expand(len(features), -1, -1)
        seq = torch.cat([visual.to(text.dtype), text, slots.to(text.dtype)], 1)
        mask = torch.cat(
            [
                torch.ones(visual.shape[:2], device=seq.device, dtype=torch.long),
                text_mask,
                torch.ones(slots.shape[:2], device=seq.device, dtype=torch.long),
            ],
            1,
        )
        return hidden(self.backbone, seq, mask).last_hidden_state[:, -slots.shape[1] :]


class ReportDecoder(nn.Module):
    def __init__(self, backbone, tokenizer, width):
        super().__init__()
        self.backbone = backbone
        self.projection = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))
        self.tokenizer = tokenizer
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": "Write a concise chest radiograph report with FINDINGS and IMPRESSION. Use the supplied clinical state.",
                },
                {"role": "user", "content": "Describe the supplied clinical state."},
            ],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        self.register_buffer(
            "prompt_ids", torch.tensor(prompt, dtype=torch.long), persistent=False
        )

    def prefix(self, state):
        embed = self.backbone.get_input_embeddings()
        clinical = self.projection(state.float()).to(embed.weight.dtype)
        prompt = embed(self.prompt_ids)[None].expand(len(state), -1, -1)
        return torch.cat([clinical, prompt], 1)

    def loss(self, state, target_ids, target_mask):
        embed = self.backbone.get_input_embeddings()
        prefix = self.prefix(state)
        seq = torch.cat([prefix, embed(target_ids)], 1)
        mask = torch.cat(
            [
                torch.ones(prefix.shape[:2], device=state.device, dtype=torch.long),
                target_mask,
            ],
            1,
        )
        outputs = hidden(self.backbone, seq, mask).last_hidden_state
        predictors = outputs[:, prefix.shape[1] - 1 : -1]
        targets = target_ids.masked_fill(~target_mask.bool(), -100)
        return chunked_ce(predictors, targets, embed.weight)

    @torch.no_grad()
    def generate(self, state, max_tokens):
        embed = self.backbone.get_input_embeddings()
        prefix = self.prefix(state)
        mask = torch.ones(prefix.shape[:2], device=state.device, dtype=torch.long)
        outputs = hidden(self.backbone, prefix, mask, use_cache=True)
        ended = torch.zeros(len(state), device=state.device, dtype=torch.bool)
        generated = []
        for _ in range(max_tokens):
            logits = F.linear(
                outputs.last_hidden_state[:, -1].to(embed.weight.dtype), embed.weight
            )
            ids = logits.argmax(-1)
            ids = torch.where(ended, self.tokenizer.eos_token_id, ids)
            generated.append(ids)
            ended |= ids == self.tokenizer.eos_token_id
            if ended.all():
                break
            mask = torch.cat([mask, torch.ones_like(mask[:, :1])], 1)
            outputs = hidden(
                self.backbone,
                embed(ids[:, None]),
                mask,
                use_cache=True,
                past=outputs.past_key_values,
            )
        return self.tokenizer.batch_decode(
            torch.stack(generated, 1).cpu(), skip_special_tokens=True
        )
