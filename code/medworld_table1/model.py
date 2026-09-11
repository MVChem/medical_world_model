"""Small MedWorld-JEPA pilot: online multimodal slots, fixed target, latent-only readouts."""
import copy
import contextlib

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

LORA_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'in_proj_qkv', 'in_proj_z',
                'in_proj_b', 'in_proj_a', 'out_proj']
HORIZON_NAMES = ['6 to 24 hours', '1 to 3 days', '3 to 7 days', '7 to 30 days']


def load_qwen(cfg):
    model = Qwen3_5ForConditionalGeneration.from_pretrained(cfg['qwen'], local_files_only=True,
                                                          dtype=torch.bfloat16, attn_implementation='sdpa')
    model.requires_grad_(False)
    return model


def adapt(text_model, cfg):
    return get_peft_model(text_model, LoraConfig(r=cfg['lora_rank'], lora_alpha=cfg['lora_alpha'],
                          lora_dropout=0., target_modules=LORA_MODULES, bias='none'))


def hidden(model, embeds, mask, *, use_cache=False, past=None):
    positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
    if past is not None:
        positions = positions[:, -embeds.shape[1]:]
    return model(inputs_embeds=embeds, attention_mask=mask, position_ids=positions,
                 use_cache=use_cache, past_key_values=past, return_dict=True)


def masked_bce(logits, labels):
    valid = (labels == 0) | (labels == 1)
    losses = F.binary_cross_entropy_with_logits(logits.float(), labels.clamp(0, 1).float(), reduction='none')
    return (losses * valid).sum() / valid.sum().clamp_min(1)


def chunked_ce(states, targets, weight, chunk=32):
    """Exact vocabulary CE without retaining full sequence x 248K logits."""
    valid = targets != -100
    states, targets = states[valid], targets[valid]
    if len(targets) == 0:
        return states.sum() * 0
    total = states.new_zeros((), dtype=torch.float32)
    def loss_fn(h, y):
        return F.cross_entropy(F.linear(h.to(weight.dtype), weight).float(), y, reduction='sum')
    for i in range(0, len(targets), chunk):
        h, y = states[i:i+chunk], targets[i:i+chunk]
        total = total + (checkpoint(loss_fn, h, y, use_reentrant=False) if torch.is_grad_enabled() else loss_fn(h, y))
    return total / len(targets)


class StateEncoder(nn.Module):
    def __init__(self, backbone, cfg, width):
        super().__init__()
        self.backbone = backbone
        self.adapter = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, width), nn.GELU(), nn.Linear(width, width))
        self.slots = nn.Parameter(torch.randn(cfg['slots'], width) * .02)
        self.visual_position = nn.Parameter(torch.randn(1, cfg['visual_grid']**2, width) * .02)

    def forward(self, features, text_ids, text_mask):
        embedding = self.backbone.get_input_embeddings()
        visual = self.adapter(features.float()) + self.visual_position
        text = embedding(text_ids)
        slots = self.slots[None].expand(len(features), -1, -1)
        seq = torch.cat([visual.to(text.dtype), text, slots.to(text.dtype)], 1)
        mask = torch.cat([torch.ones(visual.shape[:2], device=seq.device, dtype=torch.long), text_mask,
                          torch.ones(slots.shape[:2], device=seq.device, dtype=torch.long)], 1)
        return hidden(self.backbone, seq, mask).last_hidden_state[:, -slots.shape[1]:]


class WorldModel(nn.Module):
    def __init__(self, cfg, width):
        super().__init__()
        d = cfg['lwm_width']
        self.input = nn.Linear(width, d)
        self.horizon = nn.Embedding(len(HORIZON_NAMES), d)
        self.positions = nn.Parameter(torch.randn(1, cfg['slots'], d) * .02)
        layer = nn.TransformerEncoderLayer(d, 8, 4*d, dropout=0., activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, cfg['lwm_depth'], enable_nested_tensor=False)
        self.output = nn.Linear(d, width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, state, horizon):
        normed = F.layer_norm(state.float(), (state.shape[-1],))
        z = self.input(normed) + self.horizon(horizon)[:, None] + self.positions
        return normed + self.output(self.transformer(z))


class ReportDecoder(nn.Module):
    def __init__(self, backbone, tokenizer, width):
        super().__init__()
        self.backbone = backbone
        self.projection = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))
        self.tokenizer = tokenizer
        prompt = tokenizer.apply_chat_template([
            {'role': 'system', 'content': 'Write a concise chest radiograph report with FINDINGS and IMPRESSION. Use the supplied clinical state.'},
            {'role': 'user', 'content': 'Describe the supplied clinical state.'}
        ], tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
        self.register_buffer('prompt_ids', torch.tensor(prompt, dtype=torch.long), persistent=False)

    def prefix(self, state):
        embed = self.backbone.get_input_embeddings()
        clinical = self.projection(state.float()).to(embed.weight.dtype)
        prompt = embed(self.prompt_ids)[None].expand(len(state), -1, -1)
        return torch.cat([clinical, prompt], 1)

    def loss(self, state, target_ids, target_mask):
        embed = self.backbone.get_input_embeddings()
        prefix = self.prefix(state)
        seq = torch.cat([prefix, embed(target_ids)], 1)
        mask = torch.cat([torch.ones(prefix.shape[:2], device=state.device, dtype=torch.long), target_mask], 1)
        outputs = hidden(self.backbone, seq, mask).last_hidden_state
        predictors = outputs[:, prefix.shape[1]-1:-1]
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
            logits = F.linear(outputs.last_hidden_state[:, -1].to(embed.weight.dtype), embed.weight)
            ids = logits.argmax(-1)
            ids = torch.where(ended, self.tokenizer.eos_token_id, ids)
            generated.append(ids)
            ended |= ids == self.tokenizer.eos_token_id
            if ended.all():
                break
            mask = torch.cat([mask, torch.ones_like(mask[:, :1])], 1)
            outputs = hidden(self.backbone, embed(ids[:, None]), mask, use_cache=True, past=outputs.past_key_values)
        return self.tokenizer.batch_decode(torch.stack(generated, 1).cpu(), skip_special_tokens=True)


class MedWorld(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg['qwen'], local_files_only=True)
        base = load_qwen(cfg)
        width = base.config.text_config.hidden_size
        text_model = base.model.language_model
        # Distinct encoder/decoder LoRA parameters; pretrained tensors are frozen.
        decoder_base = copy.deepcopy(text_model)
        del base
        self.encoder = StateEncoder(adapt(text_model, cfg), cfg, width)
        self.decoder = ReportDecoder(adapt(decoder_base, cfg), self.tokenizer, width)
        self.world = WorldModel(cfg, width)
        self.finding = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, len(cfg['findings'])))
        self.target_encoder = None
        for b in (self.encoder.backbone, self.decoder.backbone):
            b.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    def begin_stage2(self, frozen_encoder=False):
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False).eval()
        self.target_encoder.backbone.gradient_checkpointing_disable()
        if frozen_encoder:
            self.encoder.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        if self.target_encoder is not None:
            self.target_encoder.eval()
        if not any(p.requires_grad for p in self.encoder.parameters()):
            self.encoder.eval()
        return self

    def scores(self, state):
        return self.finding(state.float().mean(1))

    def state(self, batch, which='source', image_only=False, target=False):
        encoder = self.target_encoder if target else self.encoder
        ids, mask = batch[which+'_ids'], batch[which+'_mask']
        if image_only:
            ids, mask = ids[:, :0], mask[:, :0]
        with torch.no_grad() if target else contextlib.nullcontext():
            return encoder(batch[which+'_features'], ids, mask)

    def losses(self, batch, stage):
        if stage == 1:
            state = self.state(batch, image_only=True)
            ce = self.decoder.loss(state, batch['source_target_ids'], batch['source_target_mask'])
            bce = masked_bce(self.scores(state), batch['source_labels'])
            return ce + self.cfg['finding_weight'] * bce, dict(text=ce, finding=bce)
        state = self.state(batch)
        pred = self.world(state, batch['horizon'])
        target = self.state(batch, 'target', target=True)
        latent = F.mse_loss(F.layer_norm(pred.float(), (pred.shape[-1],)), F.layer_norm(target.float(), (target.shape[-1],)))
        ce = self.decoder.loss(pred, batch['target_target_ids'], batch['target_target_mask'])
        bce = masked_bce(self.scores(pred), batch['target_labels'])
        # Replay image-only classification so report-derived answers are withheld.
        replay = pred.new_zeros(())
        if self.cfg['replay_weight']:
            current_image = self.state(batch, image_only=True)
            replay = masked_bce(self.scores(current_image), batch['source_labels'])
        total = self.cfg['latent_weight']*latent + self.cfg['text_weight']*ce + self.cfg['finding_weight']*bce + self.cfg['replay_weight']*replay
        return total, dict(latent=latent, text=ce, finding=bce, replay=replay)

    @torch.no_grad()
    def predict(self, batch):
        state = self.world(self.state(batch), batch['horizon'])
        return self.decoder.generate(state, self.cfg['generation_tokens']), self.scores(state).sigmoid()

    def compact_state(self):
        # Save only adapters and custom modules; fixed base model is pinned separately.
        return {k: v.detach().cpu() for k, v in self.state_dict().items()
                if 'lora_' in k or not ('.backbone.' in k)}

    def load_compact(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected or any('lora_' in k or '.backbone.' not in k for k in missing):
            raise ValueError((missing, unexpected))
