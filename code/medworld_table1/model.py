"""Small MedWorld-JEPA pilot: online multimodal slots, fixed target, latent-only readouts."""
import copy
import contextlib

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import common  # Adds the local shared package for direct script execution.
from medworld_common.qwen import (load_qwen, adapt, hidden, masked_bce, chunked_ce, StateEncoder, ReportDecoder)

HORIZON_NAMES = ['6 to 24 hours', '1 to 3 days', '3 to 7 days', '7 to 30 days']


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
