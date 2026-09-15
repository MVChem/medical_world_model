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


def copy_with_shared_frozen_parameters(module):
    """Clone module state and adapters while reusing immutable base Parameters.

    Modules and buffers remain independent: checkpointing flags and any cached
    rotary state must not leak between the online, decoder, and target copies.
    Reusing Parameter objects, rather than only their storage, also preserves
    the aliases when the enclosing model is moved to a device with ``to``.
    """
    memo = {id(p): p for p in module.parameters() if not p.requires_grad}
    cloned = copy.deepcopy(module, memo)
    audit_frozen_parameter_copy(module, cloned)
    return cloned


def audit_frozen_parameter_copy(original, cloned):
    source = dict(original.named_parameters())
    target = dict(cloned.named_parameters())
    if source.keys() != target.keys():
        raise RuntimeError('Shared-base copies have different parameter names')
    shared, independent = 0, 0
    for name, parameter in source.items():
        other = target[name]
        if parameter.requires_grad:
            if parameter is other or parameter.data_ptr() == other.data_ptr():
                raise RuntimeError(f'Trainable parameter aliases its copy: {name}')
            independent += parameter.numel()
        else:
            if parameter is not other:
                raise RuntimeError(f'Frozen parameter was unnecessarily copied: {name}')
            shared += parameter.numel()
    return dict(shared_frozen_parameters=shared, independent_trainable_parameters=independent)


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
        copier = copy_with_shared_frozen_parameters if cfg.get('share_frozen_backbone', False) else copy.deepcopy
        decoder_base = copier(text_model)
        # Qwen3.5-9B has untied input/output embeddings. Retain its pretrained
        # language head; the tied 0.8B path keeps its existing checkpoint layout.
        output_head = base.lm_head if not base.config.tie_word_embeddings else None
        del base
        self.encoder = StateEncoder(adapt(text_model, cfg), cfg, width)
        self.decoder = ReportDecoder(adapt(decoder_base, cfg), self.tokenizer, width, output_head=output_head)
        if cfg.get('share_frozen_backbone', False):
            self.frozen_backbone_audit = audit_frozen_parameter_copy(self.encoder.backbone, self.decoder.backbone)
        self.world = WorldModel(cfg, width)
        self.finding = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, len(cfg['findings'])))
        self.target_encoder = None
        for b in (self.encoder.backbone, self.decoder.backbone):
            b.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    def begin_stage2(self, frozen_encoder=False):
        copier = copy_with_shared_frozen_parameters if self.cfg.get('share_frozen_backbone', False) else copy.deepcopy
        self.target_encoder = copier(self.encoder).requires_grad_(False).eval()
        self.target_encoder.backbone.gradient_checkpointing_disable()
        self.stage2_frozen_encoder = frozen_encoder
        if frozen_encoder:
            self.encoder.requires_grad_(False).eval()

    def audit_shared_backbones(self):
        """Validate aliases on their current devices, including after ``to``."""
        if not self.cfg.get('share_frozen_backbone', False):
            return dict(enabled=False)
        source = dict(self.encoder.backbone.named_parameters())
        peers = {'decoder': self.decoder.backbone}
        if self.target_encoder is not None:
            peers['target'] = self.target_encoder.backbone
        results = {}
        for label, backbone in peers.items():
            target = dict(backbone.named_parameters())
            if source.keys() != target.keys():
                raise RuntimeError(f'{label} backbone parameter names differ')
            shared, independent = 0, 0
            for name, parameter in source.items():
                other = target[name]
                if 'lora_' in name or parameter.requires_grad:
                    if parameter is other or parameter.data_ptr() == other.data_ptr():
                        raise RuntimeError(f'{label} adapter aliases online parameter: {name}')
                    independent += parameter.numel()
                else:
                    if parameter is not other or parameter.data_ptr() != other.data_ptr():
                        raise RuntimeError(f'{label} frozen base storage is duplicated: {name}')
                    shared += parameter.numel()
            results[label] = dict(shared_frozen_parameters=shared, independent_adapter_parameters=independent)
        online_lora = [p for name, p in source.items() if 'lora_' in name]
        if not getattr(self, 'stage2_frozen_encoder', False) and not all(p.requires_grad for p in online_lora):
            raise RuntimeError('Freezing the target also froze an online adapter')
        target_frozen = self.target_encoder is None or all(not p.requires_grad for p in self.target_encoder.parameters())
        if not target_frozen:
            raise RuntimeError('Target encoder has trainable parameters')
        return dict(enabled=True, copies=results, target_frozen=target_frozen,
                    online_lora_trainable=all(p.requires_grad for p in online_lora),
                    devices=sorted({str(p.device) for p in source.values()}))

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
                if self._compact_key(k)}

    @staticmethod
    def _compact_key(key):
        return not key.startswith('decoder.output_head.') and ('lora_' in key or '.backbone.' not in key)

    def load_compact(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected or any(self._compact_key(k) for k in missing):
            raise ValueError((missing, unexpected))
