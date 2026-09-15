"""Forecast ablations with full-token memory or other-patient state donors.

The no-slots readouts retain every valid encoder token. Only the latent
alignment loss pools memories to eight bins, since source/future lengths differ.
"""
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from model import MedWorld, hidden, chunked_ce, masked_bce


@dataclass
class Memory:
    values: torch.Tensor
    mask: torch.Tensor


def pooled_memory(memory, bins=8):
    return torch.stack([F.adaptive_avg_pool1d(row[valid.bool()].T[None], bins)[0].T
                        for row, valid in zip(memory.values, memory.mask)])


class FullTokenEncoder(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.backbone = original.backbone
        self.adapter = original.adapter
        self.visual_position = original.visual_position

    def forward(self, features, text_ids, text_mask):
        visual = self.adapter(features.float()) + self.visual_position
        text = self.backbone.get_input_embeddings()(text_ids)
        seq = torch.cat([visual.to(text.dtype), text], 1)
        mask = torch.cat([torch.ones(visual.shape[:2], device=seq.device, dtype=torch.long), text_mask], 1)
        return Memory(hidden(self.backbone, seq, mask).last_hidden_state, mask)


class AblationMedWorld(MedWorld):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.condition = cfg.get('state_condition', 'slots')
        if self.condition not in ('slots', 'no_slots', 'shuffled'):
            raise ValueError(self.condition)
        if self.condition == 'no_slots':
            self.encoder = FullTokenEncoder(self.encoder)

    def state(self, batch, which='source', image_only=False, target=False):
        if self.condition == 'shuffled' and which == 'source' and not image_only:
            which = 'donor'
        return super().state(batch, which, image_only, target)

    def forecast(self, state, horizon):
        if not isinstance(state, Memory):
            return self.world(state, horizon)
        normed = F.layer_norm(state.values.float(), (state.values.shape[-1],))
        z = self.world.input(normed) + self.world.horizon(horizon)[:, None]
        # Continuous interpolation retains exactly the shared eight positional
        # parameter vectors; it does not introduce learned encoder state slots.
        # Assign positions by each sample's valid-token rank, so a longer peer
        # in the minibatch cannot change this sample through extra padding.
        table = self.world.positions[0]
        rank = (state.mask.cumsum(1)-1).clamp_min(0).float()
        coordinate = rank / (state.mask.sum(1)-1).clamp_min(1)[:, None] * (len(table)-1)
        lower = coordinate.long().clamp_max(len(table)-1)
        upper = (lower+1).clamp_max(len(table)-1)
        fraction = (coordinate-lower)[:, :, None]
        position = table[lower]*(1-fraction) + table[upper]*fraction
        z = self.world.transformer(z + position, src_key_padding_mask=~state.mask.bool())
        return Memory(normed + self.world.output(z), state.mask)

    def scores(self, state):
        if isinstance(state, Memory):
            values = (state.values.float() * state.mask[:, :, None]).sum(1) / state.mask.sum(1).clamp_min(1)[:, None]
            return self.finding(values)
        return super().scores(state)

    def prefix(self, memory):
        embed = self.decoder.backbone.get_input_embeddings()
        clinical = self.decoder.projection(memory.values.float()).to(embed.weight.dtype)
        clinical = clinical * memory.mask[:, :, None]
        prompt = embed(self.decoder.prompt_ids)[None].expand(len(clinical), -1, -1)
        return torch.cat([clinical, prompt], 1), torch.cat([memory.mask, torch.ones(prompt.shape[:2], device=clinical.device, dtype=torch.long)], 1)

    def report_loss(self, state, ids, target_mask):
        if not isinstance(state, Memory):
            prefix = self.decoder.prefix(state)
            mask = torch.ones(prefix.shape[:2], device=prefix.device, dtype=torch.long)
        else:
            prefix, mask = self.prefix(state)
        embed = self.decoder.backbone.get_input_embeddings()
        outputs = hidden(self.decoder.backbone, torch.cat([prefix, embed(ids)], 1), torch.cat([mask, target_mask], 1)).last_hidden_state
        return chunked_ce(outputs[:, prefix.shape[1]-1:-1], ids.masked_fill(~target_mask.bool(), -100), self.decoder.language_weight(),
                          chunk=self.cfg.get('ce_chunk', 32))

    def losses(self, batch, stage):
        if stage == 1:
            state = self.state(batch, image_only=True)
            ce = self.report_loss(state, batch['source_target_ids'], batch['source_target_mask'])
            bce = masked_bce(self.scores(state), batch['source_labels'])
            return ce + self.cfg['finding_weight']*bce, dict(text=ce, finding=bce)
        pred = self.forecast(self.state(batch), batch['horizon'])
        target = self.state(batch, 'target', target=True)
        p, t = (pooled_memory(pred), pooled_memory(target)) if isinstance(pred, Memory) else (pred, target)
        latent = F.mse_loss(F.layer_norm(p.float(), (p.shape[-1],)), F.layer_norm(t.float(), (t.shape[-1],)))
        ce = self.report_loss(pred, batch['target_target_ids'], batch['target_target_mask'])
        bce = masked_bce(self.scores(pred), batch['target_labels'])
        replay = p.new_zeros(())
        if self.cfg['replay_weight']:
            replay = masked_bce(self.scores(self.state(batch, image_only=True)), batch['source_labels'])
        total = self.cfg['latent_weight']*latent + self.cfg['text_weight']*ce + self.cfg['finding_weight']*bce + self.cfg['replay_weight']*replay
        return total, dict(latent=latent, text=ce, finding=bce, replay=replay)

    @torch.no_grad()
    def predict(self, batch):
        state = self.forecast(self.state(batch), batch['horizon'])
        if not isinstance(state, Memory):
            return self.decoder.generate(state, self.cfg['generation_tokens']), self.scores(state).sigmoid()
        prefix, mask = self.prefix(state)
        embed = self.decoder.backbone.get_input_embeddings()
        outputs = hidden(self.decoder.backbone, prefix, mask, use_cache=True)
        ended = torch.zeros(len(prefix), device=prefix.device, dtype=torch.bool)
        generated = []
        for _ in range(self.cfg['generation_tokens']):
            weight = self.decoder.language_weight()
            ids = F.linear(outputs.last_hidden_state[:, -1].to(weight.dtype), weight).argmax(-1)
            ids = torch.where(ended, self.tokenizer.eos_token_id, ids)
            generated.append(ids)
            ended |= ids == self.tokenizer.eos_token_id
            if ended.all():
                break
            mask = torch.cat([mask, torch.ones_like(mask[:, :1])], 1)
            outputs = hidden(self.decoder.backbone, embed(ids[:, None]), mask, use_cache=True, past=outputs.past_key_values)
        reports = self.tokenizer.batch_decode(torch.stack(generated, 1).cpu(), skip_special_tokens=True)
        return reports, self.scores(state).sigmoid()


def build_model(cfg):
    return AblationMedWorld(cfg) if cfg.get('state_condition') else MedWorld(cfg)
