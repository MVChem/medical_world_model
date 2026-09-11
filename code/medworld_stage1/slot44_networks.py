"""2026-09-11: clinical/spatial slot routing and query-conditioned disease lists."""
import torch
from torch import nn
import torch.nn.functional as F
from networks import FourTaskModel, masked_classification, segmentation_loss
from model import hidden, chunked_ce


class DiseaseListDecoder(nn.Module):
    def __init__(self, backbone, tokenizer, width=1024):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.projection = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))

    def prefix(self, state, query_ids, query_mask):
        embed = self.backbone.get_input_embeddings()
        slots = self.projection(state.float()).to(embed.weight.dtype)
        prefix = torch.cat([slots, embed(query_ids)], 1)
        mask = torch.cat([torch.ones(slots.shape[:2], device=state.device, dtype=torch.long), query_mask], 1)
        return prefix, mask

    def loss(self, state, query_ids, query_mask, target_ids, target_mask):
        embed = self.backbone.get_input_embeddings()
        prefix, mask = self.prefix(state, query_ids, query_mask)
        seq = torch.cat([prefix, embed(target_ids)], 1)
        mask = torch.cat([mask, target_mask], 1)
        outputs = hidden(self.backbone, seq, mask).last_hidden_state
        predictors = outputs[:, prefix.shape[1]-1:-1]
        targets = target_ids.masked_fill(~target_mask.bool(), -100)
        return chunked_ce(predictors, targets, embed.weight)

    @torch.no_grad()
    def generate(self, state, query_ids, query_mask, max_tokens):
        embed = self.backbone.get_input_embeddings()
        prefix, mask = self.prefix(state, query_ids, query_mask)
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
        texts = self.tokenizer.batch_decode(torch.stack(generated, 1).cpu(), skip_special_tokens=True)
        return texts, (~ended).cpu().tolist()


class SpatialRead(nn.Module):
    """Each image location reads the four spatial tokens independently."""
    def __init__(self, size, width=1024, dim=64):
        super().__init__()
        self.size = size
        self.slot = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, dim))
        self.query = nn.Conv2d(32, dim, 1)
        self.position = nn.Linear(2, dim)
        self.attention = nn.MultiheadAttention(dim, 4, dropout=0, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Conv2d(dim, 32, 1)

    def forward(self, x, slots):
        assert slots.shape[1] == 4
        size = min(self.size, *x.shape[-2:])
        pooled = F.adaptive_avg_pool2d(x, (size, size))
        axis = torch.linspace(-1, 1, size, device=x.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        xy = torch.stack([xx, yy], -1).reshape(1, size*size, 2)
        q = self.query(pooled).flatten(2).transpose(1, 2) + self.position(xy)
        kv = self.slot(slots.float())
        z, _ = self.attention(q, kv, kv, need_weights=False)
        z = self.norm(q + z).transpose(1, 2).reshape(len(x), 64, size, size)
        return x + .1*F.interpolate(self.output(z), x.shape[-2:], mode='bilinear', align_corners=False)


class SpatialSuperResolution(nn.Module):
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale
        self.input = nn.Conv2d(1, 32, 3, padding=1)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
                                                  nn.Conv2d(32, 32, 3, padding=1)) for _ in range(6)])
        self.spatial = nn.ModuleList([SpatialRead(64), SpatialRead(32)])
        self.out = nn.Sequential(nn.Conv2d(32, scale*scale, 3, padding=1), nn.PixelShuffle(scale))

    def forward(self, lr, s):
        x = self.input(lr)
        for i, block in enumerate(self.blocks):
            x = x + .1*block(x)
            if i in (1, 3):
                x = self.spatial[(i-1)//2](x, s)
        base = F.interpolate(lr, scale_factor=self.scale, mode='bicubic', align_corners=False)
        return base + .1*self.out(x)


class Slot44Model(FourTaskModel):
    def __init__(self, cfg, freeze_encoder=False):
        if freeze_encoder:
            raise ValueError('This run has one joint model and no frozen-encoder ablation')
        super().__init__(cfg)
        self.diagnosis = DiseaseListDecoder(self.diagnosis.backbone, self.tokenizer)
        self.sr = SpatialSuperResolution(cfg['scale'])
        self.last_state = None

    @staticmethod
    def route(s, task):
        if task == 'classification':
            return s[:, :4]
        if task in ('segmentation', 'sr'):
            return s[:, 4:]
        if task == 'diagnosis':
            return s
        raise ValueError(task)

    def loss(self, b, task, pos_weight):
        full = self.encode(b, task)
        if torch.is_grad_enabled() and full.requires_grad:
            full.retain_grad()
            self.last_state = full
        s = self.route(full, task)
        if task == 'classification':
            return masked_classification(self.classification(s), b['labels'], pos_weight)
        if task == 'diagnosis':
            return self.diagnosis.loss(s, b['query_ids'], b['query_mask'], b['target_ids'], b['target_mask'])
        if task == 'segmentation':
            pred, coarse = self.segmentation(b['seg_image'], s)
            return segmentation_loss(pred, b['seg_probs'], b['seg_mask']) + .2*segmentation_loss(
                coarse, F.interpolate(b['seg_probs'], (32,32), mode='area'),
                F.interpolate(b['seg_mask'], (32,32), mode='area'))
        pred = self.sr(b['lr'], s)
        return ((pred.float()-b['hr']).square()*b['valid']).sum()/b['valid'].sum().clamp_min(1)
