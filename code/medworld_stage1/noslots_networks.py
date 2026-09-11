"""Matched Qwen-0.8B baseline: full multimodal tokens, no learned state slots."""
import copy
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import load_qwen, adapt, hidden
from networks import SlotQueries, Classification, Segmentation, masked_classification, segmentation_loss
from slot44_networks import DiseaseListDecoder, SpatialRead, SpatialSuperResolution
from bootstrap import FINDINGS, digest


@dataclass
class TokenMemory:
    values: torch.Tensor
    mask: torch.Tensor

    def __len__(self):
        return len(self.values)

    @property
    def device(self):
        return self.values.device


class FullTokenEncoder(nn.Module):
    def __init__(self, backbone, cfg, width):
        super().__init__()
        self.backbone = backbone
        self.adapter = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, width), nn.GELU(), nn.Linear(width, width))
        self.visual_position = nn.Parameter(torch.randn(1, cfg['visual_grid']**2, width)*.02)

    def forward(self, features, text_ids, text_mask):
        embedding = self.backbone.get_input_embeddings()
        visual = self.adapter(features.float()) + self.visual_position
        text = embedding(text_ids)
        seq = torch.cat([visual.to(text.dtype), text], 1)
        mask = torch.cat([torch.ones(visual.shape[:2], device=seq.device, dtype=torch.long), text_mask], 1)
        encoded = hidden(self.backbone, seq, mask).last_hidden_state
        assert encoded.shape[1] == features.shape[1] + text_ids.shape[1]
        return TokenMemory(encoded, mask)


class MaskedQueries(SlotQueries):
    def forward(self, memory):
        z = self.project(self.norm(memory.values.float()))
        q = self.queries[None].expand(len(memory), -1, -1)
        out, _ = self.attn(q, z, z, key_padding_mask=~memory.mask.bool(), need_weights=False)
        return self.normout(q + out)


class FullTokenDiseaseDecoder(DiseaseListDecoder):
    def prefix(self, memory, query_ids, query_mask):
        embed = self.backbone.get_input_embeddings()
        context = self.projection(memory.values.float()).to(embed.weight.dtype)
        context = context * memory.mask[:,:,None]
        prefix = torch.cat([context, embed(query_ids)], 1)
        mask = torch.cat([memory.mask, query_mask], 1)
        return prefix, mask


class FullTokenSpatialRead(SpatialRead):
    def forward(self, x, memory):
        size = min(self.size, *x.shape[-2:])
        pooled = F.adaptive_avg_pool2d(x, (size, size))
        axis = torch.linspace(-1, 1, size, device=x.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        xy = torch.stack([xx, yy], -1).reshape(1, size*size, 2)
        q = self.query(pooled).flatten(2).transpose(1, 2) + self.position(xy)
        kv = self.slot(memory.values.float())
        z, _ = self.attention(q, kv, kv, key_padding_mask=~memory.mask.bool(), need_weights=False)
        z = self.norm(q + z).transpose(1, 2).reshape(len(x), 64, size, size)
        return x + .1*F.interpolate(self.output(z), x.shape[-2:], mode='bilinear', align_corners=False)


class NoSlotsModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        assert cfg['slots'] == 0
        self.tokenizer = AutoTokenizer.from_pretrained(cfg['qwen'], local_files_only=True)
        base = load_qwen(cfg)
        width = base.config.text_config.hidden_size
        assert width == 1024
        enc = base.model.language_model
        dec = copy.deepcopy(enc)
        del base
        self.encoder = FullTokenEncoder(adapt(enc, cfg), cfg, width)
        self.diagnosis = FullTokenDiseaseDecoder(adapt(dec, cfg), self.tokenizer, width)
        self.classification = Classification(width)
        self.classification.read = MaskedQueries(width, len(FINDINGS))
        self.segmentation = Segmentation(width)
        self.segmentation.read = MaskedQueries(width, 3, 64)
        self.sr = SpatialSuperResolution(cfg['scale'])
        self.sr.spatial = nn.ModuleList([FullTokenSpatialRead(64), FullTokenSpatialRead(32)])
        self.last_memory = None
        for b in [self.encoder.backbone, self.diagnosis.backbone]:
            b.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        assert 'encoder.slots' not in dict(self.named_parameters())

    def initialize_matched(self):
        path = Path(self.cfg['reference_initial_checkpoint'])
        initial = torch.load(path, map_location='cpu', weights_only=False)
        assert initial['step'] == 0 and not initial['optimizer']['state'], 'Never initialize baseline from a trained reference'
        state = dict(initial['model'])
        removed = state.pop('encoder.slots')
        assert removed.shape == (8, 1024)
        self.load_compact(state)
        actual = self.compact_state()
        assert actual.keys() == state.keys()
        assert all(torch.equal(actual[k], state[k]) for k in state)
        return dict(reference_step=0,reference_sha256=digest(path),all_common_parameters_identical=True,
                    removed_parameter='encoder.slots',removed_trainable_parameters=removed.numel(),
                    trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad))

    def encode(self, b, task):
        features = b['lr_features'] if task=='sr' else b['hr_features']
        return self.encoder(features, b['ids'], b['text_mask'])

    @staticmethod
    def route(memory, task):
        return memory

    def loss(self, b, task, pos_weight):
        memory = self.encode(b, task)
        if torch.is_grad_enabled() and memory.values.requires_grad:
            memory.values.retain_grad()
            self.last_memory = memory
        if task=='classification':
            return masked_classification(self.classification(memory), b['labels'], pos_weight)
        if task=='diagnosis':
            return self.diagnosis.loss(memory, b['query_ids'], b['query_mask'], b['target_ids'], b['target_mask'])
        if task=='segmentation':
            pred, coarse = self.segmentation(b['seg_image'], memory)
            return segmentation_loss(pred, b['seg_probs'], b['seg_mask']) + .2*segmentation_loss(
                coarse, F.interpolate(b['seg_probs'], (32,32), mode='area'),
                F.interpolate(b['seg_mask'], (32,32), mode='area'))
        if task!='sr':
            raise ValueError(task)
        pred = self.sr(b['lr'], memory)
        return ((pred.float()-b['hr']).square()*b['valid']).sum()/b['valid'].sum().clamp_min(1)

    def compact_state(self):
        return {k:v.detach().cpu() for k,v in self.state_dict().items() if 'lora_' in k or '.backbone.' not in k}

    def load_compact(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        assert not unexpected and not any('lora_' in k or '.backbone.' not in k for k in missing), (missing, unexpected)
