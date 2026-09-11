"""Frozen four-state CheXbert labels, preserving blank and uncertain.

Architecture/order follows stanfordmlgroup/CheXbert and StanfordAIMI's public
RRG_scorers checkpoint. Do not use the package's binary uncertain-positive map.
"""
from common import ROOT, digest
from importlib.metadata import version
import torch
from torch import nn
from transformers import BertModel, BertConfig, BertTokenizer

CHEXBERT_NAMES = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema',
    'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
    'Fracture', 'Support Devices', 'No Finding']


class CheXbert(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        path = ROOT / 'weights/bert-base-uncased'
        self.bert = BertModel(BertConfig.from_pretrained(path, local_files_only=True))
        self.linear_heads = nn.ModuleList([nn.Linear(768, 4) for _ in range(13)] + [nn.Linear(768, 2)])
        self.tokenizer = BertTokenizer.from_pretrained(path, local_files_only=True)
        weight = ROOT / 'weights/chexbert.pth'
        state = torch.load(weight, map_location='cpu', weights_only=True)['model_state_dict']
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        self.load_state_dict(state, strict=True)
        self.requires_grad_(False).eval().to(device)
        self.device = device
        self.provenance = dict(checkpoint='StanfordAIMI/RRG_scorers/chexbert.pth', sha256=digest(weight),
                               labels='0 blank -> -2; 1 positive -> 1; 2 negative -> 0; 3 uncertain -> -1',
                               tokenizer='bert-base-uncased', max_tokens=512, transformers_version=version('transformers'))

    @torch.no_grad()
    def labels(self, texts, findings, batch_size=32):
        indices = [CHEXBERT_NAMES.index(k) for k in findings]
        output = []
        for start in range(0, len(texts), batch_size):
            normalized = [' '.join(t.split()) for t in texts[start:start+batch_size]]
            batch = self.tokenizer(normalized, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
            cls = self.bert(**batch).last_hidden_state[:, 0]
            raw = torch.stack([self.linear_heads[i](cls).argmax(-1) for i in indices], 1)
            mapping = torch.tensor([-2, 1, 0, -1], device=self.device)
            output.extend(mapping[raw].cpu().tolist())
        return output
