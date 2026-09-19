"""Finding classification: eight-slot readout, masked BCE and AUROC/AP."""
import torch
from torch import nn
import torch.nn.functional as F
from .. import STATE_WIDTH

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

def finding_loss(logits, labels, mask=None, pos_weight=None):
    labels = labels.to(device=logits.device, dtype=torch.float32)
    mask = ((labels == 0) | (labels == 1)) if mask is None else mask.to(logits.device).bool()
    loss = F.binary_cross_entropy_with_logits(logits.float(), labels.clamp(0, 1),
                                            pos_weight=pos_weight, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1)

def classification_metrics(labels, probabilities, names):
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    per_finding = {}
    for j, name in enumerate(names):
        valid = (labels[:, j] == 0) | (labels[:, j] == 1)
        target, score = labels[valid, j], probabilities[valid, j]
        evaluable = len(np.unique(target)) == 2
        per_finding[name] = {"n": int(valid.sum()),
                             "auroc": float(roc_auc_score(target, score)) if evaluable else None,
                             "ap": float(average_precision_score(target, score)) if evaluable else None}
    def mean(key):
        values = [r[key] for r in per_finding.values() if r[key] is not None]
        return float(np.mean(values)) if values else None
    return {"macro_auroc": mean("auroc"), "macro_ap": mean("ap"), "per_finding": per_finding}
