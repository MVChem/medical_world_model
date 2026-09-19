"""Finding classification: eight-slot readout, masked BCE and AUROC/AP."""
import torch
from torch import nn
import torch.nn.functional as F
from ... import STATE_WIDTH

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
