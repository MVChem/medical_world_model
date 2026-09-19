"""Finding classification: eight-slot readout, masked BCE and AUROC/AP."""
import torch
from torch import nn
import torch.nn.functional as F
from ... import STATE_WIDTH

def finding_loss(logits, labels, mask=None, pos_weight=None):
    labels = labels.to(device=logits.device, dtype=torch.float32)
    mask = ((labels == 0) | (labels == 1)) if mask is None else mask.to(logits.device).bool()
    loss = F.binary_cross_entropy_with_logits(logits.float(), labels.clamp(0, 1),
                                            pos_weight=pos_weight, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1)
