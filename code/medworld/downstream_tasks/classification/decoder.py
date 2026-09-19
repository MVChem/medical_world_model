"""Finding classification: eight-slot readout, masked BCE and AUROC/AP."""
import torch
from torch import nn
import torch.nn.functional as F
from ... import STATE_WIDTH

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
