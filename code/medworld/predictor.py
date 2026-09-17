"""Direct signed-time prediction of all eight state tokens."""
import torch
from torch import nn
import torch.nn.functional as F

from . import STATE_SLOTS, STATE_WIDTH


def time_features(hours):
    if hours.ndim != 1 or not hours.is_floating_point() or not torch.isfinite(hours).all():
        raise ValueError("delta_hours must be a finite floating [B] tensor")
    days = hours.float() / 24.0
    magnitude = torch.log1p(days.abs())
    return torch.stack([days.sign(), magnitude, days.sign() * magnitude], -1)


class WorldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        width = cfg["predictor_width"]
        self.input = nn.Linear(STATE_WIDTH, width)
        self.time = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        self.positions = nn.Parameter(torch.randn(1, STATE_SLOTS, width) * .02)
        block = nn.TransformerEncoderLayer(width, 8, width * 4, dropout=0., activation="gelu",
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(block, cfg["predictor_depth"], enable_nested_tensor=False)
        self.output = nn.Linear(width, STATE_WIDTH)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, state, delta_hours):
        if state.ndim != 3 or tuple(state.shape[1:]) != (STATE_SLOTS, STATE_WIDTH):
            raise ValueError("Expected state [B,8,1024]")
        if delta_hours.shape != (len(state),):
            raise ValueError("One signed time interval is required per state")
        normalized = F.layer_norm(state.float(), (STATE_WIDTH,))
        values = self.input(normalized) + self.positions + self.time(time_features(delta_hours))[:, None]
        # Identity initialization; the interface accepts both signs immediately,
        # but directional behavior must be learned from bidirectional pairs.
        return normalized + self.output(self.transformer(values))
