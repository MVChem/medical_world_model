"""Shared Transformer task decoder: image tokens plus optional slot conditions."""
import torch
from torch import nn
from . import validate_state, positional_encoding
from ... import STATE_WIDTH


class TaskDecoder(nn.Module):
    def __init__(self, image_width, width=256, depth=2):
        super().__init__()
        self.image_projection = nn.Sequential(nn.LayerNorm(image_width), nn.Linear(image_width, width))
        self.slot_projection = nn.Sequential(nn.LayerNorm(STATE_WIDTH), nn.Linear(STATE_WIDTH, width))
        self.slot_type = nn.Parameter(torch.zeros(1, 8, width))
        layer = nn.TransformerEncoderLayer(width, 8, width * 4, dropout=0, activation='gelu',
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, depth, norm=nn.LayerNorm(width), enable_nested_tensor=False)

    def forward(self, image_features, slots=None):
        if image_features is None or image_features.ndim != 3 or image_features.shape[1] < 1:
            raise ValueError('Task decoder requires image tokens')
        images = self.image_projection(image_features.float())
        size = int(images.shape[1] ** .5)
        if size * size != images.shape[1]:
            raise ValueError('Expected a square spatial image-token grid')
        positions = positional_encoding(size, size, images.shape[-1]).to(images)
        tokens = images + positions
        if slots is not None:
            validate_state(slots)
            tokens = torch.cat([tokens, self.slot_projection(slots.float()) + self.slot_type], 1)
        return self.transformer(tokens)[:, :images.shape[1]]
