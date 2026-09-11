from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from starVLA.model.modules.world_model.vj2_predictor import VisionTransformerPredictorAC
from torch import nn


@dataclass(frozen=True)
class PredictorConfig:
    image_size: int = 256
    patch_size: int = 16
    state_dim: int = 2048
    action_dim: int = 2560
    query_tokens: int = 24
    predictor_dim: int = 512
    depth: int = 2
    heads: int = 8
    activation_checkpointing: bool = False

    @property
    def spatial_tokens(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


class VLAJEPAPredictor(nn.Module):
    """Thin, shape-checking wrapper around the released VLA-JEPA predictor."""

    def __init__(self, config: PredictorConfig):
        super().__init__()
        self.config = config
        if config.predictor_dim % config.heads:
            raise ValueError("predictor_dim must be divisible by heads")
        head_dim = config.predictor_dim // config.heads
        rope_axis_dim = 2 * ((head_dim // 3) // 2)
        if rope_axis_dim < 4:
            raise ValueError(
                "predictor_dim / heads is too small for the released predictor's 3D RoPE implementation"
            )
        self.predictor = VisionTransformerPredictorAC(
            num_frames=1,
            img_size=(config.image_size, config.image_size),
            patch_size=config.patch_size,
            tubelet_size=1,
            embed_dim=config.state_dim,
            action_embed_dim=config.action_dim,
            predictor_embed_dim=config.predictor_dim,
            depth=config.depth,
            num_heads=config.heads,
            num_add_tokens=config.query_tokens,
            use_activation_checkpointing=config.activation_checkpointing,
            is_frame_causal=True,
        )
        # These two modules are inherited from the generic action-conditioned
        # predictor but are not used by VLA-JEPA.forward. Freezing them avoids
        # unused-gradient failures under DDP.
        self.predictor.state_encoder.requires_grad_(False)
        self.predictor.extrinsics_encoder.requires_grad_(False)

    def forward(
        self, source_state: torch.Tensor, query_state: torch.Tensor
    ) -> torch.Tensor:
        expected_source = (self.config.spatial_tokens, self.config.state_dim)
        expected_query = (self.config.query_tokens, self.config.action_dim)
        if source_state.ndim != 3 or tuple(source_state.shape[1:]) != expected_source:
            raise ValueError(
                f"source_state must have shape [B, {expected_source[0]}, {expected_source[1]}], "
                f"got {tuple(source_state.shape)}"
            )
        if query_state.ndim != 3 or tuple(query_state.shape[1:]) != expected_query:
            raise ValueError(
                f"query_state must have shape [B, {expected_query[0]}, {expected_query[1]}], "
                f"got {tuple(query_state.shape)}"
            )
        return self.predictor(source_state, query_state)
