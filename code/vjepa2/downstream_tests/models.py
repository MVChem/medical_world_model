import torch
from torch import nn


class AgeRegressor(nn.Module):
    """A small nonlinear probe over one pooled encoder feature."""

    def __init__(self, embed_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(-1)


class PatchDecoder3D(nn.Module):
    """Decode each V-JEPA token back to its grayscale MRI tubelet patch."""

    def __init__(
        self,
        embed_dim: int,
        patch_size: int,
        tubelet_size: int,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        patch_values = tubelet_size * patch_size * patch_size
        self.projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, patch_values),
        )

    def forward(
        self, tokens: torch.Tensor, frames: int, height: int, width: int
    ) -> torch.Tensor:
        batch_size = tokens.shape[0]
        temporal_patches = frames // self.tubelet_size
        height_patches = height // self.patch_size
        width_patches = width // self.patch_size
        expected_tokens = temporal_patches * height_patches * width_patches
        if tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"Expected {expected_tokens} tokens for {(frames, height, width)}, "
                f"got {tokens.shape[1]}"
            )

        patches = self.projection(tokens)
        patches = patches.view(
            batch_size,
            temporal_patches,
            height_patches,
            width_patches,
            self.tubelet_size,
            1,
            self.patch_size,
            self.patch_size,
        )
        volume = patches.permute(0, 5, 1, 4, 2, 6, 3, 7).contiguous()
        return volume.view(batch_size, 1, frames, height, width).sigmoid()
