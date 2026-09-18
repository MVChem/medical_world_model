"""
Orientation and color detection for chest X-ray images.

Uses two-stage nearest neighbor classification with backbone features
to detect and correct image orientation and color inversion.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
from pathlib import Path


class OrientationDetector:
    """
    Detects and corrects orientation and color of chest X-ray images.

    Uses a two-stage approach:
    1. Color detection: Compare input to normal/inverted reference features
    2. Rotation detection: Compare (color-corrected) input to 0/90/180/270 rotation references
    """

    ROTATION_ANGLES = [0, 90, 180, 270]
    COLOR_MODES = ["normal", "inverted"]

    def __init__(self, reference_features: Dict[str, np.ndarray]):
        """
        Initialize the orientation detector.

        Args:
            reference_features: Dictionary containing:
                - "color_features": (2, 2048) array for normal/inverted
                - "rotation_features": (4, 2048) array for 0/90/180/270 degrees
        """
        self.color_features = torch.tensor(
            reference_features["color_features"], dtype=torch.float32
        )
        self.rotation_features = torch.tensor(
            reference_features["rotation_features"], dtype=torch.float32
        )

    def to(self, device: torch.device) -> "OrientationDetector":
        """Move reference features to specified device."""
        self.color_features = self.color_features.to(device)
        self.rotation_features = self.rotation_features.to(device)
        return self

    def _nearest_neighbor(
        self,
        query: torch.Tensor,
        references: torch.Tensor
    ) -> int:
        """
        Find nearest neighbor index using cosine similarity.

        Args:
            query: (2048,) feature vector
            references: (N, 2048) reference feature vectors

        Returns:
            Index of the nearest neighbor
        """
        query = F.normalize(query.unsqueeze(0), dim=1)
        refs = F.normalize(references, dim=1)
        similarities = torch.mm(query, refs.t())
        return similarities.argmax(dim=1).item()

    def detect_color(self, features: torch.Tensor) -> Tuple[str, bool]:
        """
        Detect if image is color-inverted.

        Args:
            features: (2048,) pooled feature vector from backbone

        Returns:
            Tuple of (color_mode, needs_inversion)
        """
        idx = self._nearest_neighbor(features, self.color_features)
        color_mode = self.COLOR_MODES[idx]
        needs_inversion = (color_mode == "inverted")
        return color_mode, needs_inversion

    def detect_rotation(self, features: torch.Tensor) -> Tuple[int, int]:
        """
        Detect image rotation.

        Args:
            features: (2048,) pooled feature vector from backbone

        Returns:
            Tuple of (detected_angle, correction_angle)
            correction_angle is the inverse rotation needed to correct the image
        """
        idx = self._nearest_neighbor(features, self.rotation_features)
        detected_angle = self.ROTATION_ANGLES[idx]
        correction_angle = (360 - detected_angle) % 360
        return detected_angle, correction_angle

    @staticmethod
    def invert_image(image: torch.Tensor) -> torch.Tensor:
        """
        Invert image colors (assumes normalized image).

        For ImageNet-normalized images, we need to:
        1. Denormalize
        2. Invert (1 - x)
        3. Renormalize

        Args:
            image: (C, H, W) or (B, C, H, W) normalized tensor

        Returns:
            Inverted and renormalized tensor
        """
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device)

        if image.dim() == 4:
            mean = mean.view(1, 3, 1, 1)
            std = std.view(1, 3, 1, 1)
        else:
            mean = mean.view(3, 1, 1)
            std = std.view(3, 1, 1)

        # Denormalize
        denorm = image * std + mean
        # Invert
        inverted = 1.0 - denorm
        # Renormalize
        renorm = (inverted - mean) / std

        return renorm

    @staticmethod
    def rotate_image(image: torch.Tensor, angle: int) -> torch.Tensor:
        """
        Rotate image by specified angle (must be 0, 90, 180, or 270).

        Args:
            image: (C, H, W) or (B, C, H, W) tensor
            angle: Rotation angle in degrees (0, 90, 180, 270)

        Returns:
            Rotated tensor
        """
        if angle == 0:
            return image

        k = angle // 90

        if image.dim() == 4:
            # (B, C, H, W) - rotate each image in batch
            return torch.rot90(image, k=k, dims=[2, 3])
        else:
            # (C, H, W)
            return torch.rot90(image, k=k, dims=[1, 2])

    @staticmethod
    def invert_image_numpy(image: np.ndarray) -> np.ndarray:
        """
        Invert raw image (0-255 range).

        Args:
            image: (H, W, C) or (H, W) uint8 array

        Returns:
            Inverted uint8 array
        """
        return 255 - image

    @staticmethod
    def rotate_image_numpy(image: np.ndarray, angle: int) -> np.ndarray:
        """
        Rotate numpy image by specified angle.

        Args:
            image: (H, W, C) or (H, W) array
            angle: Rotation angle in degrees (0, 90, 180, 270)

        Returns:
            Rotated array
        """
        if angle == 0:
            return image

        k = angle // 90
        return np.rot90(image, k=k)


def create_orientation_variants(
    image: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Create all orientation variants of an image for reference building.

    Args:
        image: (B, C, H, W) normalized tensor (single image with batch dim)

    Returns:
        Dictionary with keys "normal", "inverted", "rot_0", "rot_90", "rot_180", "rot_270"
    """
    variants = {}

    # Color variants
    variants["normal"] = image
    variants["inverted"] = OrientationDetector.invert_image(image)

    # Rotation variants (from normal)
    for angle in OrientationDetector.ROTATION_ANGLES:
        key = f"rot_{angle}"
        variants[key] = OrientationDetector.rotate_image(image, angle)

    return variants
