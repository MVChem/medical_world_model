"""
Reference feature database management.

Handles building, saving, and loading reference features for orientation detection
and landmark-based registration.
"""

import numpy as np
import torch
from typing import Dict, Optional, List
from pathlib import Path
import os

from .landmarks import LandmarkPair, compute_average_landmarks
from .orientation import OrientationDetector, create_orientation_variants


# Default reference features path
DEFAULT_REFERENCE_PATH = Path(__file__).parent.parent / "data" / "reference_features.npz"


def save_reference(
    reference: Dict,
    path: str
) -> None:
    """
    Save reference features to disk.

    Args:
        reference: Dictionary containing reference data
        path: Path to save the .npz file
    """
    np.savez_compressed(path, **reference)


def load_reference(
    path: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Load reference features from disk.

    Args:
        path: Path to .npz file. If None, uses default reference.

    Returns:
        Dictionary containing reference features
    """
    if path is None:
        path = DEFAULT_REFERENCE_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Reference features not found at {path}. "
            "Use --build-reference to create them first."
        )

    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


class ReferenceBuilder:
    """
    Builds reference features for orientation detection and registration.

    Can build from a single image or average features from a directory.
    """

    def __init__(self, model):
        """
        Initialize reference builder.

        Args:
            model: CXAS model or backbone model for feature extraction
        """
        self.model = model

    def _extract_pooled_features(
        self,
        data: torch.Tensor
    ) -> np.ndarray:
        """
        Extract pooled features from backbone.

        Args:
            data: (B, C, H, W) preprocessed image tensor

        Returns:
            (B, 2048) feature array
        """
        with torch.no_grad():
            # Access the backbone through the model hierarchy
            if hasattr(self.model, 'model'):
                # CXAS wrapper
                backbone = self.model.model.backbone
            else:
                # Direct model
                backbone = self.model.backbone

            backbone_dict = backbone(data)
            return backbone_dict["feats_pooled"].cpu().numpy()

    def build_orientation_reference(
        self,
        image_data: torch.Tensor,
        source_name: str = "unknown"
    ) -> Dict[str, np.ndarray]:
        """
        Build orientation reference features from a single image.

        Creates:
        - 2 color variants: normal, inverted
        - 4 rotation variants: 0, 90, 180, 270 degrees (from normal)

        Args:
            image_data: (1, C, H, W) preprocessed image tensor
            source_name: Name of source image for metadata

        Returns:
            Dictionary with color_features, rotation_features, source
        """
        variants = create_orientation_variants(image_data)

        # Extract features for color variants
        color_features = []
        for mode in OrientationDetector.COLOR_MODES:
            feat = self._extract_pooled_features(variants[mode])
            color_features.append(feat[0])

        # Extract features for rotation variants (from normal image)
        rotation_features = []
        for angle in OrientationDetector.ROTATION_ANGLES:
            feat = self._extract_pooled_features(variants[f"rot_{angle}"])
            rotation_features.append(feat[0])

        return {
            "color_features": np.array(color_features),
            "rotation_features": np.array(rotation_features),
            "source": np.array(source_name)
        }

    def build_landmarks_reference(
        self,
        landmarks_list: List[LandmarkPair]
    ) -> Dict[str, np.ndarray]:
        """
        Build landmark reference from multiple images.

        Computes average T4 and T10 positions.

        Args:
            landmarks_list: List of LandmarkPair from multiple images

        Returns:
            Dictionary with average landmarks
        """
        avg = compute_average_landmarks(landmarks_list)

        if avg is None:
            raise ValueError("No valid landmarks found to build reference")

        return {
            "t4_x": np.array(avg.t4.x),
            "t4_y": np.array(avg.t4.y),
            "t10_x": np.array(avg.t10.x),
            "t10_y": np.array(avg.t10.y),
            "num_samples": np.array(len(landmarks_list))
        }

    def build_full_reference(
        self,
        image_data: torch.Tensor,
        landmarks: Optional[LandmarkPair] = None,
        source_name: str = "unknown"
    ) -> Dict[str, np.ndarray]:
        """
        Build complete reference including orientation and landmarks.

        Args:
            image_data: (1, C, H, W) preprocessed image tensor
            landmarks: Optional LandmarkPair for landmark reference
            source_name: Name of source image

        Returns:
            Complete reference dictionary
        """
        reference = self.build_orientation_reference(image_data, source_name)

        if landmarks is not None and landmarks.valid:
            reference.update({
                "t4_x": np.array(landmarks.t4.x),
                "t4_y": np.array(landmarks.t4.y),
                "t10_x": np.array(landmarks.t10.x),
                "t10_y": np.array(landmarks.t10.y),
            })

        return reference


def get_default_reference_landmarks(
    reference: Dict[str, np.ndarray]
) -> Optional[LandmarkPair]:
    """
    Extract landmark pair from loaded reference.

    Args:
        reference: Loaded reference dictionary

    Returns:
        LandmarkPair or None if landmarks not in reference
    """
    from .landmarks import Landmark, LandmarkPair

    if "t4_x" not in reference:
        return None

    t4 = Landmark(
        name="T4",
        x=float(reference["t4_x"]),
        y=float(reference["t4_y"]),
        valid=True
    )
    t10 = Landmark(
        name="T10",
        x=float(reference["t10_x"]),
        y=float(reference["t10_y"]),
        valid=True
    )

    return LandmarkPair(t4=t4, t10=t10)
