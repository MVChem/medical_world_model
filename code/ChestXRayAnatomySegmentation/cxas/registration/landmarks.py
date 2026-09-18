"""
Landmark extraction from segmentation masks.

Extracts T4 and T10 vertebrae centroids for affine registration.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


# Vertebrae indices from label_mapper
T4_INDEX = 14
T10_INDEX = 20


@dataclass
class Landmark:
    """A 2D landmark point."""
    name: str
    x: float  # horizontal position (column)
    y: float  # vertical position (row)
    valid: bool = True

    def to_array(self) -> np.ndarray:
        """Return as (x, y) array."""
        return np.array([self.x, self.y])

    def to_dict(self) -> Dict:
        """Return as dictionary."""
        return {
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "valid": self.valid
        }


@dataclass
class LandmarkPair:
    """A pair of landmarks (T4 and T10)."""
    t4: Landmark
    t10: Landmark

    @property
    def valid(self) -> bool:
        """Both landmarks must be valid."""
        return self.t4.valid and self.t10.valid

    def to_array(self) -> np.ndarray:
        """Return as (2, 2) array with rows [T4, T10] and cols [x, y]."""
        return np.array([
            [self.t4.x, self.t4.y],
            [self.t10.x, self.t10.y]
        ])

    def to_dict(self) -> Dict:
        """Return as dictionary."""
        return {
            "t4": self.t4.to_dict(),
            "t10": self.t10.to_dict(),
            "valid": self.valid
        }


class LandmarkExtractor:
    """
    Extracts landmark centroids from segmentation masks.

    Uses T4 (index 14) and T10 (index 20) vertebrae for registration.
    """

    def __init__(self):
        self.t4_index = T4_INDEX
        self.t10_index = T10_INDEX

    def _compute_centroid(
        self,
        mask: np.ndarray,
        name: str
    ) -> Landmark:
        """
        Compute centroid of a binary mask.

        Args:
            mask: 2D binary mask (H, W)
            name: Name of the landmark

        Returns:
            Landmark with centroid coordinates, or invalid landmark if empty
        """
        if mask.sum() == 0:
            return Landmark(name=name, x=0.0, y=0.0, valid=False)

        # Find non-zero coordinates
        rows, cols = np.where(mask)

        # Compute centroid
        y_center = rows.mean()
        x_center = cols.mean()

        return Landmark(name=name, x=float(x_center), y=float(y_center), valid=True)

    def extract(
        self,
        segmentation: np.ndarray
    ) -> LandmarkPair:
        """
        Extract T4 and T10 landmarks from segmentation mask.

        Args:
            segmentation: (C, H, W) multi-label segmentation mask

        Returns:
            LandmarkPair with T4 and T10 centroids
        """
        # Extract individual vertebrae masks
        t4_mask = segmentation[self.t4_index]
        t10_mask = segmentation[self.t10_index]

        # Compute centroids
        t4_landmark = self._compute_centroid(t4_mask, "T4")
        t10_landmark = self._compute_centroid(t10_mask, "T10")

        return LandmarkPair(t4=t4_landmark, t10=t10_landmark)

    def extract_batch(
        self,
        segmentations: np.ndarray
    ) -> list:
        """
        Extract landmarks from a batch of segmentations.

        Args:
            segmentations: (B, C, H, W) batch of segmentation masks

        Returns:
            List of LandmarkPair objects
        """
        return [self.extract(seg) for seg in segmentations]


def compute_average_landmarks(
    landmarks_list: list
) -> Optional[LandmarkPair]:
    """
    Compute average landmarks from multiple images.

    Only includes valid landmarks in the average.

    Args:
        landmarks_list: List of LandmarkPair objects

    Returns:
        Average LandmarkPair, or None if no valid landmarks
    """
    valid_t4 = [lm.t4 for lm in landmarks_list if lm.t4.valid]
    valid_t10 = [lm.t10 for lm in landmarks_list if lm.t10.valid]

    if not valid_t4 or not valid_t10:
        return None

    avg_t4 = Landmark(
        name="T4",
        x=np.mean([lm.x for lm in valid_t4]),
        y=np.mean([lm.y for lm in valid_t4]),
        valid=True
    )

    avg_t10 = Landmark(
        name="T10",
        x=np.mean([lm.x for lm in valid_t10]),
        y=np.mean([lm.y for lm in valid_t10]),
        valid=True
    )

    return LandmarkPair(t4=avg_t4, t10=avg_t10)
