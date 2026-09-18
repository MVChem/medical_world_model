"""
Affine transformation computation and application.

Uses cv2.estimateAffinePartial2D for similarity transform (rotation, scale, translation).
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from .landmarks import LandmarkPair


def compute_affine_transform(
    src_landmarks: LandmarkPair,
    dst_landmarks: LandmarkPair
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Compute affine transformation from source to destination landmarks.

    Uses cv2.estimateAffinePartial2D which computes a similarity transform
    (rotation + uniform scale + translation) from point correspondences.

    Args:
        src_landmarks: Source landmarks (from input image)
        dst_landmarks: Destination landmarks (reference)

    Returns:
        Tuple of (2x3 affine matrix, success flag)
        Returns (None, False) if landmarks are invalid
    """
    if not src_landmarks.valid or not dst_landmarks.valid:
        return None, False

    src_pts = src_landmarks.to_array().astype(np.float32)
    dst_pts = dst_landmarks.to_array().astype(np.float32)

    # Estimate partial affine (similarity transform)
    # This constrains to rotation + uniform scale + translation
    matrix, inliers = cv2.estimateAffinePartial2D(
        src_pts.reshape(-1, 1, 2),
        dst_pts.reshape(-1, 1, 2),
        method=cv2.LMEDS
    )

    if matrix is None:
        return None, False

    return matrix, True


def compute_full_affine_transform(
    src_landmarks: LandmarkPair,
    dst_landmarks: LandmarkPair
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Compute full affine transformation (allows non-uniform scale and shear).

    Note: With only 2 points, this is underdetermined. Use compute_affine_transform
    for similarity transform which is sufficient for 2 landmarks.

    Args:
        src_landmarks: Source landmarks (from input image)
        dst_landmarks: Destination landmarks (reference)

    Returns:
        Tuple of (2x3 affine matrix, success flag)
    """
    if not src_landmarks.valid or not dst_landmarks.valid:
        return None, False

    src_pts = src_landmarks.to_array().astype(np.float32)
    dst_pts = dst_landmarks.to_array().astype(np.float32)

    # For full affine we need at least 3 points
    # With 2 points, we can only compute similarity transform
    # Fall back to partial affine
    return compute_affine_transform(src_landmarks, dst_landmarks)


def apply_affine(
    image: np.ndarray,
    matrix: np.ndarray,
    output_size: Optional[Tuple[int, int]] = None,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: int = 0
) -> np.ndarray:
    """
    Apply affine transformation to an image.

    Args:
        image: Input image (H, W) or (H, W, C)
        matrix: 2x3 affine transformation matrix
        output_size: Output size (width, height). If None, uses input size.
        interpolation: Interpolation method
        border_mode: Border extrapolation mode
        border_value: Value for constant border

    Returns:
        Transformed image
    """
    if output_size is None:
        output_size = (image.shape[1], image.shape[0])

    return cv2.warpAffine(
        image,
        matrix,
        output_size,
        flags=interpolation,
        borderMode=border_mode,
        borderValue=border_value
    )


def apply_affine_to_mask(
    mask: np.ndarray,
    matrix: np.ndarray,
    output_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Apply affine transformation to a segmentation mask.

    Uses nearest neighbor interpolation to preserve binary values.

    Args:
        mask: Input mask (C, H, W) multi-label mask
        matrix: 2x3 affine transformation matrix
        output_size: Output size (width, height). If None, uses input size.

    Returns:
        Transformed mask (C, H, W)
    """
    if output_size is None:
        output_size = (mask.shape[2], mask.shape[1])

    transformed = np.zeros(
        (mask.shape[0], output_size[1], output_size[0]),
        dtype=mask.dtype
    )

    for c in range(mask.shape[0]):
        transformed[c] = apply_affine(
            mask[c].astype(np.uint8),
            matrix,
            output_size,
            interpolation=cv2.INTER_NEAREST,
            border_value=0
        ).astype(mask.dtype)

    return transformed


def get_identity_transform() -> np.ndarray:
    """Return identity affine transformation matrix."""
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=np.float64)


def decompose_affine(matrix: np.ndarray) -> dict:
    """
    Decompose affine matrix into rotation, scale, and translation.

    Args:
        matrix: 2x3 affine transformation matrix

    Returns:
        Dictionary with keys: rotation (degrees), scale, translation_x, translation_y
    """
    # Extract components
    a, b, tx = matrix[0]
    c, d, ty = matrix[1]

    # Scale (assuming uniform from similarity transform)
    scale_x = np.sqrt(a**2 + c**2)
    scale_y = np.sqrt(b**2 + d**2)
    scale = (scale_x + scale_y) / 2

    # Rotation (in radians, then convert to degrees)
    rotation_rad = np.arctan2(c, a)
    rotation_deg = np.degrees(rotation_rad)

    return {
        "rotation": rotation_deg,
        "scale": scale,
        "translation_x": tx,
        "translation_y": ty
    }
