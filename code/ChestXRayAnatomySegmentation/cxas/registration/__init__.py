"""
CXAS Registration Module

Provides registration functionality for chest X-ray images with automatic
orientation/color correction and landmark-based affine registration.
"""

from .orientation import OrientationDetector
from .landmarks import LandmarkExtractor
from .affine import compute_affine_transform, apply_affine
from .reference import ReferenceBuilder, save_reference, load_reference
from .registrator import Registrator

__all__ = [
    "OrientationDetector",
    "LandmarkExtractor",
    "compute_affine_transform",
    "apply_affine",
    "ReferenceBuilder",
    "save_reference",
    "load_reference",
    "Registrator",
]
