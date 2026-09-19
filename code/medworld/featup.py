"""Compatibility imports; FeatUp implementation lives in downstream_tasks."""
from .downstream_tasks.featup import (
    FeatUpSpatialHead, FrozenFeatureTeacher, ReadSlots, GuidedUpsample,
    consistency_loss, feature_loss, patch_grid, view, jitter,
)
