"""segmentation decoder; the shared architecture has independent task parameters."""
from ..common.spatial import SpatialHead

class SegmentationHead(SpatialHead):
    def __init__(self):
        super().__init__("segmentation")
