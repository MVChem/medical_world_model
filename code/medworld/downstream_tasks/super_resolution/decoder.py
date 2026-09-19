"""super_resolution decoder; the shared architecture has independent task parameters."""
from ..common.spatial import SpatialHead

class SuperResolutionHead(SpatialHead):
    def __init__(self):
        super().__init__("sr")
