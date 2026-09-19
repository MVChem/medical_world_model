"""Compatibility imports; implementations live in medworld.downstream_tasks."""
from .downstream_tasks.classification import ClassificationHead
from .downstream_tasks.report import ReportDecoder, chunked_ce
from .downstream_tasks.common import validate_state, block, positional_encoding
from .downstream_tasks.spatial import SpatialHead, spatial_loss
