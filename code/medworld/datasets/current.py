"""Compatibility imports for the downstream task data adapter."""
from ..downstream_tasks.data import (
    MultiTaskData, TaskDataset, TASKS, SPLITS, FINDINGS, PENDING_TASKS,
    _read, _rows, _sha256, _split, _patient_audit,
)
