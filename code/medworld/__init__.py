"""Unified task-supervised medical states and temporal prediction."""

STATE_WIDTH = 1024
STATE_SLOTS = 8
FORMAT_VERSION = 5

WEIGHTS_ONLY_RESUME_ERROR = (
    "MedWorld checkpoints contain trained weights only; optimizer, RNG and data-stream "
    "state are not saved, so --resume is unavailable. Use the checkpoint for inference "
    "or evaluation."
)
