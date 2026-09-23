"""Manifest I/O, integrity checks and patient holdout rules."""
import hashlib
import json
from pathlib import Path
from ..downstream_tasks.registry import SPLITS


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split(value: str) -> str:
    value = {"val": "validate", "validation": "validate"}.get(value, value)
    if value not in SPLITS:
        raise ValueError(f"Unsupported split: {value}")
    return value


PRIORITY = {"train": 0, "validate": 1, "test": 2, "human_test": 3}


def patient_holdouts(current_records, observations):
    """Keep the most restrictive existing holdout; never move rows into eval."""
    result = {}
    def add(patient, split):
        patient = str(patient)
        if split not in PRIORITY:
            raise ValueError(f"Unknown split {split}")
        if patient not in result or PRIORITY[split] > PRIORITY[result[patient]]:
            result[patient] = split
    for splits in current_records.values():
        for split, rows in splits.items():
            for row in rows:
                add(row["subject_id"], split)
    for row in observations:
        add(row["patient"], row["split"])
    return result


def manifest_key(path, root):
    """Use logical project-relative paths, preserving prepared-data symlinks."""
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(Path(path).resolve())
