"""Global patient holdouts for the joint Table 1 / Table 2 training protocol."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

import numpy as np
import torch

from ..downstream_tasks.data import MultiTaskData
from ..downstream_tasks.registry import TASKS
from .temporal import TemporalData

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


class UnifiedData:
    def __init__(self, cfg, root=None):
        self.current = MultiTaskData(root=root)
        self.temporal = TemporalData(cfg["temporal_data"], cfg["bidirectional"], cfg.get("image_workers", 1))
        holdouts = patient_holdouts(self.current._records, self.temporal.observations)
        dropped = {}
        for task, splits in self.current._records.items():
            for split, rows in splits.items():
                kept = [r for r in rows if holdouts[str(r["subject_id"])] == split]
                splits[split] = kept
                dropped[f"{task}/{split}"] = len(rows) - len(kept)
        self.current.counts = {task: {s: len(r) for s, r in splits.items()}
                               for task, splits in self.current._records.items()}
        labels = np.asarray([r["labels"] for r in self.current._records["classification"]["train"]])
        if not len(labels):
            raise ValueError("Empty classification training pool after holdouts")
        self.current.pos_weight = torch.tensor(np.clip(
            (labels == 0).sum(0) / np.maximum((labels == 1).sum(0), 1), .25, 10), dtype=torch.float32)
        temporal_before, temporal_after, excluded_patients = {}, {}, {}
        for split, rows in self.temporal.pairs.items():
            kept = [r for r in rows if holdouts[str(r["patient"])] == split]
            temporal_before[split], temporal_after[split] = len(rows), len(kept)
            excluded_patients[split] = len({str(r["patient"]) for r in rows if holdouts[str(r["patient"])] != split})
            self.temporal.pairs[split] = kept
        used = {}
        def audit(patient, split):
            patient = str(patient)
            if used.setdefault(patient, split) != split:
                raise ValueError("Remaining cross-task patient leakage")
        for splits in self.current._records.values():
            for split, rows in splits.items():
                for row in rows:
                    audit(row["subject_id"], split)
        for split, rows in self.temporal.pairs.items():
            for row in rows:
                audit(row["patient"], split)
        self.metadata = {
            "current_original": self.current.metadata, "current_filtered_counts": self.current.counts,
            "current_dropped_rows": dropped, "temporal_original_pair_counts": temporal_before,
            "temporal_filtered_pair_counts": temporal_after, "temporal_excluded_patients": excluded_patients,
            "temporal_source_sha256": self.temporal.source_hashes,
            "patient_policy": "Keep test/human_test; drop conflicting train/validate rows; never relocate rows",
            "globally_patient_disjoint": True, "bidirectional": cfg["bidirectional"],
            "time_condition": "signed realized_gap_hours", "ehr": False,
        }
        self.fingerprint = hashlib.sha256(json.dumps(self.metadata, sort_keys=True).encode()).hexdigest()
        self._permutations = {}
        self._image_workers = cfg.get("image_workers", 1)
        self._current_image_pool = None
        self._directed = {split: self.temporal.directed(split) for split in self.temporal.pairs}

    def rows(self, task, split):
        if task == "temporal":
            return self._directed[split]
        return self.current._records[task][split]

    def batch(self, task, split, indices, *, source_only=False):
        if task == "temporal":
            rows = self.rows(task, split)
            return self.temporal.batch([rows[i] for i in indices], source_only=source_only)
        dataset = self.current.dataset(task, split)
        if self._image_workers > 1:
            if self._current_image_pool is None:
                self._current_image_pool = ThreadPoolExecutor(max_workers=self._image_workers, thread_name_prefix="current-image")
            examples = list(self._current_image_pool.map(dataset.__getitem__, indices))
        else:
            examples = [dataset[i] for i in indices]
        return self.current.collate(task, examples)

    def training_batch(self, task, offset, batch_size, seed):
        """Stateless permutation stream; offset/seed determine exact resume order."""
        size = len(self.rows(task, "train"))
        if size == 0:
            raise ValueError(f"Empty {task} training pool")
        indices = []
        for position in range(offset, offset + batch_size):
            epoch, index = divmod(position, size)
            key = (task, seed, epoch, size)
            if key not in self._permutations:
                digest = hashlib.sha256(f"{task}:{seed}:{epoch}".encode()).digest()
                rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
                # Only one epoch per task is needed; memory stays bounded.
                self._permutations = {k: v for k, v in self._permutations.items() if k[0] != task}
                self._permutations[key] = rng.permutation(size)
            indices.append(int(self._permutations[key][index]))
        return self.batch(task, "train", indices)
