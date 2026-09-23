"""Global patient holdouts for reviewed tasks and temporal representation learning."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .prepared import PreparedData
from .temporal import TemporalData
from .temporal_selection import SELECTION_SCHEMA

from .protocol import patient_holdouts


class UnifiedData:
    def __init__(self, cfg, root=None):
        if not cfg.get("prepared_data"):
            raise ValueError("UnifiedData requires prepared_data with reviewed segmentation")
        expected_temporal = Path(cfg["prepared_data"]) / "temporal"
        external_temporal = Path(cfg["temporal_data"]).resolve() != expected_temporal.resolve()
        if external_temporal:
            manifest_path = Path(cfg["temporal_data"]) / "manifest.json"
            if (not manifest_path.is_file()
                    or json.loads(manifest_path.read_text()).get("schema") != SELECTION_SCHEMA):
                raise ValueError("temporal_data requires a selection manifest or prepared_data/temporal")
        self.current = PreparedData(cfg, root=root)
        self.temporal = TemporalData(cfg["temporal_data"], cfg["bidirectional"], cfg.get("image_workers", 1))
        if not external_temporal:
            for name, digest in self.temporal.source_hashes.items():
                if self.current._file_sha256.get(f"temporal/{name}") != digest:
                    raise ValueError(f"Prepared manifest fingerprint mismatch: temporal/{name}")
        # These are immutable source pixels, not learned features. Bound memory
        # independently per source stream; nothing is written to disk.
        capacity = cfg.get("decoded_image_cache", 0)
        if capacity:
            self.current._pixels = lru_cache(maxsize=capacity)(self.current._pixels)
            self.current._vqa_image = lru_cache(maxsize=capacity)(self.current._vqa_image)
            self.temporal._observation = lru_cache(maxsize=capacity)(self.temporal._observation)
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
        temporal_before, temporal_after, excluded_patients = self.temporal.filter_holdouts(holdouts)
        used = {}
        def audit(patient, split):
            patient = str(patient)
            if used.setdefault(patient, split) != split:
                raise ValueError("Remaining cross-task patient leakage")
        for splits in self.current._records.values():
            for split, rows in splits.items():
                for row in rows:
                    audit(row["subject_id"], split)
        for patient, split in self.temporal.patient_splits():
            audit(patient, split)
        self.metadata = {
            "current_original": self.current.metadata, "current_filtered_counts": self.current.counts,
            "current_dropped_rows": dropped, "temporal_original_pair_counts": temporal_before,
            "temporal_filtered_pair_counts": temporal_after, "temporal_excluded_patients": excluded_patients,
            "temporal_source_sha256": self.temporal.source_hashes,
            "temporal_schema": self.temporal.schema,
            "temporal_input": "source/target images and each own report; signed actual time only",
            "patient_policy": "Keep test/human_test; drop conflicting train/validate rows; never relocate rows",
            "globally_patient_disjoint": True, "bidirectional": cfg["bidirectional"],
            "time_condition": "signed realized_gap_hours", "ehr": False,
        }
        self._segmentation_groups = {}
        self._segmentation_sampling = cfg.get("segmentation_sampling", "uniform")
        self.segmentation_layout = list(self.current.segmentation_layout)
        self.metadata.update(segmentation_layout=self.segmentation_layout,
                             segmentation_training_sampling=self._segmentation_sampling,
                             segmentation_mri_order="Seeded volume blocks with shuffled slices within each volume")
        for index, row in enumerate(self.current._records["segmentation"]["train"]):
            self._segmentation_groups.setdefault(row["dataset"], []).append(index)
        self.fingerprint = hashlib.sha256(json.dumps(self.metadata, sort_keys=True).encode()).hexdigest()
        self.base_data_fingerprint = self.fingerprint
        self.holdouts = dict(holdouts)
        self.future = None
        if cfg.get("future_enabled", False):
            from .future import FutureData
            extra = FutureData.extra_holdouts(cfg["future_data"])
            priority = {"train": 0, "validate": 1, "test": 2, "human_test": 3}
            for patient, split in extra.items():
                previous = self.holdouts.get(str(patient), "train")
                if split not in priority or priority[split] < priority[previous]:
                    raise ValueError("Future-task holdouts cannot weaken an existing patient holdout")
                self.holdouts[str(patient)] = split
            for task, splits in self.current._records.items():
                for split, rows in splits.items():
                    kept = [row for row in rows if self.holdouts[str(row["subject_id"])] == split]
                    splits[split] = kept
                    dropped[f"{task}/{split}"] += len(rows) - len(kept)
            self.current.counts = {task: {split: len(rows) for split, rows in splits.items()}
                                   for task, splits in self.current._records.items()}
            labels = np.asarray([row["labels"] for row in self.current._records["classification"]["train"]])
            if not len(labels):
                raise ValueError("Future-task holdouts emptied the classification training pool")
            self.current.pos_weight = torch.tensor(np.clip(
                (labels == 0).sum(0) / np.maximum((labels == 1).sum(0), 1), .25, 10), dtype=torch.float32)
            _, final_pairs, extra_excluded = self.temporal.filter_holdouts(self.holdouts)
            self._segmentation_groups = {}
            for index, row in enumerate(self.current._records["segmentation"]["train"]):
                self._segmentation_groups.setdefault(row["dataset"], []).append(index)
            self.future = FutureData(cfg, self.base_data_fingerprint, self.holdouts)
            self.metadata.update(
                current_filtered_counts=self.current.counts,
                temporal_filtered_pair_counts=final_pairs,
                temporal_excluded_patients={split: excluded_patients.get(split, 0) + extra_excluded.get(split, 0)
                                            for split in set(excluded_patients) | set(extra_excluded)},
                future=self.future.metadata,
                future_extra_holdouts=extra,
                base_data_fingerprint=self.base_data_fingerprint)
            self.fingerprint = hashlib.sha256(json.dumps(self.metadata, sort_keys=True).encode()).hexdigest()
        self._permutations = {}
        self._image_workers = cfg.get("image_workers", 1)
        self._current_image_pool = None
        self._directed = {split: self.temporal.directed(split) for split in self.temporal.pairs}

    def rows(self, task, split):
        if task == "temporal":
            return self._directed[split]
        if self.future is not None and task in self.future._rows:
            return self.future.rows(task, split)
        return self.current._records[task][split]

    def batch(self, task, split, indices, *, source_only=False):
        if task == "temporal":
            rows = self.rows(task, split)
            return self.temporal.batch([rows[i] for i in indices], source_only=source_only)
        if self.future is not None and task in self.future._rows:
            return self.future.batch(task, split, indices, source_only=source_only)
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
        if task == "segmentation" and self._segmentation_groups and self._segmentation_sampling == "balanced_dataset":
            return self._balanced_segmentation_batch(offset, batch_size, seed)
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

    def _balanced_segmentation_batch(self, offset, batch_size, seed):
        """Equal dataset exposure without materializing an oversampled cohort."""
        groups = sorted(self._segmentation_groups)
        indices = []
        for position in range(offset, offset + batch_size):
            cycle, group_index = divmod(position, len(groups))
            dataset = groups[group_index]
            pool = self._segmentation_groups[dataset]
            epoch, index = divmod(cycle, len(pool))
            key = ("segmentation", dataset, seed, epoch, len(pool))
            if key not in self._permutations:
                digest = hashlib.sha256(f"segmentation:{dataset}:{seed}:{epoch}".encode()).digest()
                rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
                self._permutations = {k: v for k, v in self._permutations.items()
                                      if not (k[0] == "segmentation" and k[1] == dataset)}
                rows = self.current._records["segmentation"]["train"]
                if rows[pool[0]].get("kind") == "mri":
                    volumes = {}
                    for local_index, row_index in enumerate(pool):
                        volumes.setdefault(rows[row_index]["volume_id"], []).append(local_index)
                    names = sorted(volumes)
                    self._permutations[key] = np.concatenate([
                        rng.permutation(volumes[names[i]]) for i in rng.permutation(len(names))])
                else:
                    self._permutations[key] = rng.permutation(len(pool))
            indices.append(pool[int(self._permutations[key][index])])
        return self.batch("segmentation", "train", indices)
