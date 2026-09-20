"""Four-task cohorts with on-demand source image/mask decoding.

No prepared pixel arrays, LR arrays, or encoder features are read or written.
Existing CXAS soft labels remain fixed segmentation targets.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..config import PROJECT, load_config


from ..downstream_tasks.registry import TASKS, SPLITS, FINDINGS, PENDING_TASKS


from .protocol import manifest_key, _read, _rows, _sha256, _split, _patient_audit


class TaskDataset(Dataset):
    """Small row selections; original image files are decoded per example."""

    def __init__(self, owner: "MultiTaskData", task: str, split: str):
        self.owner, self.task, self.split = owner, task, split
        self.rows = owner._records[task][split]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.owner._example(self.task, self.split, self.rows[index])


class MultiTaskData:
    """Source-image classification and segmentation, plus text task adapters."""

    def __init__(self, root: str | Path | None = None, dense_train_n: int = 4096, cfg=None):
        if dense_train_n != 4096:
            raise ValueError("This matched Table 2 adapter only supports dense_train_n=4096")
        self.root = Path(root or PROJECT).resolve()
        cfg = cfg or load_config(root=self.root)
        self.old = Path(cfg["current_data"])
        self.dense = Path(cfg["dense_data"])
        self.raw = Path(cfg["baseline_data"])
        selection_path = Path(cfg["selection_file"])
        classification_root = Path(cfg["classification_data"])
        self._arrays: dict[str, np.ndarray] = {}
        self._records = {task: {split: [] for split in SPLITS} for task in TASKS}
        source_paths = [self.old / "observations.jsonl", self.old / "manifest.json",
                        self.old / "segmentation.json", selection_path,
                        self.dense / "observations.jsonl", self.dense / "manifest.json",
                        classification_root / "classification_data.json",
                        classification_root / "classification_observations.jsonl",
                        self.raw / "protocol.json"]
        old_manifest, dense_manifest = _read(self.old / "manifest.json"), _read(self.dense / "manifest.json")
        source_hashes = {manifest_key(path, self.root): _sha256(path) for path in source_paths}

        def checked(path, expected):
            if source_hashes[manifest_key(path, self.root)] != expected:
                raise ValueError(f"Source fingerprint mismatch: {path.name}")

        checked(self.old / "observations.jsonl", old_manifest["observations_sha256"])
        checked(self.old / "observations.jsonl", _read(self.old / "segmentation.json")["observations_sha256"])
        checked(self.dense / "observations.jsonl", dense_manifest["cohort_sha256"])
        class_contract = _read(classification_root / "classification_data.json")
        checked(self.old / "observations.jsonl", class_contract["observations_sha256"])
        checked(selection_path, class_contract["selection_sha256"])
        selection = _read(selection_path)["patient_split_overrides"]
        old_rows = _rows(self.old / "observations.jsonl")
        self._old_sources = old_rows
        self._old_count = len(old_rows)
        if any(row["index"] != i for i, row in enumerate(old_rows)):
            raise ValueError("Historical observation indices differ from image array order")
        if len(old_rows) != old_manifest["valid_count"]:
            raise ValueError("Historical observation count differs from manifest")
        by_id = {row["id"]: row for row in old_rows}
        if len(by_id) != len(old_rows):
            raise ValueError("Duplicate historical image ID")

        for row in _rows(classification_root / "classification_observations.jsonl"):
            source = old_rows[row["image_index"]]
            split = selection.get(str(source["subject_id"]), source["split"])
            if (row["id"] != source["id"] or str(row["subject_id"]) != str(source["subject_id"])
                    or row["labels"] != source["labels"] or row["split"] != split):
                raise ValueError("Classification rows differ from original labels or patient split")
            if len(row["labels"]) != len(FINDINGS):
                raise ValueError("Expected the same 13 classification labels")
            record = {key: row[key] for key in ("id", "subject_id", "image_index", "labels")}
            record["box"] = source["box"]
            self._records["classification"][split].append(record)

        raw_protocol = _read(self.raw / "protocol.json")
        for split in ("validate", "test"):
            inputs_path = self.raw / "cohort" / f"table2_inputs_{split}.jsonl"
            refs_path = self.raw / "cohort" / f"table2_references_{split}.jsonl"
            for path in (inputs_path, refs_path):
                actual = _sha256(path)
                if actual != raw_protocol["file_sha256"][path.name]:
                    raise ValueError(f"Frozen evaluation cohort changed: {path.name}")
                source_hashes[manifest_key(path, self.root)] = actual
            inputs = _rows(inputs_path)
            refs = {row["id"]: row for row in _rows(refs_path)}
            for task, flag in (("classification", "classification"),):
                records = {row["id"]: row for row in self._records[task][split]}
                selected = [row for row in inputs if row[flag]]
                if set(records) != {row["id"] for row in selected}:
                    raise ValueError(f"{task}/{split} IDs differ from the frozen C0 benchmark")
                # Keep historical benchmark ordering as well as exact membership.
                self._records[task][split] = [records[row["id"]] for row in selected]
                for row in selected:
                    source, reference = by_id[row["id"]], refs[row["id"]]
                    if str(source["subject_id"]) != str(row["patient"]):
                        raise ValueError("Evaluation patient identity mismatch")
                    field = "labels" if task == "classification" else "report"
                    if source[field] != reference[field]:
                        raise ValueError(f"{task}/{split} reference target changed")

        dense_rows = _rows(self.dense / "observations.jsonl")
        self._dense_sources = dense_rows
        self._dense_count = len(dense_rows)
        if self._dense_count != dense_manifest["images"]:
            raise ValueError("Dense observation count differs from its manifest")
        for i, row in enumerate(dense_rows):
            if row["index"] != i:
                raise ValueError("Dense observation indices differ from image array order")
            for task in ("segmentation",):
                if task not in row["tasks"] or row["split"] not in SPLITS:
                    continue
                record = {key: row[key] for key in ("id", "subject_id", "index", "box", "kind")}
                if row["kind"] == "montgomery":
                    record["human_index"] = row["human_index"]
                else:
                    record["old_index"] = row["old_index"]
                    if old_rows[row["old_index"]]["id"] != row["id"]:
                        raise ValueError("Dense pseudo-target mapping differs from original image identity")
                self._records[task][row["split"]].append(record)

        from .vqa import load_vqa
        source_hashes.update(load_vqa(cfg, self._records["vqa"]))

        self.counts = {task: {split: len(rows) for split, rows in splits.items()}
                       for task, splits in self._records.items()}
        labels = np.asarray([row["labels"] for row in self._records["classification"]["train"]])
        self.pos_weight = torch.tensor(np.clip((labels == 0).sum(0) / np.maximum((labels == 1).sum(0), 1),
                                              .25, 10), dtype=torch.float32)
        self.metadata = {
            "version": 3, "input_mode": "on_demand_source_no_pixel_cache", "tasks": list(TASKS), "pending_tasks": dict(PENDING_TASKS),
            "counts": self.counts, "findings": list(FINDINGS),
            "source_sha256": source_hashes,
            "patient_audit": "Computed after global holdout filtering",
            "classification": "Image only; exact C0 labels/IDs; unknown and uncertain labels masked",
            "segmentation": "CXAS 3-organ soft targets; Montgomery human test has two lungs only",
            "jepa_features": "Computed from images by the model; no cached features loaded",
            "image_geometry": "Existing 512-square padded canvases; dense valid ROI boxes preserved",
            "declared_array_sha256": {
                manifest_key(self.old / "images.npy", self.root): class_contract["image_sha256"],
                manifest_key(self.dense / "images.npy", self.root): dense_manifest["image_sha256"],
                manifest_key(self.dense / "lr_images.npy", self.root): dense_manifest["lr_image_sha256"],
                manifest_key(self.dense / "human_masks.npy", self.root): dense_manifest["human_mask_sha256"],
            },
            "array_validation": "Only original CXAS label array is loaded with shape/dtype checks; historical pixel-array hashes are provenance only, pixel arrays are never read",
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_arrays"] = {}
        return state

    def dataset(self, task: str, split: str) -> TaskDataset:
        if task in PENDING_TASKS:
            raise NotImplementedError(PENDING_TASKS[task])
        if task not in TASKS:
            raise ValueError(f"Unsupported task: {task}")
        return TaskDataset(self, task, _split(split))

    def _array(self, key: str) -> np.ndarray:
        # This is the original supervised label array, not an input/feature cache.
        if key != "pseudo":
            raise ValueError("Prepared pixel caches are disabled; only CXAS labels may be loaded")
        if key not in self._arrays:
            array = np.load(self.old / "seg_probs.npy", mmap_mode="r", allow_pickle=False)
            if array.shape != (self._old_count, 3, 256, 256) or array.dtype != np.float16:
                raise ValueError("Original CXAS supervision shape/dtype mismatch")
            self._arrays[key] = array
        return self._arrays[key]

    def _pixels(self, key, index):
        from .pixels import source_canvas
        if key not in ("current", "dense_hr"):
            raise ValueError(f"Unknown source pixel kind: {key}")
        record = (self._old_sources if key == "current" else self._dense_sources)[index]
        pixels = source_canvas(record)
        return (pixels[0] * 255).round().byte().numpy()

    def _human_target(self, row):
        from .pixels import human_target
        return human_target(self._dense_sources[row["index"]]).numpy()

    def _example(self, task, split, row):
        from PIL import ImageOps
        if task == "vqa":
            with Image.open(row["image"]) as image:
                image = ImageOps.pad(image.convert("RGB"), (512, 512), method=Image.Resampling.BICUBIC, color="black")
            return {"task": task, "split": split, "id": row["id"], "subject_id": str(row["subject_id"]),
                    "image": image, "question": row["question"], "answer": row["answer"],
                    "semantic_type": row.get("semantic_type", "diagnosis")}
        source = self._pixels("current" if task == "classification" else "dense_hr",
                              row["image_index"] if task == "classification" else row["index"])
        pixels = torch.from_numpy(np.array(source, copy=True)).float()[None] / 255
        pixels = F.interpolate(pixels[None], (256, 256), mode="area")[0]
        result = {"task": task, "split": split, "id": row["id"], "subject_id": str(row["subject_id"]),
                  "image": Image.fromarray(source).convert("RGB"), "pixels": pixels}
        if task == "classification":
            result["labels"] = torch.tensor(row["labels"], dtype=torch.float32)
            result["label_mask"] = (result["labels"] == 0) | (result["labels"] == 1)
        elif task == "segmentation":
            target = self._human_target(row) if row["kind"] == "montgomery" else self._array("pseudo")[row["old_index"]]
            target = torch.from_numpy(np.array(target, copy=True)).float()
            mask = torch.zeros(1, 256, 256)
            y, x, height, width = [value // 2 for value in row["box"]]
            if min(y, x, height, width) < 0 or height == 0 or width == 0 or y + height > 256 or x + width > 256:
                raise ValueError("Invalid segmentation ROI")
            mask[:, y:y + height, x:x + width] = 1
            if not torch.isfinite(target).all():
                raise ValueError("Nonfinite segmentation target")
            result.update(targets=target, mask=mask)
        else:
            raise ValueError(f"Unsupported task: {task}")
        return result

    @staticmethod
    def collate(task, examples):
        if not examples or any(e["task"] != task or e["split"] != examples[0]["split"] for e in examples):
            raise ValueError("Expected a nonempty batch from one task and split")
        result = {"task": task, "split": examples[0]["split"], "images": [e["image"] for e in examples],
                  "ids": [e["id"] for e in examples], "subject_ids": [str(e["subject_id"]) for e in examples]}
        if task == "vqa":
            result.update(questions=[e["question"] for e in examples], answers=[e["answer"] for e in examples],
                          semantic_types=[e["semantic_type"] for e in examples])
        else:
            keys = ("pixels", "labels", "label_mask") if task == "classification" else ("pixels", "targets", "mask")
            for key in keys:
                result[key] = torch.stack([e[key] for e in examples])
        return result
