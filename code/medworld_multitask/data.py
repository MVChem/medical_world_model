"""Read-only four-task data access using the existing Table 2 cohorts.

Images are the already prepared uint8 canvases, never cached VLM/JEPA features.
Reports appear only in ``report_targets``. SR's two input branches both use the
same saved 128-square LR image; its 512-square image is only a target.
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


TASKS = ("classification", "report", "segmentation", "sr")
SPLITS = ("train", "validate", "test", "human_test")
FINDINGS = (
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax",
    "Support Devices",
)
PENDING_TASKS = {
    "vqa": "Official MIMIC-CXR-VQA data and full benchmark adapter are pending; derived QA is excluded.",
    "grounding": "MS-CXR lesion-phrase data and adapter are pending; anatomy boxes are excluded.",
}


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


def _patient_audit(records: dict[str, dict[str, list[dict]]]) -> dict[str, Any]:
    """Reject leakage between any tasks' splits, not just within each task."""
    patients = {
        (task, split): {str(row["subject_id"]) for row in rows}
        for task, splits in records.items() for split, rows in splits.items()
    }
    intersections = {}
    for (task, split), values in patients.items():
        for (other_task, other_split), other_values in patients.items():
            if split == other_split:
                continue
            count = len(values & other_values)
            key = f"{task}/{split}__{other_task}/{other_split}"
            intersections[key] = count
            if count:
                raise ValueError(f"Cross-task patient split leakage: {key}, {count} patients")
    return {
        "patient_counts": {
            task: {split: len(patients[task, split]) for split in splits}
            for task, splits in records.items()
        },
        "cross_split_intersection_counts": intersections,
        "globally_patient_disjoint": True,
    }


class TaskDataset(Dataset):
    """Small row selections; pixel arrays are opened lazily and read-only."""

    def __init__(self, owner: "MultiTaskData", task: str, split: str):
        self.owner, self.task, self.split = owner, task, split
        self.rows = owner._records[task][split]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.owner._example(self.task, self.split, self.rows[index])


class MultiTaskData:
    """Four real tasks with common patient holdouts and historical test IDs.

    ``dataset(task, split)`` is a PyTorch Dataset. ``collate(task, examples)``
    returns CPU tensors and PIL RGB images. All tasks have ``images``, ``pixels``,
    ``ids``, ``subject_ids``, ``boxes``, ``task`` and ``split``. Classification
    adds ``labels`` / ``label_mask``; report adds ``report_targets``; dense tasks
    add ``targets`` / ``mask``. No report text is returned for another task.

    Pixel branch shapes are [B,1,256,256], except SR [B,1,128,128]. PIL images
    are 512-square, except SR 128-square. Segmentation targets are 3x256x256,
    or 2x256x256 for the external human test. SR targets are 1x512x512.
    """

    def __init__(self, root: str | Path | None = None, dense_train_n: int = 4096):
        if dense_train_n != 4096:
            raise ValueError("This matched Table 2 adapter only supports dense_train_n=4096")
        self.root = Path(root or Path(__file__).resolve().parents[2]).resolve()
        self.old = self.root / "code/medworld_stage1/data/overnight_20260910"
        self.dense = self.root / "code/medworld_dense_baselines/runs/frozen_slots_20260913/data"
        self.raw = self.root / "code/medworld_baselines/runs/raw_models_20260911"
        selection_path = self.root / "code/medworld_stage1/data/slot44_20260911_derived_v2/selection.json"
        classification_root = self.root / "code/medworld_open_baselines/runs/comparators_20260913/dense_4096/dinov2_vitb14"
        self._arrays: dict[str, np.ndarray] = {}
        self._records = {task: {split: [] for split in SPLITS} for task in TASKS}
        source_paths = [self.old / "observations.jsonl", self.old / "manifest.json",
                        self.old / "segmentation.json", selection_path,
                        self.dense / "observations.jsonl", self.dense / "manifest.json",
                        classification_root / "classification_data.json",
                        classification_root / "classification_observations.jsonl",
                        self.raw / "protocol.json"]
        old_manifest, dense_manifest = _read(self.old / "manifest.json"), _read(self.dense / "manifest.json")
        source_hashes = {str(path.relative_to(self.root)): _sha256(path) for path in source_paths}

        def checked(path, expected):
            if source_hashes[str(path.relative_to(self.root))] != expected:
                raise ValueError(f"Source fingerprint mismatch: {path.name}")

        checked(self.old / "observations.jsonl", old_manifest["observations_sha256"])
        checked(self.old / "observations.jsonl", _read(self.old / "segmentation.json")["observations_sha256"])
        checked(self.dense / "observations.jsonl", dense_manifest["cohort_sha256"])
        class_contract = _read(classification_root / "classification_data.json")
        checked(self.old / "observations.jsonl", class_contract["observations_sha256"])
        checked(selection_path, class_contract["selection_sha256"])
        selection = _read(selection_path)["patient_split_overrides"]
        old_rows = _rows(self.old / "observations.jsonl")
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

        for row in old_rows:
            if not row["report_valid"] or not row["report"].strip():
                continue
            split = selection.get(str(row["subject_id"]), row["split"])
            self._records["report"][split].append({
                "id": row["id"], "subject_id": str(row["subject_id"]),
                "image_index": row["index"], "box": row["box"],
                "report_target": row["report"],
            })

        raw_protocol = _read(self.raw / "protocol.json")
        for split in ("validate", "test"):
            inputs_path = self.raw / "cohort" / f"table2_inputs_{split}.jsonl"
            refs_path = self.raw / "cohort" / f"table2_references_{split}.jsonl"
            for path in (inputs_path, refs_path):
                actual = _sha256(path)
                if actual != raw_protocol["file_sha256"][path.name]:
                    raise ValueError(f"Frozen evaluation cohort changed: {path.name}")
                source_hashes[str(path.relative_to(self.root))] = actual
            inputs = _rows(inputs_path)
            refs = {row["id"]: row for row in _rows(refs_path)}
            for task, flag in (("classification", "classification"), ("report", "report_generation")):
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
        self._dense_count = len(dense_rows)
        if self._dense_count != dense_manifest["images"]:
            raise ValueError("Dense observation count differs from its manifest")
        for i, row in enumerate(dense_rows):
            if row["index"] != i:
                raise ValueError("Dense observation indices differ from image array order")
            for task in ("segmentation", "sr"):
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

        self.counts = {task: {split: len(rows) for split, rows in splits.items()}
                       for task, splits in self._records.items()}
        expected = {"classification": (13681, 160, 353, 0), "report": (22646, 307, 507, 0),
                    "segmentation": (4096, 249, 447, 138), "sr": (4096, 249, 447, 0)}
        for task, counts in expected.items():
            if tuple(self.counts[task][split] for split in SPLITS) != counts:
                raise ValueError(f"Unexpected matched cohort counts for {task}: {self.counts[task]}")
        labels = np.asarray([row["labels"] for row in self._records["classification"]["train"]])
        self.pos_weight = torch.tensor(np.clip((labels == 0).sum(0) / np.maximum((labels == 1).sum(0), 1),
                                              .25, 10), dtype=torch.float32)
        self.metadata = {
            "version": 1, "tasks": list(TASKS), "pending_tasks": dict(PENDING_TASKS),
            "counts": self.counts, "findings": list(FINDINGS),
            "source_sha256": source_hashes,
            "patient_audit": _patient_audit(self._records),
            "classification": "Image only; exact C0 labels/IDs; unknown and uncertain labels masked",
            "report": "Current image only; report is target only; exact C0 validation/test cohort",
            "segmentation": "CXAS 3-organ soft targets; Montgomery human test has two lungs only",
            "sr": "x4; every encoder/decoder input is the same prepared uint8 LR; HR target only",
            "jepa_features": "Computed from images by the model; no cached features loaded",
            "image_geometry": "Existing 512-square padded canvases; dense valid ROI boxes preserved",
            "declared_array_sha256": {
                str((self.old / "images.npy").relative_to(self.root)): class_contract["image_sha256"],
                str((self.dense / "images.npy").relative_to(self.root)): dense_manifest["image_sha256"],
                str((self.dense / "lr_images.npy").relative_to(self.root)): dense_manifest["lr_image_sha256"],
                str((self.dense / "human_masks.npy").relative_to(self.root)): dense_manifest["human_mask_sha256"],
            },
            "array_validation": "Read-only memmaps with shape/dtype checks; declared large-array hashes are inherited, not recomputed",
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
        if key not in self._arrays:
            specs = {
                "current": (self.old / "images.npy", (self._old_count, 512, 512), np.uint8),
                "dense_hr": (self.dense / "images.npy", (self._dense_count, 512, 512), np.uint8),
                "dense_lr": (self.dense / "lr_images.npy", (self._dense_count, 128, 128), np.uint8),
                "pseudo": (self.old / "seg_probs.npy", (self._old_count, 3, 256, 256), np.float16),
                "human": (self.dense / "human_masks.npy", (138, 2, 256, 256), np.uint8),
            }
            path, shape, dtype = specs[key]
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(f"Prepared array shape/dtype mismatch: {key}")
            self._arrays[key] = array
        return self._arrays[key]

    def _example(self, task: str, split: str, row: dict) -> dict:
        if task in ("classification", "report"):
            source = self._array("current")[row["image_index"]]
        elif task == "sr":
            source = self._array("dense_lr")[row["index"]]
        else:
            source = self._array("dense_hr")[row["index"]]
        source = np.array(source, copy=True)
        pixels = torch.from_numpy(source).float()[None] / 255
        if task != "sr":
            pixels = F.interpolate(pixels[None], (256, 256), mode="area")[0]
        result = {"task": task, "split": split, "id": row["id"],
                  "subject_id": str(row["subject_id"]), "box": list(row["box"]),
                  "image": Image.fromarray(source).convert("RGB"), "pixels": pixels}
        if task == "classification":
            result["labels"] = torch.tensor(row["labels"], dtype=torch.float32)
            result["label_mask"] = (result["labels"] == 0) | (result["labels"] == 1)
        elif task == "report":
            result["report_target"] = row["report_target"]
        else:
            if task == "sr":
                target = torch.from_numpy(np.array(self._array("dense_hr")[row["index"]], copy=True)).float()[None] / 255
                size = 512
            else:
                target = self._array("human")[row["human_index"]] if row["kind"] == "montgomery" else self._array("pseudo")[row["old_index"]]
                target = torch.from_numpy(np.array(target, copy=True)).float()
                size = 256
            mask = torch.zeros(1, size, size, dtype=torch.float32)
            y, x, height, width = [value // (512 // size) for value in row["box"]]
            if min(y, x, height, width) < 0 or height == 0 or width == 0 or y + height > size or x + width > size:
                raise ValueError("Invalid dense target ROI")
            mask[:, y:y + height, x:x + width] = 1
            if not torch.isfinite(target).all():
                raise ValueError("Nonfinite dense target")
            result.update(targets=target, mask=mask)
        return result

    @staticmethod
    def collate(task: str, examples: list[dict]) -> dict:
        if not examples:
            raise ValueError("Cannot collate an empty task batch")
        if any(example["task"] != task for example in examples):
            raise ValueError("Cannot mix tasks within a batch")
        split = examples[0]["split"]
        if any(example["split"] != split for example in examples):
            raise ValueError("Cannot mix splits within a batch")
        result = {
            "task": task, "split": split,
            "images": [example["image"] for example in examples],
            "pixels": torch.stack([example["pixels"] for example in examples]),
            "ids": [example["id"] for example in examples],
            "subject_ids": [example["subject_id"] for example in examples],
            "boxes": [example["box"] for example in examples],
        }
        if task == "report":
            result["report_targets"] = [example["report_target"] for example in examples]
        elif task == "classification":
            for key in ("labels", "label_mask"):
                result[key] = torch.stack([example[key] for example in examples])
        elif task in ("segmentation", "sr"):
            for key in ("targets", "mask"):
                result[key] = torch.stack([example[key] for example in examples])
        else:
            raise ValueError(f"Unsupported task: {task}")
        return result
