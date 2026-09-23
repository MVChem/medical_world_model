"""Human-reviewed Atlas cohorts with original images and source masks."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..config import PROJECT
from ..downstream_tasks.registry import FINDINGS, MANUAL_SEGMENTATION_LAYOUT, SPLITS, TASKS
from .pixels import human_target, source_canvas
from .protocol import _read, _rows, _sha256, _split, manifest_key
from .vqa import load_vqa


class TaskDataset(Dataset):
    """Decode original image files for a selected task and split."""

    def __init__(self, owner, task, split):
        self.owner, self.task, self.split = owner, task, split
        self.rows = owner._records[task][split]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.owner._example(self.task, self.split, self.rows[index])


class PreparedData:
    """Load prepared-v2 cohorts with reviewed CXR and MRI segmentation.

    Source images and manual masks are decoded on demand. No prepared pixel
    arrays, pseudo-target arrays, or encoder feature caches are used.
    """

    def __init__(self, cfg, root=None):
        self.root = Path(cfg["prepared_data"]).absolute()
        project = Path(root or PROJECT).resolve()
        manifest_path = self.root / "manifest.json"
        manifest = _read(manifest_path)
        if manifest.get("schema") != "medworld-prepared-v2":
            raise ValueError("Prepared data requires the medworld-prepared-v2 schema")
        self.segmentation_layout = list(MANUAL_SEGMENTATION_LAYOUT)
        if manifest.get("segmentation_layout") != self.segmentation_layout:
            raise ValueError("Prepared manual segmentation layout differs from MedWorld")
        if cfg.get("segmentation_channels") != len(self.segmentation_layout):
            raise ValueError("Prepared manual segmentation requires segmentation_channels=6")
        if manifest.get("findings") != list(FINDINGS):
            raise ValueError("Prepared classification finding order differs from MedWorld")
        self._file_sha256 = manifest.get("file_sha256", {})
        self._checked_paths = {}
        self._sources = {task: [] for task in ("classification", "segmentation")}
        self._records = {task: {split: [] for split in SPLITS} for task in TASKS}
        hashes = {manifest_key(manifest_path, project): _sha256(manifest_path)}
        targets = {}
        target_keys = {}
        for task in self._sources:
            name = f"{task}.jsonl"
            path = self.root / name
            digest = _sha256(path)
            expected = self._file_sha256.get(name)
            if expected is None or digest != expected:
                raise ValueError(f"Prepared manifest fingerprint mismatch: {name}")
            hashes[manifest_key(path, project)] = digest
            identities = set()
            for row in _rows(path):
                row = dict(row)
                identity, split = row["id"], row["split"]
                if not isinstance(identity, str) or not identity or identity in identities:
                    raise ValueError(f"Duplicate or invalid prepared {task} ID")
                identities.add(identity)
                if split not in SPLITS or (task == "classification" and split == "human_test"):
                    raise ValueError(f"Invalid prepared {task} split")
                if not str(row["subject_id"]).strip():
                    raise ValueError("Prepared observation has no patient identity")
                row["subject_id"] = str(row["subject_id"])
                row["image"] = self._path(row["image"])
                box = row["box"]
                if (not isinstance(box, list) or len(box) != 4
                        or any(type(value) is not int or value < 0 or value % 2 for value in box)):
                    raise ValueError("Prepared ROI must contain four nonnegative even integers")
                y, x, height, width = box
                if not height or not width or y + height > 512 or x + width > 512:
                    raise ValueError("Prepared ROI exceeds the 512-square source canvas")
                if task == "classification":
                    labels = row["labels"]
                    if (not isinstance(labels, list) or len(labels) != len(FINDINGS)
                            or any(type(value) is not int or value not in (-2, -1, 0, 1) for value in labels)):
                        raise ValueError("Prepared classification requires 13 four-state labels")
                else:
                    self._manual_record(row)
                    paths = row.get("masks", []) + ([row["mask_file"]] if "mask_file" in row else [])
                    for path in paths:
                        # Thousands of slices share each MRI source file. Resolve
                        # and stat that immutable file once during manifest load.
                        if path not in target_keys:
                            target_keys[path] = manifest_key(Path(path), project)
                        key = target_keys[path]
                        if key not in targets:
                            stat = Path(path).stat()
                            targets[key] = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                            "annotation": row["annotation"]}
                            # CXR PNG annotations are small. MRI volume identity
                            # uses the builder's source digest plus file stats.
                            if row["kind"] != "mri":
                                hashes[key] = _sha256(Path(path))
                row["index"] = len(self._sources[task])
                self._sources[task].append(row)
                self._records[task][split].append(row)
        hashes.update(load_vqa(cfg, self._records["vqa"]))
        self.counts = {task: {split: len(rows) for split, rows in splits.items()}
                       for task, splits in self._records.items()}
        for task, counts in manifest.get("counts", {}).items():
            if task not in self.counts:
                continue
            for split, expected in counts.items():
                if split not in self.counts[task] or self.counts[task][split] != expected:
                    raise ValueError(f"Prepared manifest count mismatch: {task}/{split}")
        labels = np.asarray([row["labels"] for row in self._records["classification"]["train"]], dtype=np.float32)
        if not len(labels):
            raise ValueError("Empty prepared classification training pool")
        self.pos_weight = torch.tensor(np.clip(
            (labels == 0).sum(0) / np.maximum((labels == 1).sum(0), 1), .25, 10), dtype=torch.float32)
        self.metadata = {
            "version": 2, "schema": manifest["schema"], "input_mode": "on_demand_source_no_pixel_cache",
            "tasks": list(TASKS), "counts": self.counts,
            "findings": list(FINDINGS), "source_sha256": hashes,
            "preparation_sources": manifest.get("source_sha256", {}),
            "patient_audit": "Computed after global holdout filtering",
            "classification": "Atlas source images; 13 CheXpert labels; unknown and uncertain labels masked",
            "segmentation": "Human-reviewed CXR organs and MRI tumor regions; no pseudo labels",
            "segmentation_layout": self.segmentation_layout,
            "segmentation_datasets": sorted({row["dataset"] for row in self._sources["segmentation"]}),
            "supervised_target_files": targets,
            "array_validation": "Manual source masks only; MRI volumes decoded on demand",
            "image_geometry": "512-square padded grayscale canvases with recorded even ROI boxes",
        }

    def dataset(self, task, split):
        if task not in TASKS:
            raise ValueError(f"Unsupported task: {task}")
        return TaskDataset(self, task, _split(split))

    @staticmethod
    def _vqa_image(path):
        with Image.open(path) as image:
            return ImageOps.pad(image.convert("RGB"), (512, 512),
                                method=Image.Resampling.BICUBIC, color="black")

    def _manual_record(self, row):
        kind = row.get("kind")
        if kind not in ("human_cxr", "montgomery", "mri") or "target_file" in row:
            raise ValueError("Prepared v2 requires manual masks and rejects CXAS/pseudo supervision")
        annotation = row.get("annotation", "")
        if annotation not in ("manual", "human", "expert_reviewed", "human_reviewed"):
            raise ValueError("Prepared v2 requires explicit manual or expert_reviewed annotation")
        source = str(row.get("annotation_source", "")).lower()
        if "pseudo" in source or "cxas" in source:
            raise ValueError("Prepared v2 rejects pseudo annotation sources")
        expected_channels = {"human_cxr": [0, 1], "montgomery": [0], "mri": [2, 3, 4, 5]}[kind]
        if row.get("channels") != expected_channels:
            raise ValueError("Prepared manual segmentation channels differ from the declared dataset")
        if row.get("target_names") != [self.segmentation_layout[i] for i in expected_channels]:
            raise ValueError("Prepared manual target_names differ from channel semantics")
        datasets = {"human_cxr": {"mimic_cxr_human"}, "montgomery": {"montgomery"},
                    "mri": {"ucsf_alptdg", "mu_glioma_post"}}
        if row.get("dataset") not in datasets[kind]:
            raise ValueError("Unknown prepared manual segmentation dataset")
        row["volume_id"] = str(row.get("volume_id", row["id"]))
        if not row["volume_id"]:
            raise ValueError("Prepared segmentation requires a volume identity")
        if kind == "mri":
            for key in ("t1ce_sha256", "mask_sha256"):
                digest = row.get(key)
                if (not isinstance(digest, str) or len(digest) != 64
                        or any(value not in "0123456789abcdef" for value in digest)):
                    raise ValueError("Prepared MRI requires source and mask SHA256 fingerprints")
            shape = row.get("canonical_shape")
            if (not isinstance(shape, list) or len(shape) != 3
                    or any(type(value) is not int or value <= 0 for value in shape)
                    or type(row.get("slice_index")) is not int or not 0 <= row["slice_index"] < shape[2]):
                raise ValueError("Prepared MRI slice must lie inside canonical_shape")
            if "volume_id" not in row or row["volume_id"] == row["id"]:
                raise ValueError("Prepared MRI requires a shared volume_id across its slices")
            row["mask_file"] = self._path(row["mask_file"])
        else:
            if not isinstance(row.get("masks"), list) or len(row["masks"]) != 2:
                raise ValueError("Prepared CXR requires two original manual mask files")
            if kind == "montgomery" and row["split"] != "human_test":
                raise ValueError("Montgomery is reserved for human_test")
            row["masks"] = [self._path(path) for path in row["masks"]]

    def _path(self, value):
        if not isinstance(value, str) or not value:
            raise ValueError("Prepared asset path must be a nonempty string")
        if value in self._checked_paths:
            return self._checked_paths[value]
        path = Path(value)
        path = path if path.is_absolute() else self.root / path
        if not path.is_file():
            raise FileNotFoundError(f"Prepared asset is missing or its symlink is broken: {path}")
        result = str(path.absolute())
        self._checked_paths[value] = result
        return result

    def _pixels(self, key, index):
        row = self._sources[key][index]
        if row.get("kind") == "mri":
            from .mri_pixels import mri_source_canvas
            pixels = mri_source_canvas(row)
        else:
            pixels = source_canvas(row)
        return (pixels[0] * 255).round().byte().numpy()

    def _example(self, task, split, row):
        if task == "vqa":
            return {"task": task, "split": split, "id": row["id"], "subject_id": str(row["subject_id"]),
                    "image": self._vqa_image(row["image"]), "question": row["question"], "answer": row["answer"],
                    "semantic_type": row.get("semantic_type", "diagnosis")}
        source = self._pixels(task, row["index"])
        pixels = torch.from_numpy(np.array(source, copy=True)).float()[None] / 255
        pixels = F.interpolate(pixels[None], (256, 256), mode="area")[0]
        result = {"task": task, "split": split, "id": row["id"], "subject_id": row["subject_id"],
                  "image": Image.fromarray(source).convert("RGB"), "pixels": pixels}
        if task == "classification":
            result["labels"] = torch.tensor(row["labels"], dtype=torch.float32)
            result["label_mask"] = (result["labels"] == 0) | (result["labels"] == 1)
        elif task == "segmentation":
            target = torch.zeros(len(self.segmentation_layout), 256, 256)
            if row["kind"] == "mri":
                from .mri_pixels import mri_target
                target = mri_target(row)
            else:
                local_targets = human_target(row)
                if row["kind"] == "montgomery":
                    local_targets = local_targets.amax(0, keepdim=True)
                target[row["channels"]] = local_targets
            if target.shape != (len(self.segmentation_layout), 256, 256) or not torch.isfinite(target).all():
                raise ValueError("Prepared manual target has invalid shape or nonfinite values")
            if torch.any((target < 0) | (target > 1)):
                raise ValueError("Prepared manual masks must contain probabilities in [0, 1]")
            mask = torch.zeros_like(target)
            y, x, height, width = [value // 2 for value in row["box"]]
            mask[row["channels"], y:y + height, x:x + width] = 1
            result.update(targets=target, mask=mask, segmentation_dataset=row["dataset"],
                          volume_id=row["volume_id"], active_channels=row["channels"], target_names=row["target_names"])
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
        if task == "segmentation":
            for key in ("segmentation_dataset", "volume_id", "active_channels", "target_names"):
                result[key] = [row[key] for row in examples]
        return result
