"""Prepared future-task labels with a strict source-only prediction boundary."""
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from .protocol import _sha256

SCHEMA = "medworld-future-tasks-v1"
FUTURE_TASKS = ("future_vqa", "progression", "future_report", "mortality_30d", "remaining_los")
SPLITS = ("train", "validate", "test")
ROW_DTYPE = np.dtype([("source", "<u4"), ("target", "<i4"), ("query", "<i4"),
                      ("hours", "<f8"), ("label", "<f8")])
PROGRESSION_CLASSES = ("improved", "stable", "worsened")


def truncate_utf8(value, limit):
    return value.encode("utf-8")[:limit].decode("utf-8", errors="ignore")


class FutureRows(Sequence):
    """Fixed-width labels; expand provenance only for requested examples."""
    def __init__(self, data, task, split, values):
        self.data, self.task, self.split, self.values = data, task, split, values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        value = self.values[index]
        source = self.data.observations[int(value["source"])]
        row = {"id": f"{self.task}:{self.split}:{index}", "task": self.task, "split": self.split,
               "patient": source["patient"], "subject_id": source["patient"],
               "source": source["id"], "source_study_id": source["study_id"],
               "source_image": source["image"], "source_report_file": source["report_file"],
               "source_time": source["timestamp"], "delta_hours": float(value["hours"]),
               "label": float(value["label"]), "atlas_url": source["atlas_url"]}
        if int(value["target"]) >= 0:
            target = self.data.observations[int(value["target"])]
            row.update(target=target["id"], target_study_id=target["study_id"],
                       target_report_file=target["report_file"], target_time=target["timestamp"])
        if int(value["query"]) >= 0:
            row.update(self.data.queries[int(value["query"])])
        return row


class FutureData:
    @staticmethod
    def extra_holdouts(root):
        """Read patient reservations before UnifiedData filters any task stream."""
        manifest = json.loads((Path(root) / "manifest.json").read_text())
        if manifest.get("schema") != SCHEMA or manifest.get("state") != "complete":
            raise ValueError("Future tasks require a completed supported manifest")
        values = manifest.get("extra_holdouts", {})
        if any(not str(p).isdigit() or s not in ("validate", "test") for p, s in values.items()):
            raise ValueError("Invalid future gold patient reservations")
        return dict(values)

    def __init__(self, cfg, base_data_fingerprint, holdouts=None):
        self.cfg = cfg
        self.root = Path(cfg["future_data"])
        path = self.root / "manifest.json"
        manifest = json.loads(path.read_text())
        if manifest.get("schema") != SCHEMA or manifest.get("state") != "complete":
            raise ValueError("Future tasks require a completed supported manifest")
        if manifest["base_data_fingerprint"] != base_data_fingerprint:
            raise ValueError("Future-task labels were built for a different base cohort")
        if cfg.get("future_source_bytes", 383) != manifest["source_report_bytes"]:
            raise ValueError("Future source-report budget differs from prepared protocol")
        self.cxr_root = Path(manifest["cxr_root"]).resolve()
        for name, expected in manifest["files"].items():
            file = self.root / name
            if file.stat().st_size != expected["bytes"] or _sha256(file) != expected["sha256"]:
                raise ValueError(f"Future-task manifest fingerprint mismatch: {name}")
        with (self.root / "observations.jsonl").open() as stream:
            self.observations = [json.loads(line) for line in stream]
        with (self.root / "queries.jsonl").open() as stream:
            self.queries = [json.loads(line) for line in stream]
        ownership = {}
        for obs in self.observations:
            patient = str(obs["patient"])
            split = obs["split"]
            if ownership.setdefault(patient, split) != split:
                raise ValueError("Future observations cross patient holdouts")
            if holdouts is not None and patient in holdouts and holdouts[patient] != split:
                raise ValueError("Future observations violate the base patient holdout")
            for name in ("image", "report_file"):
                asset = self.cxr_root / obs[name]
                if not asset.resolve().is_relative_to(self.cxr_root):
                    raise ValueError("Future asset escapes the original CXR root")
        # Validate millions of label rows without reparsing the same timestamps
        # once for every region, finding and VQA query in an examination pair.
        observation_times = np.asarray([o["timestamp"] for o in self.observations], dtype="datetime64[us]")
        observation_patients = np.asarray([str(o["patient"]) for o in self.observations])
        observation_splits = np.asarray([o["split"] for o in self.observations])
        self._rows = {}
        for task in FUTURE_TASKS:
            self._rows[task] = {}
            for split in SPLITS:
                values = np.load(self.root / f"{task}_{split}.npy", allow_pickle=False)
                if values.dtype != ROW_DTYPE or len(values) != manifest["counts"][task][split]:
                    raise ValueError("Future row type/count differs from the manifest")
                if len(values):
                    if (values["source"].max() >= len(self.observations)
                            or values["target"].min() < -1 or values["target"].max() >= len(self.observations)
                            or values["query"].min() < -1 or values["query"].max() >= len(self.queries)
                            or not np.isfinite(values["hours"]).all() or not np.isfinite(values["label"]).all()):
                        raise ValueError("Invalid future-task row references or labels")
                    if task == "progression" and not np.isin(values["label"], (0, 1, 2)).all():
                        raise ValueError("Progression requires the three declared categorical labels")
                    if task == "mortality_30d" and not np.isin(values["label"], (0, 1)).all():
                        raise ValueError("Mortality requires binary labels")
                    if task == "remaining_los" and (values["label"] < 0).any():
                        raise ValueError("Remaining LOS cannot be negative")
                    if task in ("future_vqa", "progression") and (values["query"] < 0).any():
                        raise ValueError("Question-conditioned future tasks require query references")
                    source_ids = values["source"]
                    if (observation_splits[source_ids] != split).any():
                        raise ValueError("Future task split disagrees with observation split")
                    if task in ("future_vqa", "progression", "future_report"):
                        if (values["target"] < 0).any() or (values["hours"] <= 0).any():
                            raise ValueError("Longitudinal tasks require a positive observed future")
                        target_ids = values["target"]
                        gaps = (observation_times[target_ids] - observation_times[source_ids]) / np.timedelta64(1, "h")
                        if ((observation_patients[source_ids] != observation_patients[target_ids]).any()
                                or not np.isclose(gaps, values["hours"], atol=1e-6, rtol=0).all()):
                            raise ValueError("Future pair crosses patient identity or time")
                    elif (values["target"] != -1).any() or not (values["hours"] == (720 if task == "mortality_30d" else 24)).all():
                        raise ValueError("Outcome tasks cannot depend on a future examination or realized horizon")
                self._rows[task][split] = FutureRows(self, task, split, values)
        self.metadata = {"manifest_sha256": _sha256(path), **manifest}
        self.fingerprint = hashlib.sha256(json.dumps(self.metadata, sort_keys=True).encode()).hexdigest()
        self._pool = None

    def rows(self, task, split):
        if split == "human_test":
            return []
        return self._rows[task][split]

    def _source(self, row):
        with Image.open(self.cxr_root / row["source_image"]) as image:
            image = ImageOps.pad(image.convert("RGB"), (512, 512), method=Image.Resampling.BICUBIC, color="black")
        report = (self.cxr_root / row["source_report_file"]).read_text(encoding="utf-8")
        if not report.strip():
            raise ValueError("Future input report is empty")
        return image, truncate_utf8(report, self.cfg.get("future_source_bytes", 383))

    def batch(self, task, split, indices, *, source_only=False):
        rows = [self.rows(task, split)[i] for i in indices]
        if not rows:
            raise ValueError("Empty future task batch")
        workers = self.cfg.get("image_workers", 1)
        if workers > 1 and self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="future-source")
        sources = list(self._pool.map(self._source, rows)) if self._pool else [self._source(r) for r in rows]
        batch = {"task": task, "ids": [r["id"] for r in rows], "subject_ids": [r["patient"] for r in rows],
                 "images": [x[0] for x in sources], "reports": [x[1] for x in sources],
                 "delta_hours": torch.tensor([r["delta_hours"] for r in rows], dtype=torch.float32)}
        if task in ("future_vqa", "progression"):
            batch["questions"] = [r["question"] for r in rows]
        if not source_only:
            if task == "future_report":
                batch["answers"] = [(self.cxr_root / r["target_report_file"]).read_text(encoding="utf-8") for r in rows]
                if not all(value.strip() for value in batch["answers"]):
                    raise ValueError("Empty future report supervision")
            elif task == "future_vqa":
                batch["answers"] = [r["answer"] for r in rows]
            else:
                dtype = torch.long if task == "progression" else torch.float32
                batch["labels"] = torch.tensor([r["label"] for r in rows], dtype=dtype)
        return batch
