"""Signed actual-time pairs, with a source-only inference boundary."""
import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import json
import operator
from pathlib import Path

from PIL import Image, ImageOps
import torch

from .protocol import _rows, _sha256
from .temporal_selection import CompactPairs, SELECTION_SCHEMA, read_selection

SPLITS = ("train", "validate", "test")


def directed_pair(row, reverse=False):
    gap = float(row["realized_gap_hours"])
    if not math.isfinite(gap) or gap <= 0:
        raise ValueError("Expected a positive actual interval in the original pair")
    result = dict(row, id=row["id"] + (":backward" if reverse else ":forward"),
                  original_id=row["id"], direction="backward" if reverse else "forward",
                  delta_hours=-gap if reverse else gap)
    if reverse:
        result["source"], result["target"] = row["target"], row["source"]
        for key in row:
            if key.startswith("source_") and (other := "target_" + key[7:]) in row:
                result[key], result[other] = row[other], row[key]
    return result


class DirectedPairs(Sequence):
    """Expose forward/backward pairs without duplicating the cohort in memory."""

    def __init__(self, pairs, bidirectional):
        self.pairs, self.directions = pairs, 2 if bidirectional else 1

    def __len__(self):
        return len(self.pairs) * self.directions

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        pair, direction = divmod(index, self.directions)
        return directed_pair(self.pairs[pair], bool(direction))


class TemporalData:
    def __init__(self, root, bidirectional=True, image_workers=1):
        self.root = Path(root)
        self.bidirectional = bidirectional
        self.image_workers = image_workers
        self._image_pool = None
        self.asset_root = self.root
        self.schema = "prepared-temporal"
        manifest_path = self.root / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("schema") != SELECTION_SCHEMA:
                raise ValueError("Unsupported temporal selection manifest schema")
            self.schema = manifest["schema"]
            self.observations, self.pairs, self.source_hashes, self.asset_root = read_selection(self.root, manifest)
            self.lookup = {row["id"]: row for row in self.observations}
            return
        self.observations = _rows(self.root / "observations.jsonl")
        self.lookup = {row["id"]: row for row in self.observations}
        if len(self.lookup) != len(self.observations):
            raise ValueError("Duplicate temporal observation ID")
        self.pairs = {split: _rows(self.root / f"{split}.jsonl") for split in SPLITS}
        seen, patients = set(), {}
        for observation in self.observations:
            patient, split = str(observation["patient"]), observation["split"]
            if split not in SPLITS or patients.setdefault(patient, split) != split:
                raise ValueError("Temporal observations violate patient split membership")
        for split, rows in self.pairs.items():
            for row in rows:
                if row["id"] in seen or row["split"] != split or row["source"] == row["target"]:
                    raise ValueError("Invalid temporal pair identity or split")
                seen.add(row["id"])
                for side in ("source", "target"):
                    obs = self.lookup[row[side]]
                    if obs["split"] != split or str(obs["patient"]) != str(row["patient"]):
                        raise ValueError("Temporal pair crosses patient or split")
                    if not obs["report"].strip():
                        raise ValueError("Temporal observations require their own nonempty report")
                directed_pair(row)  # Validate the signed actual time interval.
        self.source_hashes = {name: _sha256(self.root / name) for name in
                              ("observations.jsonl", "train.jsonl", "validate.jsonl", "test.jsonl")}

    def directed(self, split):
        return DirectedPairs(self.pairs[split], self.bidirectional)

    def filter_holdouts(self, holdouts):
        before, after, excluded = {}, {}, {}
        for split, rows in self.pairs.items():
            if isinstance(rows, CompactPairs):
                kept, excluded_patients = rows.filter_patients(holdouts)
            else:
                kept = [r for r in rows if holdouts[str(r["patient"])] == split]
                excluded_patients = {str(r["patient"]) for r in rows
                                     if holdouts[str(r["patient"])] != split}
            before[split], after[split], excluded[split] = len(rows), len(kept), len(excluded_patients)
            self.pairs[split] = kept
        return before, after, excluded

    def patient_splits(self):
        for split, rows in self.pairs.items():
            patients = rows.patients() if isinstance(rows, CompactPairs) else {str(r["patient"]) for r in rows}
            for patient in patients:
                yield patient, split

    def _observation(self, observation_id):
        row = self.lookup[observation_id]
        path = Path(row["image"])
        if not path.is_absolute():
            path = self.asset_root / path
        with Image.open(path) as image:
            image = ImageOps.pad(image.convert("RGB"), (512, 512),
                                 method=Image.Resampling.BICUBIC, color="black")
        report = ((self.asset_root / row["report_file"]).read_text(encoding="utf-8")
                  if "report_file" in row else row["report"])
        if not report.strip():
            raise ValueError(f"Temporal observation has an empty report: {observation_id}")
        return image, report

    def batch(self, rows, *, source_only=False):
        if not rows:
            raise ValueError("Empty temporal batch")
        # source_only does not even look up the other endpoint. Ground truth is
        # obtained by an evaluator only after predictions have been returned.
        if self.image_workers > 1 and self._image_pool is None:
            self._image_pool = ThreadPoolExecutor(max_workers=self.image_workers, thread_name_prefix="temporal-image")
        def observations(side):
            ids = [row[side] for row in rows]
            return list(self._image_pool.map(self._observation, ids)) if self._image_pool else [self._observation(i) for i in ids]
        source = observations("source")
        result = {"source": {"images": [o[0] for o in source], "texts": [o[1] for o in source]},
                  "delta_hours": torch.tensor([row["delta_hours"] for row in rows], dtype=torch.float32)}
        if not source_only:
            target = observations("target")
            result["target"] = {"images": [o[0] for o in target], "texts": [o[1] for o in target]}
        return result
