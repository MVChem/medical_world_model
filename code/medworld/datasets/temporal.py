"""Signed actual-time pairs, with a source-only inference boundary."""
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps
import torch

from .current import _rows, _sha256

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
    return result


class TemporalData:
    def __init__(self, root, bidirectional=True, image_workers=1):
        self.root = Path(root)
        self.observations = _rows(self.root / "observations.jsonl")
        self.lookup = {row["id"]: row for row in self.observations}
        if len(self.lookup) != len(self.observations):
            raise ValueError("Duplicate temporal observation ID")
        self.pairs = {split: _rows(self.root / f"{split}.jsonl") for split in SPLITS}
        self.bidirectional = bidirectional
        self.image_workers = image_workers
        self._image_pool = None
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
                    if len(obs["labels"]) != 6 or not obs["report"].strip():
                        raise ValueError("Temporal supervision requires report and six finding labels")
                directed_pair(row)  # validate actual time independently of the old horizon bucket
        self.source_hashes = {name: _sha256(self.root / name) for name in
                              ("observations.jsonl", "train.jsonl", "validate.jsonl", "test.jsonl")}

    def directed(self, split):
        return [directed_pair(row, reverse) for row in self.pairs[split]
                for reverse in ((False, True) if self.bidirectional else (False,))]

    def _observation(self, observation_id):
        row = self.lookup[observation_id]
        path = Path(row["image"])
        if not path.is_absolute():
            path = self.root / path
        with Image.open(path) as image:
            image = ImageOps.pad(image.convert("RGB"), (512, 512),
                                 method=Image.Resampling.BICUBIC, color="black")
        return image, row["report"]

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
            result.update(target={"images": [o[0] for o in target], "texts": [o[1] for o in target]},
                          report_targets=[o[1] for o in target],
                          labels=torch.tensor([self.lookup[row["target"]]["labels"] for row in rows],
                                              dtype=torch.float32))
        return result
