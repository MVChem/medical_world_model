"""Matched Table 1 pairs with an explicit current-evidence inference boundary.

The V-JEPA array is a frozen-feature cache. Native Qwen always receives pixels
read from the current image, independently of this array. Reports are tokenized
lazily: requesting a source-only batch never tokenizes target evidence.
"""
from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch


SPLITS = ("train", "validate", "test")
SOURCE_KEYS = frozenset({
    "source_ids", "source_mask", "source_features", "_source_images", "horizon",
    "donor_ids", "donor_mask", "donor_features", "donor_images",
})


def source_view(batch):
    """Return the model's prediction inputs; supervision never crosses here."""
    return {key: value for key, value in batch.items() if key in SOURCE_KEYS}


def _rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Corpus:
    """Use the saved cohort without reselection or resplitting.

    ``batch(rows, source_only=True)`` is the inference API. In a training batch,
    ``*_ids`` are encoder context (report plus independently budgeted EHR), and
    ``*_target_ids`` are report-only causal-language targets, terminated by EOS.
    Image lists hold 512-square PIL RGB images. Only tensor values move to device.
    """

    def __init__(self, cfg, tokenizer):
        self.cfg = dict(cfg)
        self.tokenizer = tokenizer
        self.root = Path(cfg["cache"])
        self.observations = _rows(self.root / "observations.jsonl")
        self.lookup = {row["id"]: i for i, row in enumerate(self.observations)}
        if len(self.lookup) != len(self.observations):
            raise ValueError("Observation IDs must be unique")
        self.pairs = {split: _rows(self.root / f"{split}.jsonl") for split in SPLITS}
        self._audit = self._audit_patients()
        self.features = np.load(self.root / "vjepa_features.npy", mmap_mode="r")
        if self.features.ndim != 3 or len(self.features) != len(self.observations):
            raise ValueError("Frozen features must have one [tokens, width] row per observation")
        self.stage1 = [i for i, row in enumerate(self.observations) if row["split"] == "train"]
        self._tokens = {}
        self._contexts = {}
        self._permutations = {}
        self.donors = self._make_donors() if cfg.get("state_condition", cfg.get("condition")) == "shuffled" else {}
        self.metadata = self._metadata()

    source_view = staticmethod(source_view)

    def _audit_patients(self):
        by_split = {split: set() for split in SPLITS}
        for row in self.observations:
            if row["split"] not in by_split:
                raise ValueError("Unknown observation split")
            by_split[row["split"]].add(str(row["patient"]))
        intersections = {
            f"{left}:{right}": len(by_split[left] & by_split[right])
            for left, right in combinations(SPLITS, 2)
        }
        if any(intersections.values()):
            raise ValueError("Patient leakage across official splits")
        seen_pair_ids = set()
        for split, rows in self.pairs.items():
            for row in rows:
                if row["id"] in seen_pair_ids:
                    raise ValueError("Pair IDs must be unique across splits")
                seen_pair_ids.add(row["id"])
                if row["split"] != split:
                    raise ValueError("Pair split disagrees with its file")
                for side in ("source", "target"):
                    if row[side] not in self.lookup:
                        raise ValueError("Pair references an unknown observation")
                    obs = self.observations[self.lookup[row[side]]]
                    if obs["split"] != split or str(obs["patient"]) != str(row["patient"]):
                        raise ValueError("Pair observation violates patient/split membership")
                if row["source"] == row["target"]:
                    raise ValueError("Forecast pairs must have distinct source and target observations")
                if int(row["horizon"]) < 0:
                    raise ValueError("Forecast horizon must be nonnegative")
        return {"cross_split_patient_intersections": intersections,
                "observation_patients": {split: len(patients) for split, patients in by_split.items()},
                "pair_membership_checked": True}

    def _make_donors(self):
        donors = {}
        for split, rows in self.pairs.items():
            for row in rows:
                position = int(hashlib.sha256(
                    f"{self.cfg['seed']}:state-donor:{row['id']}".encode()).hexdigest(), 16) % len(rows)
                for offset in range(len(rows)):
                    donor = rows[(position + offset) % len(rows)]
                    if str(donor["patient"]) != str(row["patient"]):
                        donors[row["id"]] = donor["source"]
                        break
                else:
                    raise ValueError(f"Shuffled state requires at least two patients in {split}")
        return donors

    def _metadata(self):
        names = ["observations.jsonl", *(f"{split}.jsonl" for split in SPLITS),
                 "manifest.json", "features.json", "ehr_audit.jsonl", "selected_pair_audit.jsonl"]
        hashes = {name: _sha256(self.root / name) for name in names if (self.root / name).exists()}
        feature_path = self.root / "features.json"
        feature_provenance = json.loads(feature_path.read_text()) if feature_path.exists() else {}
        declared_observations = feature_provenance.get("observations_sha256")
        if declared_observations and declared_observations != hashes["observations.jsonl"]:
            raise ValueError("Feature cache observation manifest SHA256 mismatch")
        if "shape" in feature_provenance and list(self.features.shape) != feature_provenance["shape"]:
            raise ValueError("Feature cache shape disagrees with its provenance")
        return {
            "cache": str(self.root.resolve()),
            "counts": {split: {"pairs": len(rows), "patients": len({str(r['patient']) for r in rows})}
                       for split, rows in self.pairs.items()},
            "stage1_observations": len(self.stage1), "source_file_sha256": hashes,
            "patient_audit": self._audit,
            "frozen_features": {"path": str((self.root / 'vjepa_features.npy').resolve()),
                                "shape": list(self.features.shape), "dtype": str(self.features.dtype),
                                "bytes_on_disk": (self.root / 'vjepa_features.npy').stat().st_size,
                                "array_content_sha256_recomputed": False,
                                "provenance": feature_provenance},
            "native_image_preprocessing": "Raw image read; RGB; 512x512 aspect-preserving bicubic black letterbox",
            "report_tokens": self.cfg["report_tokens"], "ehr_tokens": self.cfg.get("ehr_tokens"),
            "use_ehr": bool(self.cfg.get("use_ehr")),
            "ehr_cutoff": "Inherited saved cohort: occurrence and store time at or before the observation image",
            "prediction_input_keys": sorted(SOURCE_KEYS),
            "future_evidence": "Training target branch and report/label supervision only",
        }

    def _report(self, index):
        if index not in self._tokens:
            tokens = self.tokenizer.encode(self.observations[index]["report"], add_special_tokens=False)
            self._tokens[index] = tokens[:self.cfg["report_tokens"]]
        return self._tokens[index]

    def _context(self, index):
        if index not in self._contexts:
            report = self._report(index)
            if self.cfg.get("use_ehr"):
                ehr = self.tokenizer.encode(self.observations[index]["ehr_text"], add_special_tokens=False)
                if len(ehr) > self.cfg["ehr_tokens"]:
                    raise ValueError("EHR serialization exceeded its independent whole-line token budget")
                report_header = self.tokenizer.encode("Current radiograph report:\n", add_special_tokens=False)
                separator = self.tokenizer.encode("\n\n", add_special_tokens=False)
                context = report_header + report + separator + ehr
            else:
                context = report
            self._contexts[index] = context
        return self._contexts[index]

    def pad(self, sequences, *, left=False):
        if not sequences:
            raise ValueError("Cannot batch zero observations")
        width = max(1, max(map(len, sequences)))
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must supply a pad or EOS token")
        ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, seq in enumerate(sequences):
            if seq:
                section = slice(width - len(seq), width) if left else slice(0, len(seq))
                ids[i, section] = torch.as_tensor(seq, dtype=torch.long)
                mask[i, section] = 1
        return ids, mask

    def _image(self, index):
        path = Path(self.observations[index]["image"])
        if not path.is_absolute():
            path = self.root / path
        with Image.open(path) as image:
            return ImageOps.pad(image.convert("RGB"), (512, 512),
                                method=Image.Resampling.BICUBIC, color="black")

    def batch(self, rows, device="cuda", stage1=False, source_only=False):
        if not rows:
            raise ValueError("Cannot batch zero pairs")
        if stage1:
            rows = [dict(source=self.observations[i]["id"], target=self.observations[i]["id"], horizon=0)
                    for i in rows]
        batch = {}
        for side in (("source",) if source_only else ("source", "target")):
            indices = [self.lookup[row[side]] for row in rows]
            batch[side + "_ids"], batch[side + "_mask"] = self.pad(
                [self._context(index) for index in indices], left=True)
            batch[side + "_features"] = torch.from_numpy(np.array(self.features[indices], copy=True))
            batch["_" + side + "_images"] = [self._image(index) for index in indices]
            if not source_only:
                if self.tokenizer.eos_token_id is None:
                    raise ValueError("Report targets require an EOS token")
                batch[side + "_target_ids"], batch[side + "_target_mask"] = self.pad(
                    [self._report(index) + [self.tokenizer.eos_token_id] for index in indices])
                batch[side + "_labels"] = torch.tensor([self.observations[index]["labels"] for index in indices])
        if self.donors and not stage1:
            indices = [self.lookup[self.donors[row["id"]]] for row in rows]
            batch["donor_ids"], batch["donor_mask"] = self.pad([self._context(i) for i in indices], left=True)
            batch["donor_features"] = torch.from_numpy(np.array(self.features[indices], copy=True))
            batch["donor_images"] = [self._image(index) for index in indices]
        batch["horizon"] = torch.tensor([row["horizon"] for row in rows], dtype=torch.long)
        return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

    def training_batch(self, stage, step, micro, device="cuda"):
        """Reproduce the old deterministic stream, including after resume."""
        if stage not in (1, 2) or step < 0 or micro < 0:
            raise ValueError("Expected stage 1/2 and nonnegative step/micro")
        population = self.stage1 if stage == 1 else self.pairs["train"]
        if not population:
            raise ValueError("Training population is empty")
        if self.cfg.get("sampling") == "permutation":
            start = (step * self.cfg["gradient_accumulation"] + micro) * self.cfg["batch_size"]
            chosen = []
            for position in range(start, start + self.cfg["batch_size"]):
                epoch, offset = divmod(position, len(population))
                key = (stage, epoch)
                if key not in self._permutations:
                    self._permutations = {key: np.random.default_rng(
                        np.random.SeedSequence([self.cfg["seed"], stage, epoch])).permutation(len(population))}
                chosen.append(self._permutations[key][offset])
        else:
            rng = np.random.default_rng(np.random.SeedSequence([self.cfg["seed"], stage, step, micro]))
            chosen = rng.choice(len(population), self.cfg["batch_size"], replace=False)
        return self.batch([population[i] for i in chosen], device=device, stage1=stage == 1)
