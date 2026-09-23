"""Compact CXR pair metadata with lazy source assets and no treatment inputs.

The selection's historical schema records how the cohort was selected. Training
reads its pair shards only; neither clinical tables nor evidence are dependencies.
"""
from collections.abc import Sequence
from datetime import datetime
import gzip
import json
import math
from pathlib import Path
import re

import numpy as np

from .protocol import _sha256


SELECTION_SCHEMA = "medworld-medication-selection-v1"
SPLITS = ("train", "validate", "test")
PAIR_DTYPE = np.dtype([("id", "S32"), ("source", "u4"), ("target", "u4"),
                       ("hours", "f8"), ("adjacent", "?")])


class CompactPairs(Sequence):
    """Keep one fixed-width row per pair; expand metadata only when requested."""

    def __init__(self, values, observations, split):
        self.values, self.observations, self.split = values, observations, split

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return CompactPairs(self.values[index], self.observations, self.split)
        value = self.values[index]
        source = self.observations[int(value["source"])]
        target = self.observations[int(value["target"])]
        result = {"id": bytes(value["id"]).decode("ascii"), "patient": source["patient"],
                  "split": self.split, "source": source["id"], "target": target["id"],
                  "realized_gap_hours": float(value["hours"]),
                  "full_timeline_adjacent": bool(value["adjacent"])}
        for side, observation in (("source", source), ("target", target)):
            for field, key in (("time", "timestamp"), ("view", "view"), ("study_id", "study_id"),
                               ("image", "image"), ("report", "report_file")):
                result[f"{side}_{field}"] = observation[key]
        return result

    def patients(self):
        return {self.observations[int(i)]["patient"] for i in np.unique(self.values["source"])}

    def filter_patients(self, holdouts):
        allowed = np.fromiter((holdouts[row["patient"]] == self.split for row in self.observations),
                              dtype=bool, count=len(self.observations))
        keep = allowed[self.values["source"]]
        excluded = {self.observations[int(i)]["patient"]
                    for i in np.unique(self.values["source"][~keep])}
        return CompactPairs(self.values[keep], self.observations, self.split), excluded


def read_selection(root, manifest):
    """Validate pair identities and return compact rows, unique endpoints, hashes."""
    if manifest.get("state") != "complete" or manifest.get("dry_run"):
        raise ValueError("Temporal selection must be complete and cannot be a dry run")
    cxr_root = Path(manifest["sources"]["cxr_root"]).resolve()
    if not cxr_root.is_dir():
        raise FileNotFoundError(f"Temporal CXR root missing: {cxr_root}")
    counts = manifest["cohort"]["splits"]
    if counts.get("unassigned", {}).get("retained", 0):
        raise ValueError("Temporal selection has patients without a split")
    observations, indices, times, patients = [], {}, [], {}
    pairs, source_hashes = {}, {"manifest.json": _sha256(root / "manifest.json")}

    def endpoint(row, side, patient, split):
        identity, study = row[side], str(row[f"{side}_study_id"])
        if not isinstance(identity, str) or not identity.startswith("cxr:") or not study.isdigit():
            raise ValueError("Invalid temporal CXR observation identity")
        dicom = identity.removeprefix("cxr:")
        if not re.fullmatch(r"[a-zA-Z0-9-]+", dicom):
            raise ValueError("Invalid temporal image identity")
        base = f"files/p{patient[:2]}/p{patient}"
        image, report = row[f"{side}_image"], row[f"{side}_report"]
        if image != f"{base}/s{study}/{dicom}.jpg" or report != f"{base}/s{study}.txt":
            raise ValueError("Temporal image/report paths violate patient or study ownership")
        view, timestamp = row[f"{side}_view"], row[f"{side}_time"]
        if view not in ("AP", "PA"):
            raise ValueError("Temporal CXR endpoint must use a frontal view")
        observation = {"id": identity, "patient": patient, "split": split, "image": image,
                       "report_file": report, "study_id": study, "timestamp": timestamp, "view": view}
        if identity in indices:
            index = indices[identity]
            if observations[index] != observation:
                raise ValueError("Inconsistent temporal observation metadata or patient/split")
            return index
        # Check symlink containment once per unique endpoint, without decoding it.
        for asset in (image, report):
            if not (cxr_root / asset).resolve().is_relative_to(cxr_root):
                raise ValueError("Temporal asset escapes the CXR root")
        time = datetime.fromisoformat(timestamp)
        if time.tzinfo is not None:
            raise ValueError("Expected native CXR acquisition times without a timezone")
        index = len(observations)
        indices[identity] = index
        observations.append(observation)
        times.append(time)
        return index

    for split in SPLITS:
        name = f"{split}.jsonl.gz"
        path = root / name
        expected = manifest["outputs"].get(name, {})
        digest = _sha256(path)
        if digest != expected.get("sha256") or path.stat().st_size != expected.get("bytes"):
            raise ValueError(f"Temporal selection fingerprint mismatch: {name}")
        source_hashes[name] = digest
        chunks, pending = [], []
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                identity, patient = row["id"], str(row["patient"])
                if not patient.isdigit() or row["split"] != split:
                    raise ValueError("Temporal pair has an invalid patient or split")
                patient = patients.setdefault(patient, (patient, split))[0]
                if patients[patient][1] != split:
                    raise ValueError("Temporal pairs cross patient split membership")
                if not isinstance(identity, str) or not re.fullmatch(r"medpair:[0-9a-f]{24}", identity):
                    raise ValueError("Invalid temporal pair identity")
                source = endpoint(row, "source", patient, split)
                target = endpoint(row, "target", patient, split)
                gap = float(row["hours"])
                actual_gap = (times[target] - times[source]).total_seconds() / 3600
                if (source == target or not math.isfinite(gap) or gap <= 0
                        or not math.isclose(gap, actual_gap, rel_tol=1e-10, abs_tol=1e-8)):
                    raise ValueError("Temporal pair must match its positive actual acquisition interval")
                adjacent = row["full_timeline_adjacent"]
                if type(adjacent) is not bool:
                    raise ValueError("Invalid temporal adjacency flag")
                pending.append((identity.encode("ascii"), source, target, gap, adjacent))
                if len(pending) == 8192:
                    chunks.append(np.asarray(pending, dtype=PAIR_DTYPE))
                    pending.clear()
        if pending:
            chunks.append(np.asarray(pending, dtype=PAIR_DTYPE))
        values = np.concatenate(chunks) if chunks else np.empty(0, dtype=PAIR_DTYPE)
        if len(values) != counts[split].get("retained", 0):
            raise ValueError(f"Temporal selection count mismatch: {split}")
        pairs[split] = CompactPairs(values, observations, split)
    identities = np.concatenate([rows.values["id"] for rows in pairs.values()])
    if len(np.unique(identities)) != len(identities):
        raise ValueError("Duplicate temporal pair identity")
    if (len(identities) != manifest["cohort"]["retained_pairs"]
            or len(patients) != manifest["cohort"]["retained_patients"]):
        raise ValueError("Temporal selection cohort count mismatch")
    return observations, pairs, source_hashes, cxr_root
