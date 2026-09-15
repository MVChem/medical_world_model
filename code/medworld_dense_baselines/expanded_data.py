"""Expand the fixed dense training cohort while preserving all held-out rows.

Uses only already prepared, independently QC-approved frontal images and their
existing pseudo masks. Existing row indices remain stable, including grounding
queries; additional training rows are appended deterministically.
"""
from __future__ import annotations

import argparse
import collections
import copy
import os
from pathlib import Path

import numpy as np

from common import OLD, PROJECT, atomic, digest, rank, read_rows, write_rows
from prepare import cache_lr


def prepare_expanded(run, source, train_count=18708):
    import json

    os.umask(0o077)
    run, source = Path(run).resolve(), Path(source).resolve()
    source = source / "data" if (source / "data/manifest.json").exists() else source
    data = run / "data"
    data.mkdir(parents=True, exist_ok=True)
    selection_path = PROJECT / "code/medworld_stage1/data/slot44_20260911_derived_v2/selection.json"
    inputs = {str(p): digest(p) for p in (source / "manifest.json", source / "observations.jsonl",
                                         OLD / "observations.jsonl", OLD / "seg_valid_qc.npy", selection_path)}
    requested = dict(source=str(source), train_count=train_count, source_sha256=inputs)
    if (data / "manifest.json").exists():
        saved = json.loads((data / "manifest.json").read_text())
        if saved.get("expansion", {}).get("request") != requested:
            raise ValueError("Existing expanded data differs from requested cohort")
        print(json.dumps(saved["coverage"], indent=2), flush=True)
        return saved

    base_manifest = json.loads((source / "manifest.json").read_text())
    if digest(source / "observations.jsonl") != base_manifest["cohort_sha256"]:
        raise ValueError("Source frozen cohort manifest mismatch")
    original = read_rows(source / "observations.jsonl")
    rows = copy.deepcopy(original)
    old_rows = read_rows(OLD / "observations.jsonl")
    valid = np.load(OLD / "seg_valid_qc.npy")
    overrides = json.loads(selection_path.read_text())["patient_split_overrides"]
    candidates = sorted((r for r in old_rows if overrides.get(r["subject_id"], r["split"]) == "train"
                         and r["view"] in ("AP", "PA") and valid[r["index"]]), key=lambda r: rank(r["id"]))
    if train_count < 1 or train_count > len(candidates):
        raise ValueError(f"train-count must be 1..{len(candidates)}")
    selected = candidates[:train_count]
    previous_train = {r["id"] for r in original if r["split"] == "train" and "sr" in r["tasks"]}
    if not previous_train <= {r["id"] for r in selected}:
        raise ValueError("Requested cohort must contain every existing training image")
    by_id = {r["id"]: i for i, r in enumerate(rows)}
    changed = []
    for row in selected:
        if row["id"] in previous_train:
            continue
        entry = dict(id=row["id"], subject_id=row["subject_id"], image=row["image"],
                     old_index=row["index"], box=row["box"], split="train", kind="mimic",
                     tasks=["segmentation", "sr"])
        if row["id"] in by_id:
            index = by_id[row["id"]]
            if rows[index]["split"] != "grounding_only":
                raise ValueError("Expansion attempts to overwrite an existing task cohort")
            rows[index] = dict(entry, index=index)
        else:
            index = len(rows)
            rows.append(dict(entry, index=index))
            by_id[row["id"]] = index
        changed.append(index)

    for split in ("validate", "test", "human_test"):
        before = [r for r in original if r["split"] == split]
        after = [r for r in rows if r["split"] == split]
        if before != after:
            raise AssertionError(f"Held-out {split} rows changed")
    sets = {s: {str(r["subject_id"]) for r in rows if r["split"] == s and "segmentation" in r["tasks"]}
            for s in ("train", "validate", "test", "human_test")}
    for a, patients in sets.items():
        for b, others in sets.items():
            if a != b and patients & others:
                raise AssertionError(f"Patient leakage between {a} and {b}")

    images = np.lib.format.open_memmap(data / "images.npy", mode="w+", dtype=np.uint8,
                                      shape=(len(rows), 512, 512))
    base_images = np.load(source / "images.npy", mmap_mode="r")
    old_images = np.load(OLD / "images.npy", mmap_mode="r")
    images[:len(original)] = base_images
    for i in changed:
        images[i] = old_images[rows[i]["old_index"]]
    images.flush()
    for filename in ("human_masks.npy", "grounding.jsonl", "query_vocab.json", "montgomery"):
        target = source / filename
        link = data / filename
        if target.exists() and not link.exists():
            link.symlink_to(target, target_is_directory=target.is_dir())
    cache_lr(data)
    # The entire original held-out pixel input remains exactly unchanged.
    lr = np.load(data / "lr_images.npy", mmap_mode="r")
    base_lr = np.load(source / "lr_images.npy", mmap_mode="r")
    holdout = [r["index"] for r in original if r["split"] in ("validate", "test", "human_test")]
    if not np.array_equal(images[holdout], base_images[holdout]) or not np.array_equal(lr[holdout], base_lr[holdout]):
        raise AssertionError("Held-out image or LR pixels changed")
    write_rows(data / "observations.jsonl", rows)
    manifest = copy.deepcopy(base_manifest)
    manifest.update(images=len(rows), cohort_sha256=digest(data / "observations.jsonl"),
                    image_sha256=digest(data / "images.npy"), lr_image_sha256=digest(data / "lr_images.npy"))
    manifest["coverage"]["images"] = dict(collections.Counter(r["split"] for r in rows))
    manifest["coverage"]["dense_patients"] = {s: len(p) for s, p in sets.items()}
    manifest["expansion"] = dict(request=requested, available_qc_frontal_train=len(candidates),
                                  previous_train_count=len(previous_train), train_count=len(selected),
                                  additional_training_images=len(changed),
                                  heldout_rows_and_pixels_exactly_preserved=True,
                                  training_selection="same rank(seed=20260912,id) and patient split overrides",
                                  targets="existing CXAS pseudo masks indexed by old_index; no new target fitting",
                                  pseudo_path=str(OLD / "seg_probs.npy"))
    atomic(data / "manifest.json", manifest)
    print(json.dumps(manifest["coverage"], indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=PROJECT / "code/medworld_dense_baselines/runs/dense_20260912/data")
    parser.add_argument("--train-count", type=int, default=18708)
    args = parser.parse_args()
    prepare_expanded(args.run, args.source, args.train_count)
