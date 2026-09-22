"""Retain immutable segmentation annotations and link official CXR-VQA data.

Historical image arrays and encoder caches are never opened. Only the existing
CXAS annotation array's header is inspected; all image inputs remain source files.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import ijson
import numpy as np


SPLIT_PRIORITY = {"train": 0, "validate": 1, "test": 2, "human_test": 3}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path):
    with path.open() as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def _require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def _relative_source(path: str, root: Path, marker: tuple[str, ...]) -> Path:
    """Resolve unchanged historical paths after the source tree was relocated."""
    source = Path(path)
    if not source.is_absolute():
        relative = source
    else:
        try:
            relative = source.relative_to(root)
        except ValueError:
            try:
                relative = source.resolve().relative_to(root.resolve())
            except ValueError:
                parts = source.parts
                positions = [i for i in range(len(parts) - len(marker) + 1)
                             if parts[i:i + len(marker)] == marker]
                if len(positions) != 1:
                    raise ValueError("Source image/mask is outside the declared source tree") from None
                relative = Path(*parts[positions[0] + len(marker):])
    if relative == Path(".") or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe source image/mask path")
    _require_file(root / relative, "source image/mask")
    return relative


def _link(link: Path, source: Path) -> None:
    """Absolute symlink targets survive the builder's staging-directory rename."""
    if link.is_symlink() and link.resolve() == source.resolve():
        return
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"Refusing to replace an existing supervision asset: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def export_supervision(output: Path, legacy_root: Path, vqa_root: Path,
                       cxr_root: Path) -> dict[str, Any]:
    """Export portable annotation rows and return patient holdout assignments.

    ``legacy_root`` contains ``current`` and ``dense``; ``cxr_root`` is the
    MIMIC-CXR ``files`` directory linked by the caller as ``output/images``.
    Counts describe emitted rows before the training adapter's global patient
    filtering. Historical segmentation IDs, splits, ROI boxes, annotation indices,
    and human-mask channel ordering are preserved exactly.
    """
    output, legacy_root, vqa_root, cxr_root = map(
        lambda path: Path(path).resolve(), (output, legacy_root, vqa_root, cxr_root))
    old_path = legacy_root / "current/observations.jsonl"
    dense_path = legacy_root / "dense/observations.jsonl"
    metadata_paths = {
        "current/observations.jsonl": old_path,
        "current/manifest.json": legacy_root / "current/manifest.json",
        "current/segmentation.json": legacy_root / "current/segmentation.json",
        "dense/observations.jsonl": dense_path,
        "dense/manifest.json": legacy_root / "dense/manifest.json",
    }
    for path in metadata_paths.values():
        _require_file(path, "segmentation provenance")
    hashes = {name: _sha256(path) for name, path in metadata_paths.items()}
    old_manifest = json.loads(metadata_paths["current/manifest.json"].read_text())
    dense_manifest = json.loads(metadata_paths["dense/manifest.json"].read_text())
    annotation = json.loads(metadata_paths["current/segmentation.json"].read_text())
    for metadata, key, source in (
        (old_manifest, "observations_sha256", "current/observations.jsonl"),
        (annotation, "observations_sha256", "current/observations.jsonl"),
        (dense_manifest, "cohort_sha256", "dense/observations.jsonl"),
    ):
        if metadata.get(key) != hashes[source]:
            raise ValueError(f"Segmentation source fingerprint mismatch: {source}")

    old_rows = list(_rows(old_path))
    if len(old_rows) != old_manifest["valid_count"]:
        raise ValueError("Original segmentation observation count mismatch")
    if any(row["index"] != index for index, row in enumerate(old_rows)):
        raise ValueError("Original segmentation observation indices are not contiguous")
    if len({row["id"] for row in old_rows}) != len(old_rows):
        raise ValueError("Duplicate original segmentation image identity")
    targets_path = _require_file(legacy_root / "current/seg_probs.npy", "CXAS annotations")
    targets = np.load(targets_path, mmap_mode="r", allow_pickle=False)
    if targets.shape != (len(old_rows), 3, 256, 256) or targets.dtype != np.float16:
        raise ValueError("CXAS annotation array shape/dtype mismatch")
    del targets
    target_stat = targets_path.stat()
    montgomery_root = legacy_root / "dense/montgomery"
    if not montgomery_root.is_dir():
        raise FileNotFoundError(f"Missing original Montgomery images and masks: {montgomery_root}")

    holdouts: dict[str, str] = {}

    def holdout(patient: Any, split: str) -> None:
        patient = str(patient)
        if not patient or split not in SPLIT_PRIORITY:
            raise ValueError("Invalid patient identity or supervision split")
        if patient not in holdouts or SPLIT_PRIORITY[split] > SPLIT_PRIORITY[holdouts[patient]]:
            holdouts[patient] = split

    records, seen = [], set()
    counts = {split: 0 for split in SPLIT_PRIORITY}
    kinds: Counter[str] = Counter()
    dense_count = 0
    for index, row in enumerate(_rows(dense_path)):
        dense_count += 1
        if row["index"] != index:
            raise ValueError("Dense observation indices are not contiguous")
        if "segmentation" not in row["tasks"]:
            continue
        if row["id"] in seen:
            raise ValueError("Duplicate dense segmentation image identity")
        seen.add(row["id"])
        split = row["split"]
        if split not in SPLIT_PRIORITY:
            raise ValueError("Unknown segmentation split")
        box = row["box"]
        if (len(box) != 4 or any(type(value) is not int for value in box)
                or min(box) < 0 or min(box[2:]) < 2
                or box[0] + box[2] > 512 or box[1] + box[3] > 512):
            raise ValueError("Invalid segmentation ROI")
        record = {key: row[key] for key in ("id", "subject_id", "split", "box")}
        if row["kind"] == "montgomery":
            if split != "human_test" or len(row["masks"]) != 2:
                raise ValueError("Montgomery requires two lung masks in human_test")
            relative = _relative_source(row["image"], montgomery_root, ("montgomery",))
            record.update(kind="montgomery", image=str(Path("segmentation/montgomery") / relative),
                          masks=[str(Path("segmentation/montgomery") /
                                     _relative_source(mask, montgomery_root, ("montgomery",)))
                                 for mask in row["masks"]])
        elif row["kind"] in ("mimic", "cxas"):
            old_index = row["old_index"]
            if type(old_index) is not int or not 0 <= old_index < len(old_rows):
                raise ValueError("CXAS old_index is outside the original annotation array")
            original = old_rows[old_index]
            if (original["id"] != row["id"]
                    or str(original["subject_id"]) != str(row["subject_id"])):
                raise ValueError("CXAS old_index identity mismatch")
            relative = _relative_source(row["image"], cxr_root, ("MIMIC_CXR", "files"))
            if relative != _relative_source(original["image"], cxr_root, ("MIMIC_CXR", "files")):
                raise ValueError("CXAS old_index source image mismatch")
            record.update(kind="cxas", image=str(Path("images") / relative),
                          target_file="segmentation/cxas_probs.npy", target_index=old_index)
        else:
            raise ValueError("Unknown segmentation annotation kind")
        records.append(record)
        counts[split] += 1
        kinds[record["kind"]] += 1
        holdout(row["subject_id"], split)
    if dense_count != dense_manifest["images"]:
        raise ValueError("Dense segmentation observation count mismatch")
    if not kinds["cxas"] or not kinds["montgomery"]:
        raise ValueError("Both CXAS and human lung-mask supervision are required")

    vqa_counts, vqa_hashes = {}, {}
    checked_images = set()
    for original, split in (("train", "train"), ("valid", "validate"), ("test", "test")):
        path = _require_file(vqa_root / f"{original}.json", "official CXR-VQA split")
        count, identities = 0, set()
        with path.open("rb") as source:
            for row in ijson.items(source, "item"):
                if row["idx"] in identities:
                    raise ValueError(f"Duplicate official CXR-VQA identity in {original}")
                identities.add(row["idx"])
                image = row["image_path"]
                if image not in checked_images:
                    _relative_source(image, cxr_root, ("MIMIC_CXR", "files"))
                    checked_images.add(image)
                holdout(row["subject_id"], split)
                count += 1
        vqa_counts[split] = count
        vqa_hashes[path.name] = _sha256(path)

    output.mkdir(parents=True, exist_ok=True)
    _link(output / "segmentation/cxas_probs.npy", targets_path)
    _link(output / "segmentation/montgomery", montgomery_root)
    _link(output / "vqa", vqa_root)
    with (output / "segmentation.jsonl").open("x") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "counts": {"segmentation": counts, "vqa": vqa_counts},
        "segmentation": {
            "records": "segmentation.jsonl", "counts": counts, "kinds": dict(kinds),
            "count_scope": "exported_before_global_patient_holdout_filtering",
            "provenance": {
                "legacy_root": str(legacy_root), "metadata_sha256": hashes,
                "cxas_annotations": annotation,
                "cxas_array": {"source": str(targets_path), "bytes": target_stat.st_size,
                               "mtime_ns": target_stat.st_mtime_ns,
                               "shape": [len(old_rows), 3, 256, 256], "dtype": "float16",
                               "validation": "header_only; immutable annotation array linked without copying"},
                "human_masks": {"source": str(montgomery_root),
                                "channel_order": ["leftMask (right lung)", "rightMask (left lung)"]},
                "preservation": "Original dense IDs, splits, boxes, CXAS indices and human mask ordering",
                "input_pixels": "Original image files; no legacy image arrays or feature caches",
            },
        },
        "vqa": {"root": "vqa", "counts": vqa_counts, "source": str(vqa_root),
                "source_sha256": vqa_hashes,
                "count_scope": "official_rows_before_global_patient_holdout_filtering"},
        "holdouts": {"patients": len(holdouts), "counts": dict(Counter(holdouts.values())),
                     "priority": list(SPLIT_PRIORITY),
                     "policy": "Keep the highest existing holdout; drop conflicting rows, never move rows"},
    }
    return {"summary": summary, "holdouts": holdouts}
