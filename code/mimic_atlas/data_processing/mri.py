"""Export expert-reviewed glioma masks as source-linked axial segmentation rows.

Slice selection and intensity windows use the source MRI alone. All visits of a
patient receive the same deterministic split; MRI files remain in their original
directories, and no processed image or mask arrays are written.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
from pathlib import Path
import re

import nibabel as nib
import numpy as np

LOG = logging.getLogger(__name__)
SPLITS = ("train", "validate", "test", "human_test")
CHANNELS = [2, 3, 4, 5]
TARGET_NAMES = ["NETC", "SNFH", "ET", "RC"]
DATASETS = {"ucsf": "ucsf_alptdg", "mu": "mu_glioma_post"}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def patient_splits(patients, dataset, seed=42):
    """Hash-rank complete patients, then allocate floor(70%)/floor(15%)/rest."""
    ordered = sorted(set(patients), key=lambda pid: (
        hashlib.sha256(f"{seed}:mri:{dataset}:{pid}".encode()).digest(), pid))
    train_end = int(len(ordered) * .70)
    validate_end = train_end + int(len(ordered) * .15)
    return {pid: "train" if i < train_end else "validate" if i < validate_end else "test"
            for i, pid in enumerate(ordered)}


def canvas_box(shape, zooms):
    """Physical in-plane aspect ratio, on the even 512-square canvas grid."""
    height, width = float(shape[1] * zooms[1]), float(shape[0] * zooms[0])
    if not np.isfinite([height, width]).all() or min(height, width) <= 0:
        raise ValueError("MRI has invalid in-plane geometry")
    scale = 512 / max(height, width)
    h, w = (max(2, min(512, int(round(value * scale / 2)) * 2)) for value in (height, width))
    return [((512 - h) // 2) // 2 * 2, ((512 - w) // 2) // 2 * 2, h, w]


def _validate_header(image):
    if len(image.shape) != 3 or min(image.shape) <= 0:
        raise ValueError("MRI source must be a nonempty 3D NIfTI")
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-12:
        raise ValueError("MRI source has an invalid affine")


def _canonical(path):
    original = nib.load(str(path))
    _validate_header(original)
    image = nib.as_closest_canonical(original)
    return image


def _canonical_header(path):
    """Canonical shape/affine without decompressing unconsumed MRI sequences."""
    original = nib.load(str(path))
    _validate_header(original)
    orientation = nib.orientations.io_orientation(original.affine)
    shape = tuple(original.shape[int(axis)] for axis in np.argsort(orientation[:, 0]))
    affine = original.affine @ nib.orientations.inv_ornt_aff(orientation, original.shape)
    return shape, affine


def _visits(root, dataset):
    """Only standard visit masks qualify; UCSF subtraction labels are excluded."""
    if dataset == "ucsf":
        for folder in sorted(root.iterdir()):
            if folder.is_dir() and re.fullmatch(r"\d+", folder.name):
                for time in (1, 2):
                    stem = folder / f"{folder.name}_time{time}_"
                    yield folder.name, time, {
                        name: Path(str(stem) + suffix + ".nii.gz")
                        for name, suffix in (("t1", "t1"), ("t1ce", "t1ce"), ("t2", "t2"), ("flair", "flair"))
                    }, Path(str(stem) + "seg.nii.gz")
    else:
        for folder in sorted(root.glob("PatientID_*/Timepoint_*")):
            if not folder.is_dir() or not re.fullmatch(r"Timepoint_\d+", folder.name):
                continue
            pid, time = folder.parent.name, int(folder.name.split("_")[-1])
            stem = folder / f"{pid}_{folder.name}_"
            yield pid, time, {
                name: Path(str(stem) + suffix + ".nii.gz")
                for name, suffix in (("t1", "brain_t1n"), ("t1ce", "brain_t1c"),
                                     ("t2", "brain_t2w"), ("flair", "brain_t2f"))
            }, Path(str(stem) + "tumorMask.nii.gz")


def _inspect(sequences, mask_path, axial_stride):
    """Validate all four sequence headers and the complete discrete target."""
    source = _canonical(sequences["t1ce"])
    mask = _canonical(mask_path) if mask_path.is_file() else None
    headers = [_canonical_header(path) for name, path in sequences.items() if name != "t1ce"]
    if mask is not None:
        headers.append((mask.shape, mask.affine))
    for shape, affine in headers:
        if shape != source.shape or not np.allclose(affine, source.affine, atol=1e-4, rtol=1e-5):
            raise ValueError("MRI modalities/mask must share the canonical source grid")
    details = {"canonical_shape": list(source.shape),
               "canonical_affine": source.affine.tolist(),
               "voxel_spacing": list(map(float, source.header.get_zooms()[:3])),
               "box": canvas_box(source.shape, source.header.get_zooms()),
               "t1ce_sha256": _sha256(sequences["t1ce"]),
               "mask_sha256": _sha256(mask_path) if mask is not None else None}
    if mask is None:
        return details
    values = np.asanyarray(mask.dataobj)
    labels = np.unique(values)
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1, 2, 3, 4]).all():
        raise ValueError("Reviewed MRI masks require discrete integer labels 0..4")
    details["label_values"] = [int(value) for value in labels]
    foreground_slices = np.flatnonzero((values != 0).any(axis=(0, 1)))
    del values
    pixels = source.get_fdata(dtype=np.float32)
    if not np.isfinite(pixels).all():
        raise ValueError("MRI source has nonfinite image values")
    positive = pixels > 0
    occupied = np.flatnonzero(positive.any(axis=(0, 1)))
    if not len(occupied):
        raise ValueError("MRI source has no positive brain foreground")
    if (len(foreground_slices)
            and (foreground_slices[0] < occupied[0] or foreground_slices[-1] > occupied[-1])):
        raise ValueError(f"MRI target foreground lies outside image-defined axial extent: {mask_path.name}")
    details["foreground_outside_image_extent"] = False
    lo, hi = np.percentile(pixels[positive], [1, 99.5])
    if hi <= lo:
        hi = lo + 1
    # Include every axial plane in the image-defined extent, including planes
    # with no tumor; target masks never determine eligibility or the window.
    details.update(normalization=[float(lo), float(hi)],
                   slice_indices=list(range(int(occupied[0]), int(occupied[-1]) + 1, axial_stride)))
    return details


def export_mri(output, ucsf_root, mu_root, *, seed=42, axial_stride=1, workers=4):
    """Write MRI-only rows/inventory and return counts for the combined builder.

    ``mri_segmentation.jsonl`` is ready to merge into ``segmentation.jsonl``.
    ``mri_volumes.jsonl`` preserves all four modality paths and the two MU visits
    without masks. Counts for segmentation refer to 2D slices; volume counts are
    reported separately so one 3D annotation is never claimed as many annotations.
    """
    if type(axial_stride) is not int or axial_stride < 1:
        raise ValueError("MRI axial_stride must be a positive integer")
    if type(workers) is not int or workers < 1:
        raise ValueError("MRI workers must be a positive integer")
    output = Path(output).absolute()
    roots = {"ucsf": Path(ucsf_root).resolve(strict=True), "mu": Path(mu_root).resolve(strict=True)}
    destinations = [output / "mri_segmentation.jsonl", output / "mri_volumes.jsonl", output / "mri"]
    if any(path.exists() or path.is_symlink() for path in destinations):
        raise FileExistsError("Refusing to replace MRI manifests or source links")
    visits = {dataset: list(_visits(root, dataset)) for dataset, root in roots.items()}
    if any(not rows for rows in visits.values()):
        raise ValueError("Both MRI datasets must contain visits")
    output.mkdir(parents=True, exist_ok=True)
    (output / "mri").mkdir()
    for dataset, root in roots.items():
        (output / "mri" / dataset).symlink_to(root, target_is_directory=True)
    counts = Counter({split: 0 for split in SPLITS})
    volume_counts = Counter({split: 0 for split in SPLITS})
    dataset_summary = {}
    holdouts = {}
    with (output / "mri_segmentation.jsonl").open("x") as rows_file, (output / "mri_volumes.jsonl").open("x") as volumes_file, ThreadPoolExecutor(max_workers=workers) as pool:
        for dataset, rows in visits.items():
            root = roots[dataset]
            splits = patient_splits([row[0] for row in rows], dataset, seed)
            stats = {"patients": len(splits), "volumes": len(rows), "annotated_volumes": 0,
                     "missing_masks": 0, "slice_counts": {split: 0 for split in SPLITS},
                     "patient_counts": dict(Counter(splits.values())),
                     "annotated_volume_counts": {split: 0 for split in SPLITS}}
            inspections = pool.map(lambda visit: _inspect(visit[2], visit[3], axial_stride), rows)
            for index, ((pid, time, sequences, mask_path), geometry) in enumerate(zip(rows, inspections)):
                if index % 50 == 0:
                    LOG.info("MRI %s: checking volume %d/%d", dataset, index, len(rows))
                for path in sequences.values():
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing original MRI sequence: {path}")
                subject, split = f"{dataset}:{pid}", splits[pid]
                holdouts[subject] = split
                base = {"volume_id": f"mri:{dataset}:{pid}:t{time}", "subject_id": subject,
                        "split": split, "dataset": DATASETS[dataset], "timepoint": time,
                        "kind": "mri", "channels": CHANNELS, "target_names": TARGET_NAMES,
                        "orientation": "RAS", "modality": "t1ce",
                        "image": str(Path("mri") / dataset / sequences["t1ce"].relative_to(root))}
                links = {name: str(Path("mri") / dataset / path.relative_to(root)) for name, path in sequences.items()}
                mask_link = str(Path("mri") / dataset / mask_path.relative_to(root)) if mask_path.is_file() else None
                inventory = {**base, **geometry, "sequences": links, "mask_file": mask_link,
                             "annotation": "expert_reviewed" if mask_link else "missing",
                             "annotation_source": "human_reviewed" if mask_link else "missing"}
                volumes_file.write(json.dumps(inventory, separators=(",", ":")) + "\n")
                if mask_link is None:
                    stats["missing_masks"] += 1
                    continue
                stats["annotated_volumes"] += 1
                stats["annotated_volume_counts"][split] += 1
                volume_counts[split] += 1
                for z in geometry["slice_indices"]:
                    row = {**base, "id": f"{base['volume_id']}:z{z}", "mask_file": mask_link,
                           "annotation": "expert_reviewed", "annotation_source": "human_reviewed", "slice_index": z,
                           **{key: geometry[key] for key in ("box", "canonical_shape", "normalization", "t1ce_sha256", "mask_sha256")}}
                    rows_file.write(json.dumps(row, separators=(",", ":")) + "\n")
                    counts[split] += 1
                    stats["slice_counts"][split] += 1
            dataset_summary[DATASETS[dataset]] = stats
    return {"holdouts": holdouts, "counts": dict(counts), "volumes": dict(volume_counts), "summary": {
        "counts": {"segmentation": dict(counts)}, "datasets": dataset_summary,
        "channels": CHANNELS, "target_names": TARGET_NAMES, "seed": seed,
        "annotated_volume_counts": dict(volume_counts),
        "split_policy": "Per-dataset patient SHA256 rank: floor(70%) train, floor(15%) validate, remainder test",
        "slice_policy": "All canonical RAS axial planes in the positive T1ce image extent, stride=" + str(axial_stride),
        "normalization": "Source T1ce positive-voxel 1st/99.5th percentiles, independent of masks",
        "source_identity": "SHA256 of each original T1ce image and standard segmentation file; checked again on lazy decode",
        "foreground_coverage": "Every target's nonzero axial extent must be contained in the image-selected extent",
        "annotation_policy": "Dataset-provided, expert-reviewed standard visit masks; no pseudo masks or subtraction targets",
        "sources": {"ucsf_alptdg": "https://doi.org/10.1148/ryai.230182",
                    "mu_glioma_post": "https://doi.org/10.1038/s41597-025-06011-7"},
    }}
