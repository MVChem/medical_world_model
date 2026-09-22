"""Human-reviewed CXR/MRI supervision, with no teacher-generated mask inputs."""
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path

import ijson
from PIL import Image

from .human_cxr import ensure_sources, export_human_cxr
from .mri import export_mri

LOG = logging.getLogger(__name__)
PRIORITY = {"train": 0, "validate": 1, "test": 2, "human_test": 3}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _box(width, height):
    scale = 512 / max(height, width)
    h, w = [max(2, min(512, int(round(n * scale / 2)) * 2)) for n in (height, width)]
    return [((512 - h) // 2) // 2 * 2, ((512 - w) // 2) // 2 * 2, h, w]


def export_reviewed_supervision(output, args):
    """Return source counts and global holdouts, writing one final mask manifest."""
    holdouts, vqa_counts, vqa_hashes = {}, {}, {}
    checked = set()
    for original, split in (("train", "train"), ("valid", "validate"), ("test", "test")):
        path = args.vqa_root / f"{original}.json"
        ids, count = set(), 0
        with path.open("rb") as stream:
            for row in ijson.items(stream, "item"):
                if row["idx"] in ids:
                    raise ValueError("Duplicate VQA identity")
                ids.add(row["idx"])
                image_path = Path(row["image_path"])
                if image_path.is_absolute() or ".." in image_path.parts:
                    raise ValueError("VQA source image path must be relative")
                if row["image_path"] not in checked:
                    if not (args.cxr_root / "files" / image_path).is_file():
                        raise FileNotFoundError("Missing official VQA source image")
                    checked.add(row["image_path"])
                pid = str(row["subject_id"])
                if pid not in holdouts or PRIORITY[split] > PRIORITY[holdouts[pid]]:
                    holdouts[pid] = split
                count += 1
        vqa_counts[split], vqa_hashes[path.name] = count, _sha256(path)
    (output / "vqa").symlink_to(args.vqa_root.resolve(), target_is_directory=True)
    LOG.info("Exporting 200 human-reviewed MIMIC heart/lung masks")
    ensure_sources(args.human_cxr_root)
    cxr = export_human_cxr(output, args.human_cxr_root, args.cxr_root, holdouts, seed=args.seed)
    holdouts.update(cxr["holdouts"])
    records = list(cxr["records"])
    LOG.info("Exporting reviewed MRI volumes and axial slice references")
    mri = export_mri(output, args.ucsf_root, args.mu_root, seed=args.seed, axial_stride=args.mri_axial_stride)
    counts = Counter({s: 0 for s in PRIORITY})
    by_dataset = {}
    def observe(row):
        split, pid = row["split"], str(row["subject_id"])
        row.setdefault("annotation_source", "human_reviewed")
        row.setdefault("annotation", "human_reviewed")
        if row["kind"] not in ("human_cxr", "mri", "montgomery"):
            raise ValueError("Only reviewed segmentation sources are allowed")
        if pid in holdouts and PRIORITY[holdouts[pid]] > PRIORITY[split]:
            raise ValueError("Supervision export retained a lower-priority patient split")
        holdouts[pid] = split
        counts[split] += 1
        dataset = row["dataset"]
        by_dataset.setdefault(dataset, Counter({s: 0 for s in PRIORITY}))[split] += 1
        return row
    if not args.montgomery_root.is_dir():
        raise FileNotFoundError(args.montgomery_root)
    link = output / "segmentation/montgomery"
    link.parent.mkdir(exist_ok=True)
    link.symlink_to(args.montgomery_root.resolve(), target_is_directory=True)
    montgomery_files = sorted((args.montgomery_root / "CXR_png").glob("*.png"))
    if not montgomery_files:
        raise ValueError("No Montgomery human test images")
    for image_path in montgomery_files:
        masks = [args.montgomery_root / "ManualMask" / side / image_path.name for side in ("leftMask", "rightMask")]
        with Image.open(image_path) as image:
            size = image.size
        for mask in masks:
            with Image.open(mask) as image:
                if image.size != size:
                    raise ValueError("Montgomery mask/image geometry differs")
        identity = "montgomery:" + image_path.stem
        records.append({"id": identity, "subject_id": identity, "volume_id": identity,
            "split": "human_test", "kind": "montgomery", "dataset": "montgomery",
            "image": str(Path("segmentation/montgomery/CXR_png") / image_path.name),
            "masks": [str(Path("segmentation/montgomery/ManualMask") / side / image_path.name)
                      for side in ("leftMask", "rightMask")],
            "box": _box(*size), "channels": [0], "target_names": ["lungs"],
            "annotation_source": "human_reviewed"})
    with (output / "segmentation.jsonl").open("w") as target:
        for row in records:
            target.write(json.dumps(observe(row), separators=(",", ":")) + "\n")
        with (output / "mri_segmentation.jsonl").open() as source:
            for line in source:
                row = observe(json.loads(line))
                target.write(json.dumps(row, separators=(",", ":")) + "\n")
    return {"holdouts": holdouts, "summary": {
        "annotation_policy": "human_reviewed_only; no CXAS or other unreviewed pseudo-mask targets",
        "counts": {"segmentation": dict(counts), "vqa": vqa_counts},
        "segmentation": {"by_dataset": {k: dict(v) for k, v in by_dataset.items()},
                         "cxr": cxr["summary"], "mri": mri,
                         "montgomery": {"images": len(montgomery_files), "source": str(args.montgomery_root),
                                        "target": "union of two original human lung masks"}},
        "vqa": {"counts": vqa_counts, "source_sha256": vqa_hashes},
    }}
