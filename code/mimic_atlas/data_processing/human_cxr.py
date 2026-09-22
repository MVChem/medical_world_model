"""Verified public, clinician-reviewed MIMIC heart/lung mask acquisition/export.

Original masks stay byte-for-byte unchanged and are linked into the prepared
dataset. No pseudo labels, pixel arrays, or derived masks are produced.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import gzip
import hashlib
import io
import json
import logging
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from urllib.request import urlopen
import zipfile

from PIL import Image


SOURCE_URL = "https://physionet.org/content/heart-lung-segmentations-data/1.0.0/"
FILES_URL = "https://physionet.org/files/heart-lung-segmentations-data/1.0.0/"
ZIP_URL = "https://physionet.org/content/heart-lung-segmentations-data/get-zip/1.0.0/"
SOURCE_DOI = "https://doi.org/10.13026/0k35-mb65"
DEFAULT_ANNOTATION_ROOT = Path.home() / "data/heart-lung-segmentations-data/1.0.0"
PRIORITY = {"train": 0, "validate": 1, "test": 2, "human_test": 3}
LOG = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(value: str) -> Path:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("Unsafe public annotation path")
    return Path(*path.parts)


def verify_sources(annotation_root: Path) -> dict:
    """Validate every release file against the publisher's SHA256SUMS.txt."""
    root = Path(annotation_root)
    checksum_file = root / "SHA256SUMS.txt"
    expected, total_bytes = {}, 0
    for line in checksum_file.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        relative = _safe_relative(name.lstrip("*"))
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid public annotation SHA256SUMS entry")
        if relative in expected:
            raise ValueError("Duplicate public annotation SHA256SUMS entry")
        if relative.suffix.lower() == ".pdf":
            raise ValueError("Unexpected PDF in the image annotation release")
        expected[relative] = digest
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing public human annotation: {relative}")
        if _sha256(path) != digest:
            raise ValueError(f"Public human annotation SHA256 mismatch: {relative}")
        total_bytes += path.stat().st_size
    if not expected:
        raise ValueError("Empty public annotation SHA256SUMS manifest")
    return {"files_verified": len(expected), "bytes": total_bytes,
            "sha256sums_sha256": _sha256(checksum_file), "verification": "all published SHA256 sums matched"}


def ensure_sources(annotation_root: Path) -> dict:
    """Download this open-access release if missing; never replace corrupt data.

    A temporary staging directory is removed after verification. Repeated calls
    verify and reuse the immutable local files without downloading them again.
    """
    root = Path(annotation_root).resolve()
    try:
        return verify_sources(root)
    except FileNotFoundError:
        pass
    root.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("Downloading public clinician-reviewed heart/lung annotations")
    with urlopen(ZIP_URL, timeout=60) as response:
        payload = response.read(32 * 1024 * 1024 + 1)
    if len(payload) > 32 * 1024 * 1024:
        raise ValueError("Unexpectedly large public annotation archive")
    with tempfile.TemporaryDirectory(prefix=".human_cxr_download_", dir=root.parent) as temporary:
        stage = Path(temporary)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = [entry for entry in archive.infolist() if not entry.is_dir()]
            checksum_entries = [entry for entry in entries if Path(entry.filename).name == "SHA256SUMS.txt"]
            if len(checksum_entries) != 1:
                raise ValueError("Annotation archive has no unique checksum manifest")
            prefix = _safe_relative(checksum_entries[0].filename).parent
            seen = set()
            for entry in entries:
                path = _safe_relative(entry.filename)
                try:
                    relative = path.relative_to(prefix)
                except ValueError:
                    raise ValueError("Annotation archive contains files outside its release tree") from None
                if relative in seen or relative.suffix.lower() == ".pdf":
                    raise ValueError("Duplicate or unexpected annotation archive member")
                if entry.file_size > 16 * 1024 * 1024:
                    raise ValueError("Unexpectedly large annotation archive member")
                seen.add(relative)
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(entry))
        verified = verify_sources(stage)
        root.mkdir(parents=True, exist_ok=True)
        for source in stage.rglob("*"):
            if not source.is_file():
                continue
            target = root / source.relative_to(stage)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if _sha256(target) != _sha256(source):
                    raise ValueError("Existing human annotation differs from the verified release")
            else:
                shutil.move(str(source), str(target))
    LOG.info("Verified %d human annotation release files", verified["files_verified"])
    return verified


def _table(root: Path, name: str) -> Path:
    for extension in (".csv.gz", ".csv"):
        path = root / (name + extension)
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing official MIMIC-CXR table: {name}")


def _rows(path: Path):
    opener = gzip.open(path, "rt", newline="") if path.suffix == ".gz" else path.open(newline="")
    with opener as source:
        yield from csv.DictReader(source)


def _canvas_box(height: int, width: int) -> list[int]:
    scale = 512 / max(height, width)
    h, w = (max(2, min(512, round(size * scale / 2) * 2)) for size in (height, width))
    return [((512 - h) // 2) // 2 * 2, ((512 - w) // 2) // 2 * 2, h, w]


def assign_patient_splits(patients, minimum_splits: dict[str, str], seed: int = 42) -> dict[str, str]:
    """Fill 70/15/15 patient targets while preserving existing nontrain holdouts.

    If existing holdouts exceed a target, they stay intact and the realized
    proportions differ. Only previously train patients receive new holdouts.
    """
    patients = sorted(set(map(str, patients)))
    if any(minimum_splits.get(patient, "train") not in PRIORITY for patient in patients):
        raise ValueError("Unknown existing human CXR patient split")
    result = {patient: minimum_splits.get(patient, "train") for patient in patients}
    free = [patient for patient in patients if result[patient] == "train"]
    free.sort(key=lambda patient: hashlib.sha256(f"{seed}:human_cxr:{patient}".encode()).digest())
    target = round(len(patients) * .15)
    offset = 0
    for split in ("test", "validate"):
        needed = max(0, target - sum(value == split for value in result.values()))
        for patient in free[offset:offset + needed]:
            result[patient] = split
        offset += needed
    return result


def export_human_cxr(output: Path, annotation_root: Path, cxr_root: Path,
                     holdouts: dict[str, str] | None = None, *, seed: int = 42) -> dict:
    """Return self-contained rows, provenance, and updated patient holdouts.

    ``cxr_root`` contains ``files`` and the official split CSV. The caller creates
    ``output/images`` and writes returned ``records`` into segmentation.jsonl.
    All 200 images are retained; only newly assigned patient holdouts change the
    source training designation, with that change explicitly recorded per row.
    """
    output, annotation_root, cxr_root = map(lambda p: Path(p).resolve(), (output, annotation_root, cxr_root))
    verified = verify_sources(annotation_root)
    link_path = annotation_root / "mimic_masks/MIMIC_links.csv"
    links = list(_rows(link_path))
    if len(links) != 200:
        raise ValueError("The public human MIMIC-CXR release must contain exactly 200 image mappings")
    source_rows = {}
    patients = set()
    for link in links:
        relative = _safe_relative(link["MIMIC-CXR_path"])
        subject, study = str(link["subject_id"]), str(link["study_id"])
        if (len(relative.parts) != 5 or relative.parts[0] != "files"
                or relative.parts[1] != "p" + subject[:2]
                or relative.parts[2] != "p" + subject or relative.parts[3] != "s" + study
                or relative.suffix != ".dcm"):
            raise ValueError("Public MIMIC mask mapping does not match patient/study identity")
        identity = relative.stem
        if identity in source_rows:
            raise ValueError("Duplicate public human MIMIC image mapping")
        source_rows[identity] = (link, relative.with_suffix(".jpg"))
        patients.add(subject)
    split_path = _table(cxr_root, "mimic-cxr-2.0.0-split")
    minimum = {str(patient): split for patient, split in (holdouts or {}).items()}
    if any(split not in PRIORITY for split in minimum.values()):
        raise ValueError("Unknown supplied patient holdout split")
    official = {}
    for row in _rows(split_path):
        subject, split = row["subject_id"], row["split"]
        if subject in patients:
            if split not in ("train", "validate", "test"):
                raise ValueError("Unknown official CXR patient split")
            if subject not in minimum or PRIORITY[split] > PRIORITY[minimum[subject]]:
                minimum[subject] = split
        if row["dicom_id"] in source_rows:
            original = source_rows[row["dicom_id"]][0]
            if (row["subject_id"] != original["subject_id"]
                    or row["study_id"] != original["study_id"]):
                raise ValueError("Public annotation identity differs from official CXR split metadata")
            if row["dicom_id"] in official:
                raise ValueError("Duplicate human image in official CXR split table")
            official[row["dicom_id"]] = split
    if set(official) != set(source_rows):
        raise ValueError("Official CXR split table does not cover every human annotation")
    assigned = assign_patient_splits(patients, minimum, seed)
    updated_holdouts = dict(minimum)
    updated_holdouts.update(assigned)
    records, dimensions = [], Counter()
    for identity, (link, relative) in source_rows.items():
        with Image.open(cxr_root / relative) as image:
            width, height = image.size
        masks = []
        for name, field in (("lungs", "lungs_mask_path"), ("heart", "heart_mask_path")):
            mask_relative = _safe_relative(link[field])
            if len(mask_relative.parts) != 2 or mask_relative.parts[0] != name:
                raise ValueError("Public heart/lung mask mapping uses an unexpected channel path")
            # Publisher's CSV retains old .jpg names; release masks are .png.
            mask_relative = Path("mimic_masks") / mask_relative.with_suffix(".png")
            with Image.open(annotation_root / mask_relative) as mask:
                if mask.size != (width, height) or mask.mode != "L":
                    raise ValueError("Human CXR mask geometry differs from its original image")
                colors = mask.getcolors(maxcolors=3)
                if colors is None or {value for _, value in colors} != {0, 255}:
                    raise ValueError("Human CXR mask must contain nonempty binary foreground/background")
            masks.append(str(Path("segmentation/heart_lung_human") / mask_relative))
        subject = link["subject_id"]
        records.append({
            "id": "human_cxr:" + identity, "subject_id": subject, "study_id": link["study_id"],
            "split": assigned[subject], "official_split": official[identity],
            "split_assignment": ("preserved_existing_holdout" if minimum[subject] != "train"
                                 else "deterministic_patient_70_15_15"),
            "image": str(Path("images") / relative.relative_to("files")),
            "box": _canvas_box(height, width), "kind": "human_cxr", "dataset": "mimic_cxr_human",
            "masks": masks, "channels": [0, 1], "target_names": ["lungs", "heart"],
            "annotation": "expert_reviewed", "annotation_source": "human_reviewed",
            "modality": "CXR", "mask_geometry": {"native_size": [height, width], "transform": "identity"},
        })
        dimensions[(width, height)] += 1
    destination = output / "segmentation/heart_lung_human"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() and destination.resolve() == annotation_root:
        pass
    elif destination.exists() or destination.is_symlink():
        raise FileExistsError("Refusing to replace an existing human annotation link")
    else:
        destination.symlink_to(annotation_root, target_is_directory=True)
    summary = {
        "dataset": "mimic_cxr_human", "images": len(records), "patients": len(patients),
        "counts": {split: sum(row["split"] == split for row in records) for split in PRIORITY},
        "patient_counts": dict(Counter(assigned.values())), "channels": [0, 1],
        "target_names": ["lungs", "heart"], "annotation": "expert_reviewed",
        "source_url": SOURCE_URL, "source_doi": SOURCE_DOI, "source_root": str(annotation_root),
        "license": "Open Data Commons Attribution License v1.0", "verification": verified,
        "source_sha256": {"MIMIC_links.csv": _sha256(link_path), "official_split": _sha256(split_path)},
        "split_policy": {"seed": seed, "desired_patient_fractions": {"train": .70, "validate": .15, "test": .15},
                         "official_image_counts": dict(Counter(official.values())),
                         "rule": "Preserve official and existing VQA holdouts; fill missing human test/validate quotas from train patients by seeded SHA256 ordering",
                         "global_use": "All tasks must drop conflicting training rows for newly held-out patients"},
        "mask_geometry": {"exact_image_size_matches": len(records) * 2,
                          "native_sizes": len(dimensions), "binary_values": [0, 255],
                          "orientation": "original raster orientation; no flip or transpose",
                          "resampling": "nearest-neighbor mask resize into the same ROI as its original image"},
    }
    return {"records": records, "summary": summary, "holdouts": updated_holdouts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ANNOTATION_ROOT,
                        help="Central raw-data directory; project data paths should link here.")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    run = Path(__file__).resolve().parents[1] / "runs" / ("human_cxr_" + datetime.now().strftime("%Y%m%d"))
    run.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler(run / "download.log")],
                        format="%(asctime)s %(levelname)s %(message)s")
    result = verify_sources(args.output) if args.verify_only else ensure_sources(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
