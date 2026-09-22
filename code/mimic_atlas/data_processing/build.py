"""Prepare source-linked MedWorld manifests without pixel or feature caches."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import shutil
import tempfile

from .. import build_mimic_transitions as cxr
from .. import link_mimic_iv_context as iv
from .pairing import select_patient_pairs
from .reviewed_supervision import export_reviewed_supervision
from .human_cxr import DEFAULT_ANNOTATION_ROOT

CODE_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "medworld-prepared-v2"
SEGMENTATION_LAYOUT = ["lungs", "heart", "NETC", "SNFH", "ET", "RC"]
SPLITS = ("train", "validate", "test", "human_test")
PRIORITY = dict(zip(SPLITS, range(4)))
LOG = logging.getLogger(__name__)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_row(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def table_path(root, name):
    return cxr._find_table(Path(root), name)


def rows(path):
    with cxr._open_csv(path) as stream:
        yield from csv.DictReader(stream)


def link_source(target, link):
    target = Path(target).resolve(strict=True)
    link.symlink_to(target, target_is_directory=target.is_dir())


def canvas_box(height, width):
    """512-square even ROI, matching the 256-square segmentation grid."""
    if not height or not width or min(height, width) <= 0:
        raise ValueError("Image dimensions must be positive")
    scale = 512 / max(height, width)
    h, w = (max(2, min(512, int(round(n * scale / 2)) * 2)) for n in (height, width))
    return [((512 - h) // 2) // 2 * 2, ((512 - w) // 2) // 2 * 2, h, w]


def observation_id(image):
    return "cxr:" + image.dicom_id


def linked_image_path(image):
    return str(Path("images") / Path(image.relative_path).relative_to("files"))


def admission_spans(admissions, audit):
    output = defaultdict(list)
    for row in admissions:
        try:
            start, end = iv.admission_start(row), iv.parse_time(row.get("dischtime"))
            if end is None or start > end:
                raise ValueError("Incomplete or reversed admission")
        except (ValueError, TypeError):
            audit["invalid_admissions"] += 1
            continue
        output[row["subject_id"]].append((start, end, row))
    return output


def image_link(image, spans, stays):
    matches = [row for start, end, row in spans if start <= image.timestamp <= end]
    unique = matches[0]["hadm_id"] if len(matches) == 1 else None
    stay_ids = [row["stay_id"] for start, end, row in stays
                if unique and row["hadm_id"] == unique and start <= image.timestamp <= end]
    return {
        "dicom_id": image.dicom_id, "view": image.view,
        "acquisition_timestamp": image.timestamp.isoformat(),
        "admission_status": "unique" if unique else "ambiguous" if matches else "unmatched",
        "hadm_id": unique, "candidate_hadm_ids": sorted(r["hadm_id"] for r in matches),
        "icu_status": "unique" if len(stay_ids) == 1 else "ambiguous" if stay_ids else "unmatched",
        "stay_id": stay_ids[0] if len(stay_ids) == 1 else None,
        "candidate_stay_ids": sorted(stay_ids),
    }


def _build(args, output):
    audit = Counter()
    source_paths = {
        "cxr_metadata": table_path(args.cxr_root, "mimic-cxr-2.0.0-metadata"),
        "cxr_split": table_path(args.cxr_root, "mimic-cxr-2.0.0-split"),
        "cxr_chexpert": table_path(args.cxr_root, "mimic-cxr-2.0.0-chexpert"),
        "iv_patients": table_path(args.iv_root, "hosp/patients"),
        "iv_admissions": table_path(args.iv_root, "hosp/admissions"),
        "iv_icustays": table_path(args.iv_root, "icu/icustays"),
    }
    link_source(args.cxr_root / "files", output / "images")
    link_source(args.iv_root, output / "iv")
    LOG.info("Linking human-reviewed segmentation annotations and official VQA sources")
    supervision = export_reviewed_supervision(output, args)
    holdouts = supervision["holdouts"]
    LOG.info("Loading full CXR metadata, official splits and CheXpert labels")
    source_audit = cxr.Audit()
    studies = cxr.load_studies(args.cxr_root, "frontal", source_audit)
    cxr.attach_splits(args.cxr_root, studies, source_audit, strict=True)
    cxr.attach_labels(args.cxr_root, studies, source_audit)
    patients = defaultdict(list)
    for study in studies.values():
        patients[study.subject_id].append(study)
        if (study.subject_id not in holdouts
                or PRIORITY[study.split] > PRIORITY[holdouts[study.subject_id]]):
            holdouts[study.subject_id] = study.split
    LOG.info("Loaded %d studies / %d patients; joining MIMIC-IV", len(studies), len(patients))
    iv_patients = {row["subject_id"] for row in rows(source_paths["iv_patients"])}
    admissions = [row for row in rows(source_paths["iv_admissions"]) if row["subject_id"] in patients]
    spans = admission_spans(admissions, audit)
    stays = defaultdict(list)
    for row in rows(source_paths["iv_icustays"]):
        if row["subject_id"] not in patients:
            continue
        try:
            start, end = iv.parse_time(row["intime"]), iv.parse_time(row["outtime"])
            if start is None or end is None or start > end:
                raise ValueError("Invalid ICU interval")
        except (ValueError, TypeError):
            audit["invalid_icu_stays"] += 1
            continue
        stays[row["subject_id"]].append((start, end, row))
    patient_ids = sorted(set(patients) & iv_patients)
    audit["cxr_patients_without_iv_patient_key"] = len(set(patients) - iv_patients)
    if args.max_patients:
        patient_ids.sort(key=lambda pid: hashlib.sha256(f"{args.seed}:patient:{pid}".encode()).digest())
        patient_ids = sorted(patient_ids[:args.max_patients])
    temporal_dir = output / "temporal"
    temporal_dir.mkdir()
    counts = {task: Counter({split: 0 for split in SPLITS}) for task in ("classification", "temporal", "observations")}
    pair_kinds = Counter()
    patient_counts = {split: set() for split in SPLITS}
    with ExitStack() as stack:
        classification = stack.enter_context((output / "classification.jsonl").open("w"))
        linkage = stack.enter_context((output / "study_links.jsonl").open("w"))
        observations = stack.enter_context((temporal_dir / "observations.jsonl").open("w"))
        pair_streams = {s: stack.enter_context((temporal_dir / f"{s}.jsonl").open("w")) for s in SPLITS[:3]}
        for index, pid in enumerate(patient_ids):
            if index % 2500 == 0:
                LOG.info("Patients %d/%d; classification %d; temporal pairs %d", index, len(patient_ids),
                         sum(counts["classification"].values()), sum(counts["temporal"].values()))
            timeline = sorted(patients[pid], key=lambda s: (s.timestamp, s.study_id))
            reports, eligible = {}, set()
            for study in timeline:
                image_links = {view: image_link(im, spans[pid], stays[pid])
                               for view, im in study.images_by_view.items()}
                for linked in image_links.values():
                    audit["frontal_images_admission_" + linked["admission_status"]] += 1
                write_row(linkage, {"subject_id": pid, "study_id": study.study_id,
                    "split": study.split, "effective_holdout": holdouts[pid],
                    "study_anchor_timestamp": study.timestamp.isoformat(),
                    "latest_image_timestamp": study.latest_image_timestamp.isoformat(),
                    "images": list(image_links.values()), "role": "retrospective_audit_only"})
                if holdouts[pid] != study.split:
                    audit["studies_excluded_global_holdout"] += 1
                    continue
                if study.labels is None:
                    audit["studies_without_labels"] += 1
                    continue
                labels = [study.labels[cxr.LABEL_COLUMNS.index(finding)] for finding in cxr.PATHOLOGY_COLUMNS]
                for view, image in study.images_by_view.items():
                    if args.linkage == "same_admission" and image_links[view]["admission_status"] != "unique":
                        continue
                    if not image.image_path.is_file():
                        audit["missing_images"] += 1
                        continue
                    if not image.rows or not image.columns:
                        audit["images_without_dimensions"] += 1
                        continue
                    write_row(classification, {
                        "id": observation_id(image), "subject_id": pid, "study_id": study.study_id,
                        "split": study.split, "image": linked_image_path(image), "labels": labels,
                        "box": canvas_box(image.rows, image.columns), "view": view,
                        "hadm_id": image_links[view]["hadm_id"],
                    })
                    counts["classification"][study.split] += 1
                if not study.images_by_view or not all(im.image_path.is_file() for im in study.images_by_view.values()):
                    continue
                try:
                    report = cxr.report_for_prompt(cxr.read_report(study.report_path, args.max_report_chars))
                except OSError:
                    audit["missing_reports"] += 1
                    continue
                if not report.strip():
                    audit["empty_reports"] += 1
                    continue
                reports[study.study_id] = report
                eligible.add(study.study_id)
            pairs, pair_audit = select_patient_pairs(timeline, [r for _, _, r in spans[pid]],
                mode=args.pair_mode, seed=args.seed, random_pairs_per_patient=args.random_pairs_per_patient,
                min_gap_hours=args.min_gap_hours, max_gap_days=args.max_gap_days,
                linkage=args.linkage, eligible_study_ids=eligible)
            audit.update(pair_audit)
            written = set()
            for pair in pairs:
                source, target = pair["source_study"], pair["target_study"]
                for study, image in ((source, pair["source_image"]), (target, pair["target_image"])):
                    identity = observation_id(image)
                    if identity not in written:
                        write_row(observations, {"id": identity, "patient": pid, "subject_id": pid,
                            "study_id": study.study_id, "split": study.split,
                            "image": "../" + linked_image_path(image), "report": reports[study.study_id],
                            "labels": [study.labels[cxr.LABEL_COLUMNS.index(f)] for f in cxr.PATHOLOGY_COLUMNS],
                            "timestamp": image.timestamp.isoformat(), "view": image.view})
                        written.add(identity)
                        counts["observations"][study.split] += 1
                source_id, target_id = observation_id(pair["source_image"]), observation_id(pair["target_image"])
                identity = "pair:" + hashlib.sha256(f"{source_id}\0{target_id}".encode()).hexdigest()[:24]
                write_row(pair_streams[source.split], {
                    "id": identity, "patient": pid, "split": source.split,
                    "source": source_id, "target": target_id,
                    "realized_gap_hours": pair["realized_gap_hours"],
                    "matched_view": pair["matched_view"], "hadm_id": pair["hadm_id"],
                    "pairing": pair["kind"], "source_order": pair["source_order"],
                    "target_order": pair["target_order"],
                })
                counts["temporal"][source.split] += 1
                pair_kinds[pair["kind"]] += 1
                patient_counts[source.split].add(pid)
    LOG.info("Hashing source tables and validating manifest counts")
    artifact_names = ["classification.jsonl", "segmentation.jsonl", "study_links.jsonl",
                      "temporal/observations.jsonl", *(f"temporal/{s}.jsonl" for s in SPLITS[:3])]
    artifact_names += [name for name in ("mri_segmentation.jsonl", "mri_volumes.jsonl") if (output / name).exists()]
    manifest = {
        "schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
        "findings": list(cxr.PATHOLOGY_COLUMNS),
        "segmentation_layout": SEGMENTATION_LAYOUT,
        "annotation_policy": "human_reviewed_only",
        "counts": {task: dict(value) for task, value in counts.items()},
        "supervision": supervision["summary"],
        "temporal_patient_counts": {s: len(v) for s, v in patient_counts.items()},
        "pair_kinds": dict(pair_kinds), "audit": dict(audit), "source_audit": source_audit.as_dict(),
        "source_paths": {k: str(v.resolve()) for k, v in source_paths.items()},
        "source_sha256": {k: sha256(v) for k, v in source_paths.items()},
        "file_sha256": {name: sha256(output / name) for name in artifact_names},
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "patient_policy": "Official CXR splits; higher-priority VQA/segmentation holdouts remove conflicting rows, never relocate rows.",
        "selection": "Full shared IV patient cohort; one representative image per AP/PA view per study; no label-value filters.",
        "linkage": "Exact subject_id; selected image acquisition within [min(edregtime,admittime),dischtime]; ambiguous matches retained only in audit.",
        "ehr": "Link identifiers and raw IV source symlink are retrospective provenance, not model inputs.",
        "time_condition": "MedWorld uses signed realized gaps for representation learning, not a fixed-horizon prospective forecast.",
        "reports": "Findings/impression or bounded unsectioned fallback; missing/empty reports excluded from temporal supervision.",
        "asset_policy": "Original images, VQA and reviewed human CXR/MRI annotations are symlinked; no pseudo-mask targets or pixel/feature caches.",
        "evaluation": "Both trained arms: classification/VQA, human-reviewed MIMIC heart/lungs, UCSF and MU MRI, external Montgomery lungs; per-dataset mean IoU/Dice. Native Qwen segmentation N/A.",
    }
    # Counts here describe exported rows, before UnifiedData's final cross-task audit.
    for task in ("segmentation", "vqa"):
        task_counts = supervision["summary"].get("counts", {}).get(task)
        if task_counts is not None:
            manifest["counts"][task] = task_counts
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / ".medworld-prepared").write_text(SCHEMA + "\n")
    if not sum(counts["classification"].values()) or not sum(counts["temporal"].values()):
        raise ValueError("No usable classification observations or temporal pairs; output not published")
    return manifest


def build_dataset(args):
    """Publish a complete directory; explicit replacement archives the prior bundle."""
    destination = args.output_dir.expanduser().absolute()
    # Keep the project's stable alias intact; publish beside its central target.
    if destination.is_symlink():
        destination = destination.resolve(strict=True)
    previous = None
    if destination.exists() or destination.is_symlink():
        if (not args.replace or destination.is_symlink() or
                not (destination / ".medworld-prepared").is_file()):
            raise FileExistsError(f"Output already exists; use a new --output-dir or --replace for a marked bundle: {destination}")
        previous = destination.with_name(destination.name + "_previous_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        if previous.exists():
            raise FileExistsError(previous)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        manifest = _build(args, staging)
        if previous is not None:
            destination.rename(previous)
            LOG.info("Archived previous bundle to %s", previous)
        elif destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Output appeared during build: {destination}")
        try:
            staging.rename(destination)
        except BaseException:
            if previous is not None and not destination.exists():
                previous.rename(destination)
            raise
    except BaseException:
        shutil.rmtree(staging)
        raise
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=CODE_ROOT / "data/medworld_0922")
    parser.add_argument("--replace", action="store_true", help="Archive and replace a marked generated dataset after building succeeds.")
    parser.add_argument("--cxr-root", type=Path, default=CODE_ROOT / "data/MIMIC_CXR")
    parser.add_argument("--iv-root", type=Path, default=CODE_ROOT / "data/mimic-iv-3.1")
    parser.add_argument("--vqa-root", type=Path, default=CODE_ROOT / "data/MIMIC_CXR_VQA/MIMIC-Ext-MIMIC-CXR-VQA/dataset")
    parser.add_argument("--human-cxr-root", type=Path, default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--ucsf-root", type=Path, default=CODE_ROOT / "data/UCSF-ALPTDG")
    parser.add_argument("--mu-root", type=Path, default=CODE_ROOT / "data/MU-Glioma-Post")
    parser.add_argument("--montgomery-root", type=Path, default=CODE_ROOT / "data/medworld/dense/montgomery")
    parser.add_argument("--mri-axial-stride", type=int, default=1, help="Keep every Nth axial slice selected from image-only brain extent.")
    parser.add_argument("--pair-mode", choices=("adjacent", "random", "adjacent_random", "all"), default="adjacent_random")
    parser.add_argument("--random-pairs-per-patient", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-gap-hours", type=float, default=1.0)
    parser.add_argument("--max-gap-days", type=float, default=365.0)
    parser.add_argument("--max-report-chars", type=int, default=6000)
    parser.add_argument("--max-patients", type=int, default=0, help="Seeded source patient limit; 0 uses all (VQA/segmentation remain complete).")
    parser.add_argument("--linkage", choices=("same_admission", "patient"), default="same_admission")
    parser.add_argument("--run-dir", type=Path, default=Path(__file__).resolve().parent / "runs" / ("build_" + datetime.now().strftime("%Y%m%d")))
    args = parser.parse_args(argv)
    if args.random_pairs_per_patient < 0 or args.max_patients < 0 or args.max_report_chars <= 0 or args.mri_axial_stride <= 0:
        parser.error("Pair/patient limits must be nonnegative and max-report-chars positive")
    if (not math.isfinite(args.min_gap_hours) or not math.isfinite(args.max_gap_days)
            or args.min_gap_hours < 0 or args.max_gap_days <= 0 or args.max_gap_days * 24 < args.min_gap_hours):
        parser.error("Invalid gap bounds")
    for key in ("cxr_root", "iv_root", "vqa_root", "human_cxr_root", "ucsf_root", "mu_root", "montgomery_root"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    return args


def main(argv=None):
    args = parse_args(argv)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.run_dir / "build.log")])
    manifest = build_dataset(args)
    LOG.info("Published %s: %s", args.output_dir, json.dumps(manifest["counts"]))
    (args.run_dir / "result.json").write_text(json.dumps({"output_dir": str(args.output_dir),
        "manifest_sha256": sha256(args.output_dir / "manifest.json"), "counts": manifest["counts"]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
