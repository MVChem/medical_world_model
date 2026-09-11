#!/usr/bin/env python3
"""Build small, leakage-aware current-state -> future-state MIMIC-CXR examples.

The script uses only the Python standard library.  It joins the official
MIMIC-CXR-JPG metadata, split, and CheXpert CSV files with the local JPGs and
radiology reports, pairs chronologically adjacent same-patient studies, and
writes both human-readable and machine-readable outputs.
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import gzip
import hashlib
import html
import json
import math
import os
import re
import shutil
import sys
import tempfile
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import TextIO

try:
    from .forecast_contract import (
        build_coarse_horizon_prompt,
        horizon_bin_from_elapsed_hours,
    )
except ImportError:  # Support `python build_mimic_transitions.py`.
    from forecast_contract import (  # type: ignore[no-redef]
        build_coarse_horizon_prompt,
        horizon_bin_from_elapsed_hours,
    )

SCHEMA_VERSION = "mimic-current-future-v1"
DATASET_NAME = "MIMIC-CXR-JPG-2.0.0"
DATASET_DISPLAY_NAME = "MIMIC-CXR-JPG v2.0.0"
OUTPUT_MARKER = ".mimic_transition_output.json"

# Keep the official column order. "No Finding" is a meta-label and is not used
# to count binary pathology-label flips below.
LABEL_COLUMNS = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
)
PATHOLOGY_COLUMNS = tuple(label for label in LABEL_COLUMNS if label != "No Finding")

LABEL_NOT_MENTIONED = -2
LABEL_UNCERTAIN = -1
LABEL_ABSENT = 0
LABEL_PRESENT = 1
LABEL_NAMES = {
    LABEL_NOT_MENTIONED: "not_mentioned",
    LABEL_UNCERTAIN: "uncertain",
    LABEL_ABSENT: "absent",
    LABEL_PRESENT: "present",
}

SECTION_HEADING_RE = re.compile(r"(?m)^[ \t]*([A-Z][A-Z0-9 /_()\-]{1,60}):[ \t]*")


@dataclass(slots=True)
class ImageInfo:
    dicom_id: str
    view: str
    timestamp: datetime
    image_path: Path
    relative_path: str
    rows: int | None = None
    columns: int | None = None
    procedure: str | None = None
    orientation: str | None = None


@dataclass(slots=True)
class Study:
    subject_id: str
    study_id: str
    timestamp: datetime
    latest_image_timestamp: datetime
    report_path: Path
    report_relative_path: str
    images_by_view: dict[str, ImageInfo] = field(default_factory=dict)
    split: str | None = None
    labels: tuple[int, ...] | None = None

    @property
    def acquisition_span_seconds(self) -> float:
        return (self.latest_image_timestamp - self.timestamp).total_seconds()


@dataclass(slots=True)
class Candidate:
    source: Study
    target: Study
    source_image: ImageInfo
    target_image: ImageInfo
    matched_view: str
    source_order: int
    target_order: int
    elapsed_hours: float
    binary_label_flips: int
    future_positive_findings: int

    def rank_key(self) -> tuple[object, ...]:
        """Prefer more binary label flips, then populated short follow-ups."""

        # Twenty-four hours is a useful demonstration horizon.  This affects
        # only gallery selection, not the validity of a pair.
        horizon_distance = abs(math.log1p(self.elapsed_hours) - math.log1p(24.0))
        return (
            -self.binary_label_flips,
            -self.future_positive_findings,
            horizon_distance,
            self.elapsed_hours,
            self.source.subject_id,
            self.source.study_id,
            self.target.study_id,
        )


def _stable_random_key(seed: int, namespace: str, *parts: str) -> tuple[str, ...]:
    """Return a process-independent pseudorandom ordering key."""

    payload = "\0".join((str(seed), namespace, *parts)).encode("utf-8")
    return (hashlib.sha256(payload).hexdigest(), *parts)


def _candidate_identity(candidate: Candidate) -> tuple[str, ...]:
    return (
        candidate.source.subject_id,
        candidate.source.study_id,
        candidate.target.study_id,
        candidate.source_image.dicom_id,
        candidate.target_image.dicom_id,
        candidate.matched_view,
    )


def select_candidates(
    candidates: Sequence[Candidate], config: BuildConfig, audit: Audit
) -> list[Candidate]:
    """Order/select eligible pairs according to the declared cohort policy.

    In deterministic-random one-per-patient mode, pair choice is made within
    each patient first. Patient ordering is then determined from only the seed
    and patient identifier. Neither key contains target report or label values,
    and patients with more eligible pairs do not gain more chances to enter a
    size-limited cohort.
    """

    if config.selection_strategy == "change_enriched":
        return sorted(candidates, key=Candidate.rank_key)

    if not config.one_per_patient:
        return sorted(
            candidates,
            key=lambda candidate: _stable_random_key(
                config.seed, "pair-order", *_candidate_identity(candidate)
            ),
        )

    by_patient: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_patient[candidate.source.subject_id].append(candidate)
    chosen: list[Candidate] = []
    for patient_candidates in by_patient.values():
        chosen.append(
            min(
                patient_candidates,
                key=lambda candidate: _stable_random_key(
                    config.seed, "within-patient-pair", *_candidate_identity(candidate)
                ),
            )
        )
    audit.increment("candidates_skipped_one_per_patient", len(candidates) - len(chosen))
    chosen.sort(
        key=lambda candidate: _stable_random_key(
            config.seed, "patient-order", candidate.source.subject_id
        )
    )
    return chosen


@dataclass(slots=True)
class BuildConfig:
    mimic_cxr_root: Path
    output_dir: Path
    num_examples: int | None = 5
    split: str = "train"
    view: str = "AP"
    min_gap_hours: float = 1.0
    max_gap_days: float = 365.0
    min_label_flips: int = 0
    one_per_patient: bool = True
    selection_strategy: str = "deterministic_random"
    seed: int = 42
    asset_mode: str = "symlink"
    render_gallery: bool = True
    max_report_chars: int = 6000
    curated_pairs: tuple[dict[str, str], ...] = ()
    curated_pairs_path: Path | None = None


@dataclass(slots=True)
class BuildResult:
    packets: list[dict[str, object]]
    forecast_rows: list[dict[str, object]]
    input_rows: list[dict[str, object]]
    target_rows: list[dict[str, object]]
    summary: dict[str, object]
    output_paths: dict[str, Path]


@dataclass(slots=True)
class Audit:
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def increment(self, key: str, amount: int = 1) -> None:
        self.counts[key] += amount

    def as_dict(self) -> dict[str, int]:
        return dict(sorted(self.counts.items()))


def _open_csv(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def _find_table(root: Path, stem: str) -> Path:
    candidates = (root / f"{stem}.csv", root / f"{stem}.csv.gz")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    choices = " or ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Required table not found: {choices}")


def resolve_mimic_cxr_root(path: Path) -> Path:
    """Accept either the MIMIC parent directory or its MIMIC_CXR child."""

    root = path.expanduser().resolve()
    metadata_stem = "mimic-cxr-2.0.0-metadata"
    if any(
        (root / f"{metadata_stem}{suffix}").is_file() for suffix in (".csv", ".csv.gz")
    ):
        return root
    child = root / "MIMIC_CXR"
    if any(
        (child / f"{metadata_stem}{suffix}").is_file() for suffix in (".csv", ".csv.gz")
    ):
        return child
    raise FileNotFoundError(
        f"Could not find MIMIC-CXR metadata under {root} or {child}. "
        "Pass --mimic-cxr-root pointing to the MIMIC_CXR directory."
    )


def _plain_identifier(value: object, prefix: str) -> str:
    text = str(value or "").strip()
    return text.removeprefix(prefix)


def load_curated_pairs(path: Path) -> tuple[dict[str, str], ...]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    entries = payload.get("pairs") if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            f"Curated pair file must contain a nonempty 'pairs' list: {resolved}"
        )
    required = {
        "subject_id",
        "source_study_id",
        "target_study_id",
        "category",
        "audit_note",
    }
    output: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for index, raw in enumerate(entries):
        if not isinstance(raw, dict) or not required.issubset(raw):
            raise ValueError(
                f"Curated pair {index} must contain fields {sorted(required)}"
            )
        entry = {key: str(value) for key, value in raw.items()}
        entry["subject_id"] = _plain_identifier(entry["subject_id"], "p")
        entry["source_study_id"] = _plain_identifier(entry["source_study_id"], "s")
        entry["target_study_id"] = _plain_identifier(entry["target_study_id"], "s")
        key = (
            entry["subject_id"],
            entry["source_study_id"],
            entry["target_study_id"],
        )
        if key in seen:
            raise ValueError(f"Duplicate curated pair: {key}")
        seen.add(key)
        output.append(entry)
    return tuple(output)


def _digits(value: str | None) -> str:
    return re.sub(r"\D", "", value or "")


def parse_study_datetime(
    date_text: str | None, time_text: str | None
) -> datetime | None:
    """Parse MIMIC StudyDate/StudyTime without using IDs as chronology."""

    date_digits = _digits((date_text or "").split(".", 1)[0])
    if len(date_digits) != 8:
        return None

    raw_time = (time_text or "").strip()
    if not re.fullmatch(r"\d{1,6}(?:\.\d+)?", raw_time):
        return None
    whole_time, dot, fraction = raw_time.partition(".")
    time_digits = _digits(whole_time).zfill(6)
    if len(time_digits) > 6:
        time_digits = time_digits[:6]
    fraction_digits = _digits(fraction)[:6].ljust(6, "0") if dot else "000000"
    try:
        # MIMIC StudyDate/StudyTime are intentionally timezone-naive fields.
        base = datetime.strptime(date_digits + time_digits, "%Y%m%d%H%M%S")  # noqa: DTZ007
        return base.replace(microsecond=int(fraction_digits))
    except ValueError:
        return None


def _to_optional_int(value: str | None) -> int | None:
    try:
        return int(float(value or ""))
    except (TypeError, ValueError):
        return None


def _parse_label(value: str | None) -> int:
    text = (value or "").strip()
    if not text:
        return LABEL_NOT_MENTIONED
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"Unexpected CheXpert label value: {value!r}") from exc
    if number == 1.0:
        return LABEL_PRESENT
    if number == 0.0:
        return LABEL_ABSENT
    if number == -1.0:
        return LABEL_UNCERTAIN
    raise ValueError(f"Unexpected CheXpert label value: {value!r}")


def _image_rank(image: ImageInfo) -> tuple[object, ...]:
    area = (image.rows or 0) * (image.columns or 0)
    return (-area, image.dicom_id)


def _allowed_views(requested_view: str) -> set[str]:
    if requested_view == "frontal":
        return {"AP", "PA"}
    return {requested_view.upper()}


def load_studies(
    root: Path, requested_view: str, audit: Audit
) -> dict[tuple[str, str], Study]:
    metadata_path = _find_table(root, "mimic-cxr-2.0.0-metadata")
    allowed_views = _allowed_views(requested_view)
    studies: dict[tuple[str, str], Study] = {}

    with _open_csv(metadata_path) as handle:
        reader = csv.DictReader(handle)
        required = {
            "dicom_id",
            "subject_id",
            "study_id",
            "ViewPosition",
            "StudyDate",
            "StudyTime",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{metadata_path} is missing columns: {sorted(missing)}")
        for row in reader:
            audit.increment("metadata_rows")
            timestamp = parse_study_datetime(row.get("StudyDate"), row.get("StudyTime"))
            if timestamp is None:
                audit.increment("metadata_rows_rejected_timestamp")
                continue
            subject_id = (row.get("subject_id") or "").strip()
            study_id = (row.get("study_id") or "").strip()
            dicom_id = (row.get("dicom_id") or "").strip()
            if not (subject_id and study_id and dicom_id):
                audit.increment("metadata_rows_rejected_identifier")
                continue
            patient_dir = root / "files" / f"p{subject_id[:2]}" / f"p{subject_id}"
            study_dir = patient_dir / f"s{study_id}"
            key = (subject_id, study_id)
            study = studies.get(key)
            if study is None:
                study = Study(
                    subject_id=subject_id,
                    study_id=study_id,
                    timestamp=timestamp,
                    latest_image_timestamp=timestamp,
                    report_path=patient_dir / f"s{study_id}.txt",
                    report_relative_path=f"files/p{subject_id[:2]}/p{subject_id}/s{study_id}.txt",
                )
                studies[key] = study
            else:
                study.timestamp = min(study.timestamp, timestamp)
                study.latest_image_timestamp = max(
                    study.latest_image_timestamp, timestamp
                )

            # Every study remains in the patient timeline. Only requested
            # projections are eligible as the paired primary image.
            view = (row.get("ViewPosition") or "").strip().upper()
            if view not in allowed_views:
                audit.increment("metadata_rows_not_requested_view")
                continue
            image = ImageInfo(
                dicom_id=dicom_id,
                view=view,
                timestamp=timestamp,
                image_path=study_dir / f"{dicom_id}.jpg",
                relative_path=f"files/p{subject_id[:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg",
                rows=_to_optional_int(row.get("Rows")),
                columns=_to_optional_int(row.get("Columns")),
                procedure=(row.get("PerformedProcedureStepDescription") or "").strip()
                or None,
                orientation=(
                    row.get("PatientOrientationCodeSequence_CodeMeaning") or ""
                ).strip()
                or None,
            )
            previous = study.images_by_view.get(view)
            if previous is None or _image_rank(image) < _image_rank(previous):
                study.images_by_view[view] = image
            if previous is not None:
                audit.increment("eligible_extra_images_not_selected")

    audit.increment("all_timeline_studies_from_metadata", len(studies))
    audit.increment(
        "studies_with_requested_view",
        sum(bool(study.images_by_view) for study in studies.values()),
    )
    audit.increment(
        "studies_with_multiple_acquisition_times",
        sum(study.acquisition_span_seconds > 0 for study in studies.values()),
    )
    audit.increment(
        "studies_with_acquisition_span_over_one_hour",
        sum(study.acquisition_span_seconds > 3600 for study in studies.values()),
    )
    return studies


def attach_splits(
    root: Path, studies: Mapping[tuple[str, str], Study], audit: Audit
) -> None:
    split_path = _find_table(root, "mimic-cxr-2.0.0-split")
    with _open_csv(split_path) as handle:
        reader = csv.DictReader(handle)
        required = {"subject_id", "study_id", "split"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{split_path} is missing columns: {sorted(missing)}")
        for row in reader:
            key = (
                (row.get("subject_id") or "").strip(),
                (row.get("study_id") or "").strip(),
            )
            study = studies.get(key)
            if study is None:
                continue
            split = (row.get("split") or "").strip().lower()
            if split not in {"train", "validate", "test"}:
                raise ValueError(f"Invalid split {split!r} for subject/study {key}")
            if study.split is not None and study.split != split:
                raise ValueError(
                    f"Conflicting splits for subject/study {key}: {study.split!r} versus {split!r}"
                )
            study.split = split

    missing = [key for key, study in studies.items() if study.split is None]
    if missing:
        raise ValueError(
            f"Split table did not cover {len(missing)} metadata studies; first missing key: {missing[0]}"
        )
    patient_splits: dict[str, set[str]] = defaultdict(set)
    for study in studies.values():
        assert study.split is not None
        patient_splits[study.subject_id].add(study.split)
    conflicts = {
        subject_id: splits
        for subject_id, splits in patient_splits.items()
        if len(splits) != 1
    }
    if conflicts:
        subject_id, splits = next(iter(conflicts.items()))
        raise ValueError(
            f"Patient-level split violation for subject {subject_id}: {sorted(splits)}"
        )
    audit.increment("patients_with_validated_single_split", len(patient_splits))


def attach_labels(
    root: Path, studies: Mapping[tuple[str, str], Study], audit: Audit
) -> None:
    label_path = _find_table(root, "mimic-cxr-2.0.0-chexpert")
    with _open_csv(label_path) as handle:
        reader = csv.DictReader(handle)
        required = {"subject_id", "study_id", *LABEL_COLUMNS}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{label_path} is missing columns: {sorted(missing)}")
        for row in reader:
            key = (
                (row.get("subject_id") or "").strip(),
                (row.get("study_id") or "").strip(),
            )
            study = studies.get(key)
            if study is None:
                continue
            study.labels = tuple(
                _parse_label(row.get(label)) for label in LABEL_COLUMNS
            )
            audit.increment("eligible_studies_with_labels")


def label_state(labels: Sequence[int]) -> dict[str, str]:
    return {
        name: LABEL_NAMES[value]
        for name, value in zip(LABEL_COLUMNS, labels, strict=True)
    }


def _change_name(source: int, target: int) -> str:
    if source == LABEL_ABSENT and target == LABEL_PRESENT:
        return "changed_to_present"
    if source == LABEL_PRESENT and target == LABEL_ABSENT:
        return "changed_to_absent"
    if source == target and source in {LABEL_ABSENT, LABEL_PRESENT}:
        return "stable"
    if source == target == LABEL_UNCERTAIN:
        return "stable_uncertain"
    if LABEL_NOT_MENTIONED in {source, target}:
        return "not_comparable"
    return "indeterminate"


def state_delta(
    source_labels: Sequence[int], target_labels: Sequence[int]
) -> dict[str, object]:
    by_finding: dict[str, dict[str, str]] = {}
    changed: list[dict[str, str]] = []
    stable_present: list[str] = []
    for index, finding in enumerate(LABEL_COLUMNS):
        source = source_labels[index]
        target = target_labels[index]
        change = _change_name(source, target)
        item = {
            "current": LABEL_NAMES[source],
            "future": LABEL_NAMES[target],
            "change": change,
        }
        by_finding[finding] = item
        if finding in PATHOLOGY_COLUMNS and change in {
            "changed_to_present",
            "changed_to_absent",
        }:
            changed.append({"finding": finding, **item})
        if (
            finding in PATHOLOGY_COLUMNS
            and change == "stable"
            and target == LABEL_PRESENT
        ):
            stable_present.append(finding)
    return {
        "by_finding": by_finding,
        "binary_chexpert_label_flips": changed,
        "stable_present_findings": stable_present,
        "num_binary_chexpert_label_flips": len(changed),
        "label_source": "MIMIC-CXR CheXpert report labeler",
        "limitation": (
            "These are CheXpert report-label flips, not verified disease transitions. "
            "They do not reliably encode onset, resolution, improved, or worsened severity."
        ),
    }


def _count_binary_label_flips(source: Sequence[int], target: Sequence[int]) -> int:
    return sum(
        _change_name(source[index], target[index])
        in {"changed_to_present", "changed_to_absent"}
        for index, finding in enumerate(LABEL_COLUMNS)
        if finding in PATHOLOGY_COLUMNS
    )


def _count_future_positive(labels: Sequence[int]) -> int:
    return sum(
        labels[index] == LABEL_PRESENT
        for index, finding in enumerate(LABEL_COLUMNS)
        if finding in PATHOLOGY_COLUMNS
    )


def find_candidates(
    studies: Iterable[Study], config: BuildConfig, audit: Audit
) -> list[Candidate]:
    by_patient: dict[str, list[Study]] = defaultdict(list)
    for study in studies:
        by_patient[study.subject_id].append(study)

    audit.increment("patients_with_timeline_studies", len(by_patient))
    candidates: list[Candidate] = []
    max_gap_hours = config.max_gap_days * 24.0
    for patient_studies in by_patient.values():
        patient_studies.sort(key=lambda item: (item.timestamp, item.study_id))
        timestamp_counts = Counter(study.timestamp for study in patient_studies)
        patient_candidates: list[Candidate] = []
        for source_order, (source, target) in enumerate(pairwise(patient_studies)):
            audit.increment("adjacent_pairs_considered")
            if (
                timestamp_counts[source.timestamp] > 1
                or timestamp_counts[target.timestamp] > 1
            ):
                audit.increment("pairs_rejected_ambiguous_tied_timestamp")
                continue
            if source.split is None or target.split is None:
                audit.increment("pairs_rejected_missing_split")
                continue
            if config.split != "all" and (
                source.split != config.split or target.split != config.split
            ):
                audit.increment("pairs_rejected_requested_split")
                continue
            if source.labels is None or target.labels is None:
                audit.increment("pairs_rejected_missing_labels")
                continue
            if (
                not source.report_path.is_file()
                or source.report_path.stat().st_size == 0
            ):
                audit.increment("pairs_rejected_missing_source_report")
                continue
            common_views = set(source.images_by_view).intersection(
                target.images_by_view
            )
            preference = ("PA", "AP") if config.view == "frontal" else (config.view,)
            matched_view = next(
                (view for view in preference if view in common_views), None
            )
            if matched_view is None:
                audit.increment("pairs_rejected_view_mismatch")
                continue
            source_image = source.images_by_view[matched_view]
            target_image = target.images_by_view[matched_view]
            if not source_image.image_path.is_file():
                audit.increment("pairs_rejected_missing_source_image")
                continue
            if not target_image.image_path.is_file():
                audit.increment("pairs_rejected_missing_target_image")
                continue
            if source.split != target.split:
                audit.increment("pairs_rejected_split_mismatch")
                continue
            if source.latest_image_timestamp >= target.timestamp:
                audit.increment("pairs_rejected_overlapping_acquisition_windows")
                continue
            elapsed_hours = (
                target_image.timestamp - source_image.timestamp
            ).total_seconds() / 3600.0
            if elapsed_hours <= 0:
                audit.increment("pairs_rejected_nonpositive_selected_image_time")
                continue
            if elapsed_hours < config.min_gap_hours:
                audit.increment("pairs_rejected_too_short")
                continue
            if elapsed_hours > max_gap_hours:
                audit.increment("pairs_rejected_too_long")
                continue
            binary_label_flips = _count_binary_label_flips(source.labels, target.labels)
            if binary_label_flips < config.min_label_flips:
                audit.increment("pairs_rejected_insufficient_label_flips")
                continue
            patient_candidates.append(
                Candidate(
                    source=source,
                    target=target,
                    source_image=source_image,
                    target_image=target_image,
                    matched_view=matched_view,
                    source_order=source_order,
                    target_order=source_order + 1,
                    elapsed_hours=elapsed_hours,
                    binary_label_flips=binary_label_flips,
                    future_positive_findings=_count_future_positive(target.labels),
                )
            )
        candidates.extend(patient_candidates)

    audit.increment("valid_ranked_candidates", len(candidates))
    return candidates


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_report_sections(text: str, max_chars: int = 6000) -> dict[str, str | None]:
    """Extract the two clinically useful sections without external NLP tools."""

    normalized_newlines = text.replace("\r\n", "\n").replace("\r", "\n")
    matches = list(SECTION_HEADING_RE.finditer(normalized_newlines))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        heading = _normalize_text(match.group(1)).upper()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(normalized_newlines)
        )
        value = _normalize_text(normalized_newlines[match.end() : end])
        if value:
            sections[heading] = value

    findings = sections.get("FINDINGS")
    impression = sections.get("IMPRESSION")
    combined = sections.get("FINDINGS AND IMPRESSION") or sections.get(
        "FINDINGS/IMPRESSION"
    )
    if not findings and combined:
        findings = combined
    if not impression and combined:
        impression = combined

    # A small number of valid reports lack standard section headers. Preserve
    # a bounded normalized fallback so they can still be audited.
    fallback = None
    if not findings and not impression:
        without_empty_headings = SECTION_HEADING_RE.sub("", normalized_newlines)
        fallback_candidate = _normalize_text(without_empty_headings)
        if fallback_candidate.upper() not in {"", "FINAL REPORT"}:
            fallback = fallback_candidate
    return {
        "findings": findings[:max_chars] if findings else None,
        "impression": impression[:max_chars] if impression else None,
        "unsectioned_report": fallback[:max_chars] if fallback else None,
    }


def read_report(path: Path, max_chars: int) -> dict[str, str | None]:
    return extract_report_sections(
        path.read_text(encoding="utf-8", errors="replace"), max_chars=max_chars
    )


def read_optional_evaluation_report(
    path: Path, max_chars: int
) -> dict[str, str | None]:
    """Read target-side report metadata without making it cohort eligibility."""

    try:
        if not path.is_file() or path.stat().st_size == 0:
            return {"findings": None, "impression": None, "unsectioned_report": None}
        return read_report(path, max_chars)
    except OSError:
        return {"findings": None, "impression": None, "unsectioned_report": None}


def report_for_prompt(report: Mapping[str, str | None]) -> str:
    parts = []
    if report.get("findings"):
        parts.append(f"FINDINGS: {report['findings']}")
    if report.get("impression"):
        parts.append(f"IMPRESSION: {report['impression']}")
    if not parts and report.get("unsectioned_report"):
        parts.append(str(report["unsectioned_report"]))
    return "\n".join(parts)


def format_elapsed_time(hours: float) -> str:
    if hours < 48:
        return f"{hours:.1f} hours"
    return f"{hours / 24.0:.1f} days"


def build_forecast_prompt(
    current_report: Mapping[str, str | None], elapsed_hours: float
) -> str:
    """Construct the model prompt from t0 text and a prespecified coarse horizon."""

    rounded_elapsed_hours = round(elapsed_hours, 6)
    return build_coarse_horizon_prompt(
        current_report, horizon_bin_from_elapsed_hours(rounded_elapsed_hours)
    )


def _timestamp_text(value: datetime) -> str:
    return value.isoformat(timespec="microseconds")


def _transition_id(candidate: Candidate) -> str:
    raw = (
        f"{candidate.source.subject_id}|{candidate.source.study_id}|"
        f"{candidate.target.study_id}|{candidate.source_image.dicom_id}|"
        f"{candidate.target_image.dicom_id}|{candidate.matched_view}"
    )
    # The join key must be opaque: spelling a target study/DICOM ID inside an
    # inference input is itself future-side metadata leakage.
    digest = hashlib.sha256(raw.encode("ascii")).hexdigest()[:20]
    return f"mimiccxr_{digest}"


def _candidate_key(candidate: Candidate) -> tuple[str, str, str]:
    return (
        candidate.source.subject_id,
        candidate.source.study_id,
        candidate.target.study_id,
    )


def _image_record(image: ImageInfo) -> dict[str, object]:
    return {
        "path": str(image.image_path.resolve()),
        "relative_path": image.relative_path,
        "dicom_id": image.dicom_id,
        "view": image.view,
        "acquisition_timestamp": _timestamp_text(image.timestamp),
        "rows": image.rows,
        "columns": image.columns,
        "procedure": image.procedure,
        "orientation": image.orientation,
    }


def make_packet(
    candidate: Candidate, max_report_chars: int
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    source = candidate.source
    target = candidate.target
    assert source.labels is not None and target.labels is not None
    source_report = read_report(source.report_path, max_report_chars)
    target_report = read_optional_evaluation_report(
        target.report_path, max_report_chars
    )
    elapsed_hours = round(candidate.elapsed_hours, 6)
    horizon_bin = horizon_bin_from_elapsed_hours(elapsed_hours)
    prompt = build_coarse_horizon_prompt(source_report, horizon_bin)
    transition_id = _transition_id(candidate)
    delta = state_delta(source.labels, target.labels)
    source_findings = label_state(source.labels)
    target_findings = label_state(target.labels)

    packet: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "transition_id": transition_id,
        "source_dataset": DATASET_NAME,
        "patient_id": f"p{source.subject_id}",
        "split": source.split,
        "pairing": "adjacent_studies_in_full_patient_timeline",
        "matched_view": candidate.matched_view,
        "current_state": {
            "study_id": f"s{source.study_id}",
            "timeline_order": candidate.source_order,
            "timestamp": _timestamp_text(source.timestamp),
            "study_acquisition_span_seconds": source.acquisition_span_seconds,
            "image": _image_record(candidate.source_image),
            "report_path": str(source.report_path.resolve()),
            "report_relative_path": source.report_relative_path,
            "text": source_report,
            "structured_findings": source_findings,
            "structured_findings_source": "MIMIC-CXR CheXpert report labeler",
        },
        "interval": {
            "elapsed_hours": elapsed_hours,
            "elapsed_human": format_elapsed_time(candidate.elapsed_hours),
            "horizon_bin": horizon_bin,
            "horizon_policy": "prespecified_coarse_bin_v1",
            "definition": "selected_target_image_time - selected_source_image_time",
            "study_anchor_elapsed_hours": round(
                (target.timestamp - source.timestamp).total_seconds() / 3600.0, 6
            ),
            "actions": [],
            "actions_status": "not_linked_in_this_cxr_only_example",
        },
        "future_state": {
            "role": "observed_followup_anchor",
            "study_id": f"s{target.study_id}",
            "timeline_order": candidate.target_order,
            "timestamp": _timestamp_text(target.timestamp),
            "study_acquisition_span_seconds": target.acquisition_span_seconds,
            "image": _image_record(candidate.target_image),
            "report_path": str(target.report_path.resolve()),
            "report_relative_path": target.report_relative_path,
            "text_for_evaluation_only": target_report,
            "structured_findings_for_evaluation_only": target_findings,
            "structured_findings_source": "MIMIC-CXR CheXpert report labeler",
        },
        "state_delta_for_evaluation_only": delta,
        "task_contract": {
            "task_type": "forecasting",
            "available_model_inputs": [
                "current_state.image",
                "current_state.text",
                "current_state.structured_findings",
                "interval.horizon_bin",
            ],
            "fields_rendered_into_prompt": [
                "current_state.text",
                "interval.horizon_bin",
            ],
            "supervision_or_evaluation_only": [
                "future_state.image",
                "future_state.text_for_evaluation_only",
                "future_state.structured_findings_for_evaluation_only",
                "state_delta_for_evaluation_only",
            ],
        },
    }

    # Trainer-compatible row. Extra fields are metadata; prompt is deliberately
    # assembled only from current_report + coarse horizon above.
    forecast_row: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "task_type": "forecasting",
        "transition_id": transition_id,
        "source_dataset": DATASET_NAME,
        "patient_id": f"p{source.subject_id}",
        "split": source.split,
        "source_study_id": f"s{source.study_id}",
        "target_study_id": f"s{target.study_id}",
        "target_role": "observed_followup_anchor",
        "source_order": candidate.source_order,
        "target_order": candidate.target_order,
        "view": candidate.matched_view,
        "elapsed_hours": elapsed_hours,
        "horizon_bin": horizon_bin,
        "horizon_policy": "prespecified_coarse_bin_v1",
        "source_image": str(candidate.source_image.image_path.resolve()),
        "target_image": str(candidate.target_image.image_path.resolve()),
        "source_image_relative": candidate.source_image.relative_path,
        "target_image_relative": candidate.target_image.relative_path,
        "prompt": prompt,
        "source_report": source_report,
        "source_state": source_findings,
        "target_state_for_eval_only": target_findings,
        "state_delta_for_eval_only": delta,
        "future_report_for_eval_only": target_report,
    }

    # This is the safest inference-time boundary: it contains no target path,
    # future report, future state, or delta at all.
    input_row: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "task_type": "forecasting_input",
        "transition_id": transition_id,
        "source_dataset": DATASET_NAME,
        "patient_id": f"p{source.subject_id}",
        "split": source.split,
        "source_study_id": f"s{source.study_id}",
        "source_order": candidate.source_order,
        "view": candidate.matched_view,
        "horizon_bin": horizon_bin,
        "horizon_policy": "prespecified_coarse_bin_v1",
        "source_image": str(candidate.source_image.image_path.resolve()),
        "source_image_relative": candidate.source_image.relative_path,
        "prompt": prompt,
        "source_report": source_report,
        "source_state": source_findings,
    }
    target_row: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "task_type": "forecasting_target",
        "transition_id": transition_id,
        "source_dataset": DATASET_NAME,
        "patient_id": f"p{source.subject_id}",
        "split": target.split,
        "target_study_id": f"s{target.study_id}",
        "target_order": candidate.target_order,
        "view": candidate.matched_view,
        "target_image": str(candidate.target_image.image_path.resolve()),
        "target_image_relative": candidate.target_image.relative_path,
        "future_report": target_report,
        "target_state": target_findings,
        "state_delta": delta,
    }
    return packet, forecast_row, input_row, target_row


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=False, allow_nan=False)
                + "\n"
            )
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _validate_staged_output(
    staging_dir: Path, expected_rows: int, asset_mode: str, expect_gallery: bool
) -> None:
    names = (
        "transitions.jsonl",
        "forecast_manifest.jsonl",
        "forecast_inputs.jsonl",
        "forecast_targets.jsonl",
    )
    id_lists: list[list[str]] = []
    for name in names:
        rows = _read_jsonl(staging_dir / name)
        if len(rows) != expected_rows:
            raise RuntimeError(
                f"Staged {name} has {len(rows)} rows; expected {expected_rows}"
            )
        id_lists.append([str(row["transition_id"]) for row in rows])
    if any(ids != id_lists[0] for ids in id_lists[1:]):
        raise RuntimeError("Staged manifests do not have identical transition ID order")
    if any(not re.fullmatch(r"mimiccxr_[0-9a-f]{20}", value) for value in id_lists[0]):
        raise RuntimeError("Staged manifests contain a non-opaque transition ID")

    forbidden_input_keys = {
        "elapsed_hours",
        "target_image",
        "target_study_id",
        "future_report",
        "future_report_for_eval_only",
        "target_state",
        "state_delta",
    }
    for row in _read_jsonl(staging_dir / "forecast_inputs.jsonl"):
        overlap = forbidden_input_keys.intersection(row)
        if overlap:
            raise RuntimeError(
                f"Forecast input contains future-side keys: {sorted(overlap)}"
            )

    if expect_gallery and not (staging_dir / "index.html").is_file():
        raise RuntimeError("Staged gallery is missing index.html")
    if not expect_gallery and (
        (staging_dir / "index.html").exists() or (staging_dir / "assets").exists()
    ):
        raise RuntimeError("No-gallery output unexpectedly contains gallery artifacts")
    if expect_gallery and asset_mode != "none":
        assets = list((staging_dir / "assets").iterdir())
        if len(assets) != expected_rows * 2 or not all(
            path.resolve().is_file() for path in assets
        ):
            raise RuntimeError("Staged gallery assets are incomplete")


def _commit_staged_output(staging_dir: Path, output_dir: Path) -> None:
    backup: Path | None = None
    if output_dir.exists():
        if not output_dir.is_dir() or not (output_dir / OUTPUT_MARKER).is_file():
            raise FileExistsError(
                f"Refusing to replace unrecognized output directory: {output_dir}. "
                "Choose a fresh --output-dir."
            )
        backup = output_dir.parent / f".{output_dir.name}.backup-{uuid.uuid4().hex}"
        output_dir.rename(backup)
    try:
        staging_dir.rename(output_dir)
    except Exception:
        if backup is not None and not output_dir.exists():
            backup.rename(output_dir)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def _materialize_asset(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink():
            if mode == "symlink" and destination.resolve() == source.resolve():
                return
        elif (
            mode == "copy"
            and destination.is_file()
            and filecmp.cmp(destination, source, shallow=False)
        ):
            return
        raise FileExistsError(
            f"Refusing to replace an unexpected existing asset: {destination}. "
            "Choose a fresh --output-dir or remove that generated asset explicitly."
        )
    if mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "copy":
        shutil.copy2(source, destination)
    elif mode != "none":
        raise ValueError(f"Unknown asset mode: {mode}")


def _positive_findings(state: Mapping[str, str]) -> list[str]:
    return [name for name in PATHOLOGY_COLUMNS if state.get(name) == "present"]


def _report_html(report: Mapping[str, str | None]) -> str:
    blocks = []
    for key, title in (
        ("findings", "Findings"),
        ("impression", "Impression"),
        ("unsectioned_report", "Report"),
    ):
        if report.get(key):
            blocks.append(f"<h4>{title}</h4><p>{html.escape(str(report[key]))}</p>")
    return "".join(blocks) or "<p class=muted>No report text extracted.</p>"


def _pills(items: Sequence[str]) -> str:
    if not items:
        return '<span class="pill none">no positive structured finding</span>'
    return "".join(f'<span class="pill">{html.escape(item)}</span>' for item in items)


def _technique_html(image: Mapping[str, object]) -> str:
    parts = [str(image.get("view") or "view unknown")]
    if image.get("orientation"):
        parts.append(str(image["orientation"]))
    if image.get("procedure"):
        parts.append(str(image["procedure"]))
    if image.get("acquisition_timestamp"):
        parts.append(f"image time {image['acquisition_timestamp']}")
    return html.escape(" · ".join(parts))


def _image_src(
    packet: Mapping[str, object], role: str, output_dir: Path, asset_mode: str
) -> str:
    transition_id = str(packet["transition_id"])
    state_key = "current_state" if role == "current" else "future_state"
    state = packet[state_key]
    assert isinstance(state, Mapping)
    image_record = state["image"]
    assert isinstance(image_record, Mapping)
    source = Path(str(image_record["path"]))
    if asset_mode == "none":
        return source.as_uri()
    suffix = source.suffix.lower() or ".jpg"
    relative = Path("assets") / f"{transition_id}_{role}{suffix}"
    _materialize_asset(source, output_dir / relative, asset_mode)
    return relative.as_posix()


def render_gallery(
    packets: Sequence[Mapping[str, object]], output_dir: Path, asset_mode: str
) -> str:
    cards: list[str] = []
    for packet in packets:
        current = packet["current_state"]
        future = packet["future_state"]
        interval = packet["interval"]
        delta = packet["state_delta_for_evaluation_only"]
        assert isinstance(current, Mapping)
        assert isinstance(future, Mapping)
        assert isinstance(interval, Mapping)
        assert isinstance(delta, Mapping)
        curation = packet.get("curation")
        curation_html = ""
        if isinstance(curation, Mapping):
            curation_html = (
                '<p class="curation"><strong>Selection category: '
                f"{html.escape(str(curation.get('category', 'curated')))}</strong> — "
                f"{html.escape(str(curation.get('audit_note', '')))}</p>"
            )
        current_text = current["text"]
        future_text = future["text_for_evaluation_only"]
        current_state = current["structured_findings"]
        future_state = future["structured_findings_for_evaluation_only"]
        current_image = current["image"]
        future_image = future["image"]
        assert isinstance(current_text, Mapping)
        assert isinstance(future_text, Mapping)
        assert isinstance(current_state, Mapping)
        assert isinstance(future_state, Mapping)
        assert isinstance(current_image, Mapping)
        assert isinstance(future_image, Mapping)
        changed = delta.get("binary_chexpert_label_flips", [])
        change_items = "".join(
            "<li><strong>{finding}</strong>: {current} &rarr; {future} "
            "(<span class=change>{change}</span>)</li>".format(
                finding=html.escape(str(item["finding"])),
                current=html.escape(str(item["current"])),
                future=html.escape(str(item["future"])),
                change=html.escape(str(item["change"])),
            )
            for item in changed
            if isinstance(item, Mapping)
        )
        if not change_items:
            change_items = "<li>No absent/present binary CheXpert label flip.</li>"
        source_src = _image_src(packet, "current", output_dir, asset_mode)
        target_src = _image_src(packet, "future", output_dir, asset_mode)
        cards.append(
            f"""
<article class="transition">
  <header>
    <h2>{html.escape(str(packet["transition_id"]))}</h2>
    <p>{html.escape(str(packet["patient_id"]))} &middot; {html.escape(str(packet["split"]))} split
       &middot; {html.escape(str(current["image"]["view"]))} &middot;
       {html.escape(str(interval["elapsed_human"]))} later</p>
    {curation_html}
  </header>
  <div class="states">
    <section class="state current">
      <span class="source-badge">DATABASE · {DATASET_DISPLAY_NAME}</span>
      <div class="state-title"><span>t0</span><h3>Current state (model input)</h3></div>
      <p class="meta">{html.escape(str(current["study_id"]))} &middot; {html.escape(str(current["timestamp"]))}</p>
      <img loading="lazy" src="{html.escape(source_src)}" alt="Current chest radiograph">
      <p class="meta">{_technique_html(current_image)}</p>
      <div class="pills">{_pills(_positive_findings(current_state))}</div>
      {_report_html(current_text)}
    </section>
    <div class="arrow" aria-hidden="true">&rarr;<small>{html.escape(str(interval["elapsed_human"]))}</small></div>
    <section class="state future">
      <span class="source-badge">DATABASE · {DATASET_DISPLAY_NAME}</span>
      <div class="state-title"><span>t1</span><h3>Observed future (target / eval only)</h3></div>
      <p class="meta">{html.escape(str(future["study_id"]))} &middot; {html.escape(str(future["timestamp"]))}</p>
      <img loading="lazy" src="{html.escape(target_src)}" alt="Observed future chest radiograph">
      <p class="meta">{_technique_html(future_image)}</p>
      <div class="pills">{_pills(_positive_findings(future_state))}</div>
      {_report_html(future_text)}
    </section>
  </div>
  <section class="delta">
    <h3>Binary CheXpert report-label flips (target-side metadata)</h3>
    <ul>{change_items}</ul>
  </section>
</article>"""
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MIMIC-CXR current state to future state</title>
  <style>
    :root {{ color-scheme: light; --ink:#17202a; --muted:#667085; --line:#d9e0e8;
            --current:#e8f1ff; --future:#fff1dc; --accent:#9a4f00; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:#f4f6f8; color:var(--ink); font-family:Inter,system-ui,sans-serif; line-height:1.45; }}
    main {{ max-width:1440px; margin:0 auto; padding:28px; }}
    .intro,.transition {{ background:white; border:1px solid var(--line); border-radius:16px; box-shadow:0 6px 22px #24364a12; }}
    .intro {{ padding:24px; margin-bottom:24px; }}
    .warning {{ border-left:5px solid #d97706; background:#fff8e8; padding:12px 16px; }}
    .curation {{ border-left:4px solid #2563eb; background:#eff6ff; padding:9px 12px; margin-bottom:0; }}
    .files a {{ margin-right:16px; }}
    .transition {{ overflow:hidden; margin:24px 0; }}
    .transition>header,.delta {{ padding:18px 24px; }}
    .transition>header {{ border-bottom:1px solid var(--line); }}
    h1,h2,h3,h4,p {{ margin-top:0; }} h2 {{ overflow-wrap:anywhere; }}
    .states {{ display:grid; grid-template-columns:minmax(0,1fr) 76px minmax(0,1fr); }}
    .state {{ padding:24px; min-width:0; }} .current {{ background:var(--current); }} .future {{ background:var(--future); }}
    .source-badge {{ display:inline-block; margin-bottom:10px; padding:3px 8px; border-radius:999px; background:#dbeaf5; color:#234b68; font-size:12px; font-weight:700; letter-spacing:.03em; }}
    .state-title {{ display:flex; align-items:center; gap:10px; }}
    .state-title span {{ font:700 13px ui-monospace,monospace; border:1px solid currentColor; border-radius:20px; padding:4px 9px; }}
    .state-title h3 {{ margin:0; }} .meta,.muted {{ color:var(--muted); }}
    .state img {{ display:block; width:100%; height:min(58vw,650px); object-fit:contain; background:#080a0c; border-radius:10px; margin:15px 0; }}
    .arrow {{ display:flex; flex-direction:column; align-items:center; justify-content:center; font-size:42px; color:var(--accent); background:white; }}
    .arrow small {{ font-size:11px; text-align:center; color:var(--muted); }}
    .pills {{ margin:10px 0 20px; }} .pill {{ display:inline-block; background:#164e63; color:white; border-radius:18px; padding:4px 9px; margin:3px; font-size:12px; }}
    .pill.none {{ background:#667085; }} .state h4 {{ margin-bottom:3px; }} .state p {{ overflow-wrap:anywhere; }}
    .delta {{ border-top:1px solid var(--line); }} .change {{ color:#9a3412; font-weight:700; }}
    code {{ background:#edf0f3; border-radius:4px; padding:2px 5px; }}
    @media (max-width:850px) {{ .states {{ grid-template-columns:1fr; }} .arrow {{ padding:10px; transform:rotate(90deg); }} .state img {{ height:70vh; }} }}
  </style>
</head>
<body><main>
  <section class="intro">
    <h1>Current state &rarr; observed future state</h1>
    <p><strong>Database provenance:</strong> all radiographs, reports, study metadata, official splits, and CheXpert report labels on this page come from {DATASET_DISPLAY_NAME}. Each card is a chronologically ordered, same-patient, same-view pair. The displayed future is one retrospective observation, not the only possible future.</p>
    <p class="warning"><strong>Leakage boundary:</strong> the future report, future CheXpert state, and label flips are target-side supervision/evaluation only. They are absent from <code>forecast_inputs.jsonl</code> and never inserted into the forecasting prompt.</p>
    <p class="warning"><strong>Technique and sampling caveat:</strong> AP matching does not eliminate changes in posture, inspiration, rotation, magnification, or support devices. The time of the next clinically ordered CXR is also informative; this gallery is not a fixed-schedule prognosis cohort.</p>
    <p class="warning"><strong>Restricted data:</strong> these local artifacts remain governed by the MIMIC data-use agreement and must not be redistributed as unrestricted data.</p>
    <p class="files"><a href="transitions.jsonl">transition packets</a><a href="forecast_manifest.jsonl">trainer manifest</a><a href="forecast_inputs.jsonl">inference-safe inputs</a><a href="forecast_targets.jsonl">separate targets</a><a href="summary.json">audit summary</a></p>
  </section>
  {"".join(cards)}
</main></body></html>
"""


def validate_config(config: BuildConfig) -> None:
    if config.num_examples is not None and config.num_examples <= 0:
        raise ValueError("--num-examples must be positive")
    if config.num_examples is None and config.one_per_patient:
        raise ValueError("--all-matching requires multiple pairs per patient")
    for name, value in (
        ("--min-gap-hours", config.min_gap_hours),
        ("--max-gap-days", config.max_gap_days),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if config.min_gap_hours < 0:
        raise ValueError("--min-gap-hours must be non-negative")
    if config.max_gap_days <= 0:
        raise ValueError("--max-gap-days must be positive")
    if config.max_gap_days * 24 < config.min_gap_hours:
        raise ValueError("--max-gap-days must not be shorter than --min-gap-hours")
    if config.min_label_flips < 0:
        raise ValueError("--min-label-flips must be non-negative")
    if config.curated_pairs and config.min_label_flips != 0:
        raise ValueError("Curated pair selection requires min_label_flips=0")
    if config.max_report_chars <= 0:
        raise ValueError("--max-report-chars must be positive")
    if config.view not in {"AP", "PA", "frontal"}:
        raise ValueError("--view must be AP, PA, or frontal")
    if config.selection_strategy not in {"deterministic_random", "change_enriched"}:
        raise ValueError(
            "--selection-strategy must be deterministic_random or change_enriched"
        )
    if isinstance(config.seed, bool) or not isinstance(config.seed, int):
        raise TypeError("--seed must be an integer")
    if (
        config.one_per_patient
        and not config.curated_pairs
        and config.selection_strategy != "deterministic_random"
    ):
        raise ValueError(
            "one-per-patient automatic cohorts require deterministic_random selection"
        )
    if (
        config.one_per_patient
        and not config.curated_pairs
        and config.min_label_flips != 0
    ):
        raise ValueError(
            "one-per-patient automatic cohorts require --min-label-flips 0 to avoid target-conditioned inclusion"
        )
    if config.asset_mode not in {"symlink", "copy", "none"}:
        raise ValueError("--asset-mode must be symlink, copy, or none")


def build_dataset(config: BuildConfig) -> BuildResult:
    validate_config(config)
    root = resolve_mimic_cxr_root(config.mimic_cxr_root)
    output_dir = config.output_dir.expanduser().resolve()
    audit = Audit()
    studies = load_studies(root, config.view, audit)
    attach_splits(root, studies, audit)
    attach_labels(root, studies, audit)
    candidates = find_candidates(studies.values(), config, audit)
    curated_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    if config.curated_pairs:
        curated_by_key = {
            (
                entry["subject_id"],
                entry["source_study_id"],
                entry["target_study_id"],
            ): entry
            for entry in config.curated_pairs
        }
        candidate_by_key = {
            _candidate_key(candidate): candidate for candidate in candidates
        }
        missing = [key for key in curated_by_key if key not in candidate_by_key]
        if missing:
            raise RuntimeError(
                "Curated pairs were not valid under the requested split/view/time filters; "
                f"first missing pair: {missing[0]}"
            )
        candidates = [candidate_by_key[key] for key in curated_by_key]
        audit.increment("curated_pairs_matched", len(candidates))
    else:
        candidates = select_candidates(candidates, config, audit)

    packets: list[dict[str, object]] = []
    forecast_rows: list[dict[str, object]] = []
    input_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    selected_patients: set[str] = set()
    for candidate in candidates:
        if (
            config.curated_pairs
            and config.one_per_patient
            and candidate.source.subject_id in selected_patients
        ):
            audit.increment("curated_candidates_skipped_already_selected_patient")
            continue
        try:
            packet, forecast_row, input_row, target_row = make_packet(
                candidate, config.max_report_chars
            )
        except OSError:
            audit.increment("ranked_candidates_rejected_report_read_error")
            continue
        if not report_for_prompt(forecast_row["source_report"]):
            audit.increment("ranked_candidates_rejected_empty_current_text")
            continue
        annotation = curated_by_key.get(_candidate_key(candidate))
        if annotation:
            curation = {
                "category": annotation["category"],
                "audit_note": annotation["audit_note"],
                "audit_scope": "local example selection; not clinical adjudication",
            }
            packet["curation"] = curation
            for row in (forecast_row, input_row, target_row):
                row["selection_category"] = annotation["category"]
        packets.append(packet)
        forecast_rows.append(forecast_row)
        input_rows.append(input_row)
        target_rows.append(target_row)
        selected_patients.add(candidate.source.subject_id)
        if config.num_examples is not None and len(packets) >= config.num_examples:
            break

    if config.num_examples is not None and len(packets) < config.num_examples:
        raise RuntimeError(
            f"Requested {config.num_examples} examples but found {len(packets)} after filtering. "
            "Try --min-label-flips 0, a wider --max-gap-days, another --view, or --split all."
        )

    if config.curated_pairs:
        selection_policy: dict[str, object] = {
            "purpose": "small manually audited local demonstration; not clinical adjudication",
            "target_conditioned": True,
            "curated_pairs_path": str(config.curated_pairs_path.resolve())
            if config.curated_pairs_path
            else None,
            "categories": dict(
                sorted(Counter(row["selection_category"] for row in input_rows).items())
            ),
        }
    elif config.selection_strategy == "deterministic_random":
        selection_policy = {
            "purpose": "deterministic leakage-safe cohort sampling",
            "strategy": "deterministic_random",
            "seed": config.seed,
            "target_conditioned": config.min_label_flips > 0,
            "patient_order_key": "sha256(seed, patient_id)",
            "within_patient_pair_key": (
                "sha256(seed, patient/source/target/image identifiers)"
                if config.one_per_patient
                else None
            ),
            "uses_future_report_or_label_values_for_ordering": False,
        }
    else:
        selection_policy = {
            "purpose": "change-enriched local audit gallery, not a representative training cohort",
            "strategy": "change_enriched",
            "target_conditioned": True,
            "ranking": [
                "descending binary CheXpert label-flip count",
                "descending future positive-label count",
                "distance from a 24-hour observed follow-up",
            ],
        }

    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "source_dataset": DATASET_NAME,
        "mimic_cxr_root": str(root),
        "output_dir": str(output_dir),
        "generated_examples": len(packets),
        "filters": {
            "split": config.split,
            "view": config.view,
            "same_exact_view_required": True,
            "min_gap_hours": config.min_gap_hours,
            "max_gap_days": config.max_gap_days,
            "min_binary_chexpert_label_flips": config.min_label_flips,
            "one_per_patient": config.one_per_patient,
            "requested_examples": config.num_examples,
        },
        "selection_policy": selection_policy,
        "selected_horizon_bins": dict(
            sorted(
                Counter(
                    str(packet["interval"]["horizon_bin"]) for packet in packets
                ).items()
            )
        ),
        "gallery_rendered": config.render_gallery,
        "asset_mode": config.asset_mode if config.render_gallery else "none",
        "leakage_boundary": {
            "forecast_inputs_contains_future_fields": False,
            "prompt_sources": ["current report", "prespecified coarse horizon bin"],
            "exact_realized_interval_rendered_into_prompt": False,
            "future_report_usage": "target-side supervision/evaluation only; forbidden as model input",
        },
        "audit_counts": audit.as_dict(),
        "warnings": [
            "Dates are de-identified but within-patient intervals are preserved by MIMIC-CXR.",
            "CheXpert labels are weak labels derived from reports, not adjudicated image ground truth.",
            "The observed follow-up is one possible future and must not be interpreted causally.",
            "The next-CXR time is driven by clinical observation and ordering processes, not a fixed schedule.",
            "Exact AP/PA matching does not control posture, inspiration, rotation, magnification, or device changes.",
            "Generated artifacts remain subject to the MIMIC data-use agreement; do not publish them as unrestricted data.",
        ],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        write_jsonl(staging_dir / "transitions.jsonl", packets)
        write_jsonl(staging_dir / "forecast_manifest.jsonl", forecast_rows)
        write_jsonl(staging_dir / "forecast_inputs.jsonl", input_rows)
        write_jsonl(staging_dir / "forecast_targets.jsonl", target_rows)
        _atomic_write_text(
            staging_dir / "summary.json",
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        )
        if config.render_gallery:
            _atomic_write_text(
                staging_dir / "index.html",
                render_gallery(packets, staging_dir, config.asset_mode),
            )
        _atomic_write_text(
            staging_dir / OUTPUT_MARKER,
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "generator": str(Path(__file__).resolve()),
                },
                indent=2,
            )
            + "\n",
        )
        _validate_staged_output(
            staging_dir,
            len(packets),
            config.asset_mode,
            expect_gallery=config.render_gallery,
        )
        _commit_staged_output(staging_dir, output_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise

    output_paths = {
        "transitions": output_dir / "transitions.jsonl",
        "forecast_manifest": output_dir / "forecast_manifest.jsonl",
        "forecast_inputs": output_dir / "forecast_inputs.jsonl",
        "forecast_targets": output_dir / "forecast_targets.jsonl",
        "summary": output_dir / "summary.json",
    }
    if config.render_gallery:
        output_paths = {"gallery": output_dir / "index.html", **output_paths}
    return BuildResult(
        packets=packets,
        forecast_rows=forecast_rows,
        input_rows=input_rows,
        target_rows=target_rows,
        summary=summary,
        output_paths=output_paths,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build current-state -> future-state examples from local MIMIC-CXR-JPG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mimic-cxr-root",
        type=Path,
        default=Path("/home/data1/data/MIMIC/MIMIC_CXR"),
        help="MIMIC_CXR directory, or its parent MIMIC directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "example_output",
    )
    limit_group = parser.add_mutually_exclusive_group()
    limit_group.add_argument(
        "--num-examples",
        type=int,
        help="Maximum output rows; defaults to 5 unless --all-matching is used",
    )
    limit_group.add_argument(
        "--all-matching",
        action="store_true",
        help="Write every eligible adjacent pair (implies multiple pairs per patient)",
    )
    parser.add_argument(
        "--split", choices=("train", "validate", "test", "all"), default="train"
    )
    parser.add_argument(
        "--view",
        choices=("AP", "PA", "frontal"),
        default="AP",
        help="Use one projection; frontal accepts AP/PA studies but still requires exact pair-wise view matching",
    )
    parser.add_argument("--min-gap-hours", type=float, default=1.0)
    parser.add_argument("--max-gap-days", type=float, default=365.0)
    parser.add_argument(
        "--min-label-flips",
        "--min-explicit-changes",
        dest="min_label_flips",
        type=int,
        default=0,
        help="Minimum absent<->present binary CheXpert label flips per pair",
    )
    parser.add_argument(
        "--allow-multiple-per-patient",
        action="store_true",
        help="Allow more than one selected transition from the same patient",
    )
    parser.add_argument(
        "--selection-strategy",
        choices=("deterministic_random", "change_enriched"),
        default="deterministic_random",
        help="Leakage-safe stable sampling or target-conditioned audit-gallery ranking",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for process-independent deterministic cohort ordering",
    )
    parser.add_argument(
        "--asset-mode",
        choices=("symlink", "copy", "none"),
        default="symlink",
        help="How the HTML gallery accesses images; symlink avoids duplicating restricted data",
    )
    parser.add_argument(
        "--no-gallery",
        action="store_true",
        help="Skip index.html and image assets for machine-training manifests",
    )
    parser.add_argument("--max-report-chars", type=int, default=6000)
    parser.add_argument(
        "--curated-pairs",
        type=Path,
        help="Optional JSON pair list for a manually audited gallery; preserves file order",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.all_matching and args.curated_pairs:
        print(
            "error: --all-matching cannot be combined with --curated-pairs",
            file=sys.stderr,
        )
        return 2
    try:
        curated_pairs = (
            load_curated_pairs(args.curated_pairs) if args.curated_pairs else ()
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    config = BuildConfig(
        mimic_cxr_root=args.mimic_cxr_root,
        output_dir=args.output_dir,
        num_examples=(
            None
            if args.all_matching
            else args.num_examples
            if args.num_examples is not None
            else 5
        ),
        split=args.split,
        view=args.view,
        min_gap_hours=args.min_gap_hours,
        max_gap_days=args.max_gap_days,
        # Curated lists may intentionally contain stable/uncertain examples.
        min_label_flips=0 if curated_pairs else args.min_label_flips,
        one_per_patient=not (args.allow_multiple_per_patient or args.all_matching),
        selection_strategy=args.selection_strategy,
        seed=args.seed,
        asset_mode=args.asset_mode,
        render_gallery=not args.no_gallery,
        max_report_chars=args.max_report_chars,
        curated_pairs=curated_pairs,
        curated_pairs_path=args.curated_pairs,
    )
    try:
        result = build_dataset(config)
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Built {len(result.packets)} transition examples.")
    for name, path in result.output_paths.items():
        print(f"{name}: {path}")
    if config.render_gallery:
        print("Serve the gallery with:")
        print(
            f"  python -m http.server 8000 --directory {config.output_dir.expanduser().resolve()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
