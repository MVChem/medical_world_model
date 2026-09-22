"""Temporal sampling over complete patient timelines without copying assets.

Adjacency always refers to the complete timeline, including unusable studies.
Random additions are nonadjacent chronological combinations. Their bounded,
seeded selection uses identifiers alone, never future report/label values.
"""

from __future__ import annotations

import hashlib
import heapq
import math
from collections import Counter
from collections.abc import Collection, Iterable, Mapping
from datetime import datetime, timedelta
from typing import Any

from ..build_mimic_transitions import Study
from ..link_mimic_iv_context import admission_start, parse_time

PAIRING_MODES = ("adjacent", "random", "adjacent_random", "all")
LINKAGE_MODES = ("same_admission", "patient")


def _sampling_key(row: Mapping[str, Any], seed: int) -> tuple[int, tuple[str, ...]]:
    identity = (
        row["source_study"].subject_id,
        row["source_study"].study_id,
        row["target_study"].study_id,
        row["source_image"].dicom_id,
        row["target_image"].dicom_id,
        row["matched_view"],
    )
    payload = "\0".join((str(seed), "medworld-nonadjacent-v1", *identity))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest(), "big"), identity


def select_patient_pairs(
    studies: Iterable[Study],
    admissions: Iterable[Mapping[str, str]],
    *,
    mode: str = "adjacent_random",
    seed: int = 42,
    random_pairs_per_patient: int = 8,
    min_gap_hours: float = 1.0,
    max_gap_days: float = 365.0,
    linkage: str = "same_admission",
    eligible_study_ids: Collection[str] | None = None,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Return selected pairs and aggregate rejection/selection counts.

    The caller supplies exactly one patient's *complete* timeline and validates
    report/image availability, optionally marking usable studies with
    ``eligible_study_ids``. The existing loader selects the largest image per
    view (DICOM ID breaks area ties); PA is preferred when both endpoints share
    PA and AP. Study ordering uses the earliest acquisition, whereas intervals
    and admission matches use the selected images' actual acquisition times.

    ``same_admission`` requires each endpoint to match exactly one admission,
    and those admissions must be equal. This conservatively tightens Atlas's
    unique-common-admission rule by rejecting an independently ambiguous
    endpoint. ``patient`` permits cross-admission pairs; the caller must have
    confirmed membership in MIMIC-IV (which can include no-admission patients).

    ``adjacent`` retains all eligible full-timeline neighbors. ``random`` keeps
    at most ``random_pairs_per_patient`` nonadjacent pairs. ``adjacent_random``
    combines those policies, and ``all`` retains every eligible combination.
    The random heap retains O(k) pairs, with O(n) study/admission metadata.
    """
    if mode not in PAIRING_MODES:
        raise ValueError(f"mode must be one of {PAIRING_MODES}")
    if linkage not in LINKAGE_MODES:
        raise ValueError(f"linkage must be one of {LINKAGE_MODES}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if (
        isinstance(random_pairs_per_patient, bool)
        or not isinstance(random_pairs_per_patient, int)
        or random_pairs_per_patient < 0
    ):
        raise ValueError("random_pairs_per_patient must be a nonnegative integer")
    if (
        isinstance(min_gap_hours, bool)
        or isinstance(max_gap_days, bool)
        or not math.isfinite(min_gap_hours)
        or not math.isfinite(max_gap_days)
        or min_gap_hours < 0
        or max_gap_days <= 0
        or min_gap_hours > max_gap_days * 24
    ):
        raise ValueError("Require finite 0 <= min_gap_hours <= max_gap_days * 24")

    timeline = sorted(studies, key=lambda study: (study.timestamp, study.study_id))
    audit: Counter[str] = Counter(timeline_studies=len(timeline))
    if not timeline:
        return [], audit
    subject_id = timeline[0].subject_id
    if any(study.subject_id != subject_id for study in timeline):
        raise ValueError("Pairing requires exactly one patient's complete timeline")
    if len({study.study_id for study in timeline}) != len(timeline):
        raise ValueError("Duplicate study identifier in patient timeline")
    patient_splits = {study.split for study in timeline if study.split is not None}
    if len(patient_splits) > 1:
        raise ValueError("Patient-level split violation in patient timeline")
    timestamp_counts = Counter(study.timestamp for study in timeline)
    allowed = set(eligible_study_ids) if eligible_study_ids is not None else None

    spans: list[tuple[datetime, datetime, str]] = []
    for admission in admissions:
        if str(admission.get("subject_id", "")) != subject_id:
            audit["admissions_rejected_other_subject"] += 1
            continue
        try:
            start = admission_start(admission)
            end = parse_time(admission.get("dischtime"))
        except (ValueError, TypeError):
            audit["admissions_rejected_invalid_interval"] += 1
            continue
        hadm_id = str(admission.get("hadm_id") or "")
        if end is None or end < start or not hadm_id:
            audit["admissions_rejected_invalid_interval"] += 1
            continue
        spans.append((start, end, hadm_id))

    endpoint_matches: dict[datetime, tuple[str, ...]] = {}

    def matches(timestamp: datetime) -> tuple[str, ...]:
        if timestamp not in endpoint_matches:
            endpoint_matches[timestamp] = tuple(
                hadm_id for start, end, hadm_id in spans if start <= timestamp <= end
            )
        return endpoint_matches[timestamp]

    def candidate(source_order: int, target_order: int) -> dict[str, Any] | None:
        source, target = timeline[source_order], timeline[target_order]
        audit["pairs_considered"] += 1
        if allowed is not None and (
            source.study_id not in allowed or target.study_id not in allowed
        ):
            audit["pairs_rejected_unavailable_study"] += 1
            return None
        if timestamp_counts[source.timestamp] > 1 or timestamp_counts[target.timestamp] > 1:
            audit["pairs_rejected_ambiguous_tied_timestamp"] += 1
            return None
        if source.split not in {"train", "validate", "test"} or source.split != target.split:
            audit["pairs_rejected_split"] += 1
            return None
        if source.labels is None or target.labels is None:
            audit["pairs_rejected_missing_labels"] += 1
            return None
        if source.latest_image_timestamp >= target.timestamp:
            audit["pairs_rejected_overlapping_acquisition_windows"] += 1
            return None
        view = next(
            (view for view in ("PA", "AP") if view in source.images_by_view and view in target.images_by_view),
            None,
        )
        if view is None:
            audit["pairs_rejected_view_mismatch"] += 1
            return None
        source_image, target_image = source.images_by_view[view], target.images_by_view[view]
        hours = (target_image.timestamp - source_image.timestamp).total_seconds() / 3600
        if hours <= 0 or hours < min_gap_hours or hours > max_gap_days * 24:
            audit["pairs_rejected_gap"] += 1
            return None
        source_matches, target_matches = matches(source_image.timestamp), matches(target_image.timestamp)
        hadm_id = (
            source_matches[0]
            if len(source_matches) == len(target_matches) == 1 and source_matches == target_matches
            else None
        )
        if linkage == "same_admission" and hadm_id is None:
            reason = (
                "ambiguous_admission"
                if len(source_matches) > 1 or len(target_matches) > 1
                else "no_same_admission"
            )
            audit[f"pairs_rejected_{reason}"] += 1
            return None
        kind = "adjacent" if target_order == source_order + 1 else "nonadjacent"
        audit[f"eligible_{kind}_pairs"] += 1
        return {
            "source_study": source,
            "target_study": target,
            "source_image": source_image,
            "target_image": target_image,
            "matched_view": view,
            "source_order": source_order,
            "target_order": target_order,
            "realized_gap_hours": hours,
            "hadm_id": hadm_id,
            "kind": kind,
        }

    selected: list[dict[str, Any]] = []
    if mode in {"adjacent", "adjacent_random", "all"}:
        for source_order in range(len(timeline) - 1):
            row = candidate(source_order, source_order + 1)
            if row is not None:
                selected.append(row)

    heap: list[tuple[int, tuple[str, ...], dict[str, Any]]] = []
    if mode == "all" or mode in {"random", "adjacent_random"} and random_pairs_per_patient:
        for source_order, source in enumerate(timeline):
            latest_target = source.latest_image_timestamp + timedelta(days=max_gap_days)
            for target_order in range(source_order + 2, len(timeline)):
                if timeline[target_order].timestamp > latest_target:
                    break
                row = candidate(source_order, target_order)
                if row is None:
                    continue
                if mode == "all":
                    selected.append(row)
                    continue
                rank, identity = _sampling_key(row, seed)
                item = (-rank, identity, row)
                if len(heap) < random_pairs_per_patient:
                    heapq.heappush(heap, item)
                elif rank < -heap[0][0]:
                    heapq.heapreplace(heap, item)
    selected.extend(item[2] for item in heap)
    selected.sort(key=lambda row: (row["source_order"], row["target_order"], row["matched_view"]))
    audit["selected_pairs"] = len(selected)
    audit["selected_adjacent_pairs"] = sum(row["kind"] == "adjacent" for row in selected)
    audit["selected_nonadjacent_pairs"] = len(selected) - audit["selected_adjacent_pairs"]
    return selected, audit


def iter_patient_pairs(studies, admissions, **kwargs):
    """Yield selected rows; use :func:`select_patient_pairs` for audit counts."""
    yield from select_patient_pairs(studies, admissions, **kwargs)[0]
