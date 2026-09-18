#!/usr/bin/env python3
"""Attach retrospective MIMIC-IV context to MIMIC-CXR transitions.

MIMIC-CXR and MIMIC-IV share ``subject_id`` and aligned de-identified patient
timelines, but MIMIC-CXR does not expose ``hadm_id`` or ``stay_id``.  This
script therefore links a CXR transition in two steps:

1. exact ``subject_id`` equality;
2. containment of both CXR acquisition timestamps in one MIMIC-IV admission.

ICU stays, care units, diagnoses, admission procedures, and optional ICU input
events are then attached through the resolved ``hadm_id``/``stay_id``.  Events
after the current CXR are explicitly retrospective audit context, never fields
of the current-only forecasting input.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import os
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO

SCHEMA_VERSION = "mimic-cxr-mimic-iv-link-v1"
MIMIC_IV_DATASET_NAME = "MIMIC-IV-3.1"


def parse_time(value: object) -> datetime | None:
    text = str(value or "").strip()
    return datetime.fromisoformat(text) if text else None


def _csv_path(path: Path) -> Path:
    if path.suffix == ".gz" and not path.is_file():
        return path.with_suffix("")
    return path


def _open_csv(path: Path) -> TextIO:
    path = _csv_path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def resolve_mimic_iv_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    candidates = (root, root / "mimic-iv-3.1")
    for candidate in candidates:
        if all(
            _csv_path(candidate / name).is_file()
            for name in ("hosp/admissions.csv.gz", "icu/icustays.csv.gz")
        ):
            return candidate
    raise FileNotFoundError(
        f"Could not find mimic-iv-3.1 under {root}; expected hosp/ and icu/ tables"
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"No transitions found in {path}")
    return rows


def _subject_id(packet: Mapping[str, object]) -> str:
    patient_id = str(packet.get("patient_id") or "")
    subject_id = patient_id.removeprefix("p")
    if not subject_id.isdigit():
        raise ValueError(f"Invalid patient_id in transition: {patient_id!r}")
    return subject_id


def _state_time(packet: Mapping[str, object], field: str) -> datetime:
    state = packet.get(field)
    if not isinstance(state, Mapping):
        raise TypeError(f"Transition is missing object field {field!r}")
    timestamp = parse_time(state.get("timestamp"))
    if timestamp is None:
        raise ValueError(f"Transition {field!r} is missing a timestamp")
    return timestamp


def load_subject_rows(path: Path, subject_ids: set[str]) -> list[dict[str, str]]:
    """Stream one MIMIC table and retain only requested subjects."""

    rows: list[dict[str, str]] = []
    with _open_csv(path) as handle:
        reader = csv.reader(handle)
        fields = next(reader, [])
        if "subject_id" not in fields:
            raise ValueError(f"Table has no subject_id column: {path}")
        subject_column = fields.index("subject_id")
        for row in reader:
            if len(row) > subject_column and row[subject_column] in subject_ids:
                rows.append(dict(zip(fields, row)))
    return rows


def load_dictionary(
    path: Path, key_fields: tuple[str, ...]
) -> dict[tuple[str, ...], dict[str, str]]:
    output: dict[tuple[str, ...], dict[str, str]] = {}
    with _open_csv(path) as handle:
        for row in csv.DictReader(handle):
            key = tuple(str(row.get(field) or "") for field in key_fields)
            output[key] = dict(row)
    return output


def admission_start(row: Mapping[str, str]) -> datetime:
    values = [parse_time(row.get("edregtime")), parse_time(row.get("admittime"))]
    present = [value for value in values if value is not None]
    if not present:
        raise ValueError(f"Admission {row.get('hadm_id')} has no start time")
    return min(present)


def _contains(start: datetime | None, end: datetime | None, value: datetime) -> bool:
    return start is not None and end is not None and start <= value <= end


def find_common_admissions(
    source_time: datetime,
    target_time: datetime,
    rows: Iterable[Mapping[str, str]],
) -> list[Mapping[str, str]]:
    return [
        row
        for row in rows
        if _contains(
            admission_start(row), parse_time(row.get("dischtime")), source_time
        )
        and _contains(
            admission_start(row), parse_time(row.get("dischtime")), target_time
        )
    ]


def _matching_stays(
    timestamp: datetime, hadm_id: str, rows: Iterable[Mapping[str, str]]
) -> list[Mapping[str, str]]:
    return [
        row
        for row in rows
        if row.get("hadm_id") == hadm_id
        and _contains(
            parse_time(row.get("intime")), parse_time(row.get("outtime")), timestamp
        )
    ]


def _matching_transfers(
    timestamp: datetime, hadm_id: str, rows: Iterable[Mapping[str, str]]
) -> list[Mapping[str, str]]:
    matches: list[Mapping[str, str]] = []
    for row in rows:
        if row.get("hadm_id") != hadm_id:
            continue
        start = parse_time(row.get("intime"))
        end = parse_time(row.get("outtime"))
        if (
            start is not None
            and start <= timestamp
            and (end is None or timestamp < end)
        ):
            matches.append(row)
    return matches


def _safe_int(value: object, default: int = 10**9) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _admission_record(row: Mapping[str, str]) -> dict[str, object]:
    return {
        "hadm_id": row.get("hadm_id"),
        "admittime": row.get("admittime"),
        "dischtime": row.get("dischtime"),
        "edregtime": row.get("edregtime") or None,
        "edouttime": row.get("edouttime") or None,
        "admission_type": row.get("admission_type") or None,
        "admission_location": row.get("admission_location") or None,
        "discharge_location": row.get("discharge_location") or None,
        "hospital_expire_flag": row.get("hospital_expire_flag") or None,
    }


def _stay_record(row: Mapping[str, str]) -> dict[str, object]:
    return {
        "stay_id": row.get("stay_id"),
        "first_careunit": row.get("first_careunit") or None,
        "last_careunit": row.get("last_careunit") or None,
        "intime": row.get("intime"),
        "outtime": row.get("outtime"),
    }


def _location_record(
    timestamp: datetime,
    hadm_id: str,
    stays: Iterable[Mapping[str, str]],
    transfers: Iterable[Mapping[str, str]],
) -> dict[str, object]:
    stay_matches = _matching_stays(timestamp, hadm_id, stays)
    transfer_matches = _matching_transfers(timestamp, hadm_id, transfers)
    return {
        "timestamp": timestamp.isoformat(),
        "stay_match_status": (
            "unique"
            if len(stay_matches) == 1
            else "none"
            if not stay_matches
            else "ambiguous"
        ),
        "stay": _stay_record(stay_matches[0]) if len(stay_matches) == 1 else None,
        "careunit_match_status": (
            "unique"
            if len(transfer_matches) == 1
            else "none"
            if not transfer_matches
            else "ambiguous"
        ),
        "careunit": (
            transfer_matches[0].get("careunit") if len(transfer_matches) == 1 else None
        ),
    }


def _diagnosis_records(
    hadm_id: str,
    rows: Iterable[Mapping[str, str]],
    dictionary: Mapping[tuple[str, ...], Mapping[str, str]],
) -> list[dict[str, object]]:
    selected = sorted(
        (row for row in rows if row.get("hadm_id") == hadm_id),
        key=lambda row: _safe_int(row.get("seq_num")),
    )
    output = []
    for row in selected:
        key = (str(row.get("icd_code") or ""), str(row.get("icd_version") or ""))
        title = dictionary.get(key, {}).get("long_title")
        output.append(
            {
                "sequence": _safe_int(row.get("seq_num"), default=0),
                "icd_code": row.get("icd_code"),
                "icd_version": row.get("icd_version"),
                "title": title or None,
                "scope": "discharge-coded admission context; not event timing",
            }
        )
    return output


def _procedure_icd_records(
    hadm_id: str,
    rows: Iterable[Mapping[str, str]],
    dictionary: Mapping[tuple[str, ...], Mapping[str, str]],
) -> list[dict[str, object]]:
    selected = sorted(
        (row for row in rows if row.get("hadm_id") == hadm_id),
        key=lambda row: (
            str(row.get("chartdate") or ""),
            _safe_int(row.get("seq_num")),
        ),
    )
    output = []
    for row in selected:
        key = (str(row.get("icd_code") or ""), str(row.get("icd_version") or ""))
        title = dictionary.get(key, {}).get("long_title")
        output.append(
            {
                "chartdate": row.get("chartdate"),
                "sequence": _safe_int(row.get("seq_num"), default=0),
                "icd_code": row.get("icd_code"),
                "icd_version": row.get("icd_version"),
                "title": title or None,
                "time_precision": "date_only",
            }
        )
    return output


def _event_overlap(
    row: Mapping[str, str], source_time: datetime, target_time: datetime
) -> tuple[bool, str]:
    start = parse_time(row.get("starttime"))
    if start is None:
        return False, ""
    end = parse_time(row.get("endtime")) or start
    if start > target_time or end < source_time:
        return False, ""
    if start <= source_time and end >= target_time:
        role = "spans_source_to_target"
    elif start <= source_time:
        role = "ongoing_at_source"
    elif end > target_time:
        role = "starts_in_interval_continues_after_target"
    else:
        role = "within_observed_interval"
    return True, role


def _icu_event_records(
    hadm_id: str,
    source_time: datetime,
    target_time: datetime,
    rows: Iterable[Mapping[str, str]],
    items: Mapping[tuple[str, ...], Mapping[str, str]],
    *,
    kind: str,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        if row.get("hadm_id") != hadm_id:
            continue
        overlaps, temporal_role = _event_overlap(row, source_time, target_time)
        if not overlaps:
            continue
        item = items.get((str(row.get("itemid") or ""),), {})
        record: dict[str, object] = {
            "starttime": row.get("starttime"),
            "endtime": row.get("endtime"),
            "temporal_role": temporal_role,
            "stay_id": row.get("stay_id") or None,
            "itemid": row.get("itemid"),
            "label": item.get("label") or None,
            "category": item.get("category") or None,
            "status": row.get("statusdescription") or None,
        }
        if kind == "input":
            record.update(
                {
                    "amount": row.get("amount") or None,
                    "amount_unit": row.get("amountuom") or None,
                    "rate": row.get("rate") or None,
                    "rate_unit": row.get("rateuom") or None,
                }
            )
        else:
            record.update(
                {
                    "value": row.get("value") or None,
                    "value_unit": row.get("valueuom") or None,
                    "location": row.get("location") or None,
                }
            )
        output.append(record)
    return sorted(output, key=lambda row: (str(row["starttime"]), str(row["itemid"])))


def _nearest_recorded_cxr(
    timestamp: datetime,
    hadm_id: str,
    procedure_rows: Iterable[Mapping[str, str]],
    items: Mapping[tuple[str, ...], Mapping[str, str]],
) -> dict[str, object] | None:
    candidates: list[tuple[float, Mapping[str, str]]] = []
    for row in procedure_rows:
        if row.get("hadm_id") != hadm_id:
            continue
        item = items.get((str(row.get("itemid") or ""),), {})
        if item.get("label") != "Chest X-Ray":
            continue
        event_time = parse_time(row.get("starttime"))
        if event_time is not None:
            candidates.append((abs((event_time - timestamp).total_seconds()), row))
    if not candidates:
        return None
    difference, row = min(candidates, key=lambda item: item[0])
    within_one_hour = difference <= 3600
    return {
        "status": "concordant_within_one_hour"
        if within_one_hour
        else "no_match_within_one_hour",
        "procedureevents_starttime": row.get("starttime") if within_one_hour else None,
        "absolute_difference_minutes": round(difference / 60.0, 3),
        "interpretation": (
            "Independent timestamp concordance check; never used as a join key. "
            "The nearest difference is retained even when it exceeds one hour."
        ),
    }


def _event_counts(rows: Iterable[Mapping[str, object]]) -> dict[str, int]:
    counts = Counter(
        str(row.get("label") or f"itemid {row.get('itemid')}") for row in rows
    )
    return dict(sorted(counts.items()))


@dataclass
class IVTables:
    """Selected raw rows and dictionaries, held only in process memory."""

    by_subject: dict
    diagnosis_dictionary: dict
    procedure_dictionary: dict
    item_dictionary: dict
    include_icu_inputs: bool


def load_iv_tables(
    mimic_iv_root: Path,
    subject_ids: set[str],
    *,
    include_icu_inputs: bool,
    progress: Callable[[str], None] | None = None,
    index=None,
    dictionaries=None,
) -> IVTables:
    def read(path: Path) -> list[dict[str, str]]:
        if progress:
            progress(f"MIMIC-IV · {path.name}")
        if index is not None:
            name = path.parent.name + "." + path.name.split(".csv")[0]
            return [
                row
                for subject in subject_ids
                for row in index.read_subject(name, subject)
            ]
        return load_subject_rows(path, subject_ids)

    def dictionary(path, fields):
        key = (str(path), fields)
        if dictionaries is not None and key in dictionaries:
            return dictionaries[key]
        if index is None:
            result = load_dictionary(path, fields)
        else:
            name = path.parent.name + "." + path.name.split(".csv")[0]
            result = {
                tuple(row[f] for f in fields): row for row in index.iter_table(name)
            }
        if dictionaries is not None:
            dictionaries[key] = result
        return result

    hosp = mimic_iv_root / "hosp"
    icu = mimic_iv_root / "icu"

    admissions = read(hosp / "admissions.csv.gz")
    transfers = read(hosp / "transfers.csv.gz")
    diagnoses = read(hosp / "diagnoses_icd.csv.gz")
    procedures_icd = read(hosp / "procedures_icd.csv.gz")
    stays = read(icu / "icustays.csv.gz")
    procedureevents = read(icu / "procedureevents.csv.gz")
    inputevents = read(icu / "inputevents.csv.gz") if include_icu_inputs else []

    diagnosis_dictionary = dictionary(
        hosp / "d_icd_diagnoses.csv.gz", ("icd_code", "icd_version")
    )
    procedure_dictionary = dictionary(
        hosp / "d_icd_procedures.csv.gz", ("icd_code", "icd_version")
    )
    item_dictionary = dictionary(icu / "d_items.csv.gz", ("itemid",))

    by_subject: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for name, rows in (
        ("admissions", admissions),
        ("transfers", transfers),
        ("diagnoses", diagnoses),
        ("procedures_icd", procedures_icd),
        ("stays", stays),
        ("procedureevents", procedureevents),
        ("inputevents", inputevents),
    ):
        for row in rows:
            by_subject[row["subject_id"]][name].append(row)

    return IVTables(
        by_subject,
        diagnosis_dictionary,
        procedure_dictionary,
        item_dictionary,
        include_icu_inputs,
    )


def link_packets(
    packets: Sequence[dict[str, object]],
    mimic_iv_root: Path,
    *,
    include_icu_inputs: bool,
    tables: IVTables | None = None,
) -> list[dict[str, object]]:
    """Link packets using either fresh table scans or caller-owned memory."""
    if tables is None:
        tables = load_iv_tables(
            mimic_iv_root,
            {_subject_id(p) for p in packets},
            include_icu_inputs=include_icu_inputs,
        )
    if include_icu_inputs != tables.include_icu_inputs:
        raise ValueError("ICU input selection differs from the loaded tables")
    by_subject = tables.by_subject
    diagnosis_dictionary = tables.diagnosis_dictionary
    procedure_dictionary = tables.procedure_dictionary
    item_dictionary = tables.item_dictionary
    linked: list[dict[str, object]] = []
    for packet in packets:
        subject_id = _subject_id(packet)
        source_time = _state_time(packet, "current_state")
        target_time = _state_time(packet, "future_state")
        subject = by_subject[subject_id]
        matches = find_common_admissions(
            source_time, target_time, subject["admissions"]
        )
        base: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "transition_id": packet.get("transition_id"),
            "subject_id": subject_id,
            "source_databases": {
                "mimic_cxr_transition": packet.get("source_dataset"),
                "mimic_iv_linkage_and_context": MIMIC_IV_DATASET_NAME,
            },
            "linkage_method": {
                "patient": "exact subject_id equality",
                "admission": (
                    "both CXR acquisition timestamps contained in the same interval "
                    "[min(edregtime, admittime), dischtime]"
                ),
                "icu": "CXR timestamp contained in [icustays.intime, icustays.outtime]",
                "not_join_keys": ["study_id", "dicom_id"],
            },
            "mimic_cxr_transition": packet,
        }
        if len(matches) != 1:
            base.update(
                {
                    "linkage_status": "unmatched" if not matches else "ambiguous",
                    "candidate_hadm_ids": [row.get("hadm_id") for row in matches],
                    "retrospective_context_for_audit_only": None,
                }
            )
            linked.append(base)
            continue

        admission = matches[0]
        hadm_id = str(admission["hadm_id"])
        interval_procedures = _icu_event_records(
            hadm_id,
            source_time,
            target_time,
            subject["procedureevents"],
            item_dictionary,
            kind="procedure",
        )
        interval_inputs = _icu_event_records(
            hadm_id,
            source_time,
            target_time,
            subject["inputevents"],
            item_dictionary,
            kind="input",
        )
        base.update(
            {
                "linkage_status": "unique_common_admission",
                "candidate_hadm_ids": [hadm_id],
                "retrospective_context_for_audit_only": {
                    "availability": (
                        "observed records; rows after current_state.timestamp are forbidden "
                        "as current-only forecast input"
                    ),
                    "causal_status": (
                        "observational co-timing only; no treatment effect is identified"
                    ),
                    "admission": _admission_record(admission),
                    "source_location": _location_record(
                        source_time, hadm_id, subject["stays"], subject["transfers"]
                    ),
                    "target_location": _location_record(
                        target_time, hadm_id, subject["stays"], subject["transfers"]
                    ),
                    "admission_diagnoses": _diagnosis_records(
                        hadm_id, subject["diagnoses"], diagnosis_dictionary
                    ),
                    "admission_procedures": _procedure_icd_records(
                        hadm_id, subject["procedures_icd"], procedure_dictionary
                    ),
                    "interval_icu_procedureevents": interval_procedures,
                    "interval_icu_inputevents": (
                        interval_inputs if include_icu_inputs else None
                    ),
                    "interval_event_summary": {
                        "procedureevent_counts": _event_counts(interval_procedures),
                        "inputevent_counts": (
                            _event_counts(interval_inputs)
                            if include_icu_inputs
                            else None
                        ),
                    },
                    "timestamp_concordance_checks": {
                        "source_cxr": _nearest_recorded_cxr(
                            source_time,
                            hadm_id,
                            subject["procedureevents"],
                            item_dictionary,
                        ),
                        "target_cxr": _nearest_recorded_cxr(
                            target_time,
                            hadm_id,
                            subject["procedureevents"],
                            item_dictionary,
                        ),
                    },
                },
            }
        )
        linked.append(base)
    return linked


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    content = "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
    )
    _atomic_write_text(path, content)


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _study_id(row: Mapping[str, object], state_name: str) -> str:
    packet = _mapping(row.get("mimic_cxr_transition"))
    state = _mapping(packet.get(state_name))
    return str(state.get("study_id") or "")


def _csv_rows(linked: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for index, row in enumerate(linked, start=1):
        packet = _mapping(row.get("mimic_cxr_transition"))
        sources = _mapping(row.get("source_databases"))
        interval = _mapping(packet.get("interval"))
        curation = _mapping(packet.get("curation"))
        context = _mapping(row.get("retrospective_context_for_audit_only"))
        admission = _mapping(context.get("admission"))
        source_location = _mapping(context.get("source_location"))
        target_location = _mapping(context.get("target_location"))
        source_stay = _mapping(source_location.get("stay"))
        target_stay = _mapping(target_location.get("stay"))
        diagnoses = context.get("admission_diagnoses")
        diagnosis_titles = (
            [
                str(item.get("title") or item.get("icd_code"))
                for item in diagnoses
                if isinstance(item, Mapping)
            ]
            if isinstance(diagnoses, list)
            else []
        )
        summary = _mapping(context.get("interval_event_summary"))
        procedure_counts = _mapping(summary.get("procedureevent_counts"))
        input_counts = _mapping(summary.get("inputevent_counts"))
        output.append(
            {
                "case": chr(64 + index) if index <= 26 else str(index),
                "transition_id": row.get("transition_id"),
                "subject_id": row.get("subject_id"),
                "cxr_source_database": sources.get("mimic_cxr_transition"),
                "linkage_context_source_database": sources.get(
                    "mimic_iv_linkage_and_context"
                ),
                "category": curation.get("category"),
                "source_study_id": _study_id(row, "current_state"),
                "target_study_id": _study_id(row, "future_state"),
                "elapsed_hours": interval.get("elapsed_hours"),
                "horizon_bin": interval.get("horizon_bin"),
                "linkage_status": row.get("linkage_status"),
                "hadm_id": admission.get("hadm_id"),
                "source_stay_id": source_stay.get("stay_id"),
                "target_stay_id": target_stay.get("stay_id"),
                "source_careunit": source_location.get("careunit"),
                "target_careunit": target_location.get("careunit"),
                "diagnoses": " | ".join(diagnosis_titles[:8]),
                "interval_icu_procedures": " | ".join(
                    f"{name} ({count})" for name, count in procedure_counts.items()
                ),
                "interval_icu_inputs": " | ".join(
                    f"{name} ({count})" for name, count in input_counts.items()
                ),
            }
        )
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _report_excerpt(packet: Mapping[str, object], state_name: str) -> str:
    state = _mapping(packet.get(state_name))
    text_field = "text" if state_name == "current_state" else "text_for_evaluation_only"
    report = _mapping(state.get(text_field))
    value = (
        report.get("impression")
        or report.get("findings")
        or report.get("unsectioned_report")
    )
    clean = " ".join(str(value or "Report unavailable").split())
    return clean if len(clean) <= 650 else clean[:647].rstrip() + "..."


def _asset_path(
    row: Mapping[str, object], state_name: str, transitions_path: Path, output_dir: Path
) -> str:
    suffix = "current" if state_name == "current_state" else "future"
    candidate = (
        transitions_path.parent / "assets" / f"{row.get('transition_id')}_{suffix}.jpg"
    )
    if candidate.is_file():
        return Path(os.path.relpath(candidate, output_dir)).as_posix()
    packet = _mapping(row.get("mimic_cxr_transition"))
    state = _mapping(packet.get(state_name))
    image = _mapping(state.get("image"))
    path = Path(str(image.get("path") or ""))
    return path.as_uri() if path.is_file() else ""


def _list_html(values: Sequence[str]) -> str:
    if not values:
        return '<p class="muted">No rows in the selected source tables.</p>'
    return (
        "<ul>" + "".join(f"<li>{html.escape(value)}</li>" for value in values) + "</ul>"
    )


def _display_event_names(context: Mapping[str, object]) -> list[str]:
    procedures = context.get("interval_icu_procedureevents")
    inputs = context.get("interval_icu_inputevents")
    procedure_counter: Counter[str] = Counter()
    input_counter: Counter[str] = Counter()
    if isinstance(procedures, list):
        for row in procedures:
            if not isinstance(row, Mapping):
                continue
            category = str(row.get("category") or "")
            label = str(row.get("label") or f"itemid {row.get('itemid')}")
            if category.startswith("Access Lines") or category == "7-Communication":
                continue
            if label != "Chest X-Ray":
                procedure_counter[label] += 1
    generic_inputs = {
        "Dextrose 5%",
        "Gastric Meds",
        "GT Flush",
        "NaCl 0.9%",
        "Piggyback",
        "PO Intake",
        "Solution",
    }
    if isinstance(inputs, list):
        for row in inputs:
            if not isinstance(row, Mapping):
                continue
            label = str(row.get("label") or f"itemid {row.get('itemid')}")
            if label not in generic_inputs:
                input_counter[label] += 1
    ranked_procedures = sorted(
        procedure_counter.items(), key=lambda item: (-item[1], item[0])
    )
    ranked_inputs = sorted(input_counter.items(), key=lambda item: (-item[1], item[0]))
    return [
        *(f"{name} × {count}" for name, count in ranked_procedures),
        *(f"{name} × {count}" for name, count in ranked_inputs),
    ]


def render_html(
    linked: Sequence[Mapping[str, object]], transitions_path: Path, output_dir: Path
) -> str:
    cards: list[str] = []
    for index, row in enumerate(linked, start=1):
        packet = _mapping(row.get("mimic_cxr_transition"))
        interval = _mapping(packet.get("interval"))
        context = _mapping(row.get("retrospective_context_for_audit_only"))
        admission = _mapping(context.get("admission"))
        source_location = _mapping(context.get("source_location"))
        target_location = _mapping(context.get("target_location"))
        diagnoses = context.get("admission_diagnoses")
        diagnosis_names = (
            [
                str(item.get("title") or item.get("icd_code"))
                for item in diagnoses
                if isinstance(item, Mapping)
            ][:5]
            if isinstance(diagnoses, list)
            else []
        )
        events = _display_event_names(context)
        source_image = _asset_path(row, "current_state", transitions_path, output_dir)
        target_image = _asset_path(row, "future_state", transitions_path, output_dir)
        linkage_status = str(row.get("linkage_status") or "unknown")
        linkage_label = {
            "unique_common_admission": "Matched common admission",
            "unmatched": "No common admission match",
            "ambiguous": "Ambiguous admission match",
        }.get(linkage_status, linkage_status.replace("_", " ").title())
        linkage_class = (
            linkage_status
            if linkage_status in {"unique_common_admission", "unmatched", "ambiguous"}
            else "other"
        )
        cards.append(
            f"""
            <section class="case">
              <h2>Case {chr(64 + index)} · {html.escape(str(_mapping(packet.get("curation")).get("category") or "example"))}</h2>
              <p class="ids">subject_id {html.escape(str(row.get("subject_id")))} → hadm_id {html.escape(str(admission.get("hadm_id") or "unmatched"))} · {html.escape(str(interval.get("elapsed_hours")))} h · {html.escape(str(interval.get("horizon_bin") or ""))}</p>
              <div class="grid">
                <article class="state input">
                  <span class="source-badge source-cxr">DATABASE · MIMIC-CXR-JPG v2.0.0</span>
                  <h3>CURRENT CXR · MODEL INPUT</h3>
                  <img src="{html.escape(source_image)}" alt="Current de-identified chest radiograph">
                  <p>{html.escape(_report_excerpt(packet, "current_state"))}</p>
                </article>
                <article class="context">
                  <span class="source-badge source-iv">DATABASE · MIMIC-IV v3.1</span>
                  <span class="link-status {html.escape(linkage_class)}">{html.escape(linkage_label)}</span>
                  <h3>RETROSPECTIVE CONTEXT · AUDIT ONLY</h3>
                  <p><strong>Location:</strong> {html.escape(str(source_location.get("careunit") or "not in an identified unit"))} → {html.escape(str(target_location.get("careunit") or "not in an identified unit"))}</p>
                  <h4>Admission diagnoses</h4>
                  {_list_html(diagnosis_names)}
                  <h4>Observed interval ICU records</h4>
                  {_list_html(events[:10])}
                </article>
                <article class="state target">
                  <span class="source-badge source-cxr">DATABASE · MIMIC-CXR-JPG v2.0.0</span>
                  <h3>FOLLOW-UP CXR · TARGET ONLY</h3>
                  <img src="{html.escape(target_image)}" alt="Observed follow-up de-identified chest radiograph">
                  <p>{html.escape(_report_excerpt(packet, "future_state"))}</p>
                </article>
              </div>
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Linked MIMIC-CXR and MIMIC-IV examples</title>
<style>
body{{margin:0;background:#fafaf7;color:#24313a;font:15px/1.45 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:28px}}h1{{margin-bottom:4px}}.lead{{max-width:1000px;color:#52616b}}.case{{margin:28px 0;padding:22px;background:white;border:1px solid #d8dedf;border-radius:12px}}h2{{margin:0}}.ids{{color:#667680}}.grid{{display:grid;grid-template-columns:minmax(250px,1fr) minmax(320px,1.25fr) minmax(250px,1fr);gap:18px}}article{{padding:14px;border-radius:9px}}.input{{background:#eef3f7}}.context{{background:#f2f1ed;border:1px dashed #9da6a8}}.target{{background:#edf4ee}}.source-badge{{display:inline-block;margin-bottom:8px;padding:3px 8px;border-radius:999px;font-size:12px;font-weight:700;letter-spacing:.03em}}.source-cxr{{background:#dbeaf5;color:#234b68}}.source-iv{{background:#eadfca;color:#5a4520}}.link-status{{display:block;margin-bottom:6px;font-size:12px;font-weight:650}}.link-status.unmatched,.link-status.ambiguous{{color:#8a4b16}}img{{display:block;width:100%;height:420px;object-fit:contain;background:#101417}}h3{{font-size:14px;letter-spacing:.04em}}h4{{margin-bottom:5px}}ul{{margin-top:5px;padding-left:20px}}.muted{{color:#78858b}}footer{{border-top:1px solid #ccd3d4;margin-top:30px;padding-top:18px;color:#5c696f}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}img{{height:auto}}}}
</style></head><body><main>
<h1>Linked MIMIC-CXR ↔ MIMIC-IV examples</h1>
<p class="lead"><strong>Database provenance:</strong> the left and right columns come from MIMIC-CXR-JPG v2.0.0; the middle column is queried from MIMIC-IV v3.1. Exact subject_id plus aligned timestamp containment resolves hadm_id/stay_id. The middle column is retrospective observational context for audit and figure design; records after the current CXR are not current-only forecast inputs.</p>
{"".join(cards)}
<footer>Illustrative, manually selected examples only. Future reports and all post-current MIMIC-IV events are evaluation/audit-side information. Co-timing does not identify a causal treatment effect. Restricted MIMIC data must remain in an approved environment.</footer>
</main></body></html>"""


def build_outputs(
    *,
    mimic_root: Path,
    transitions_path: Path,
    output_dir: Path,
    include_icu_inputs: bool,
    render_gallery: bool = True,
) -> list[dict[str, object]]:
    iv_root = resolve_mimic_iv_root(mimic_root)
    transitions_path = transitions_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    packets = read_jsonl(transitions_path)
    linked = link_packets(packets, iv_root, include_icu_inputs=include_icu_inputs)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "linked_transitions.jsonl", linked)
    write_csv(output_dir / "appendix_cases.csv", _csv_rows(linked))
    status_counts = Counter(str(row["linkage_status"]) for row in linked)
    cxr_sources = sorted(
        {
            str(packet["source_dataset"])
            for packet in packets
            if packet.get("source_dataset")
        }
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "source_databases": {
            "mimic_cxr_transitions": cxr_sources,
            "mimic_iv_linkage_and_context": MIMIC_IV_DATASET_NAME,
        },
        "mimic_iv_root": str(iv_root),
        "source_transitions": str(transitions_path),
        "generated_examples": len(linked),
        "linkage_status_counts": dict(sorted(status_counts.items())),
        "icu_inputevents_included": include_icu_inputs,
        "outputs": [
            "linked_transitions.jsonl",
            "appendix_cases.csv",
            *(["index.html"] if render_gallery else []),
        ],
        "warnings": [
            "MIMIC-CXR has no direct hadm_id/stay_id; study_id is not an admission key.",
            "All post-current clinical events are retrospective audit context, not forecasting inputs.",
            "Diagnosis codes are admission-level discharge coding and do not timestamp onset.",
            "ICD procedure chartdate is date-granular and must not be ordered within a day.",
            "Observed treatment/event co-timing does not establish a causal effect.",
            "Generated artifacts remain subject to the MIMIC data-use agreement.",
        ],
    }
    _atomic_write_text(
        output_dir / "summary.json",
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )
    if render_gallery:
        _atomic_write_text(
            output_dir / "index.html",
            render_html(linked, transitions_path, output_dir),
        )
    return linked


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mimic-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data/MIMIC",
        help="MIMIC parent directory or mimic-iv-3.1 directory",
    )
    parser.add_argument(
        "--transitions",
        type=Path,
        default=Path(__file__).resolve().parent
        / "runs"
        / "exports"
        / ("generated_" + datetime.now().astimezone().strftime("%Y%m%d"))
        / "cxr"
        / "transitions.jsonl",
        help="CXR transition packets produced by build_mimic_transitions.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "runs"
        / "exports"
        / ("generated_" + datetime.now().astimezone().strftime("%Y%m%d"))
        / "linked",
    )
    parser.add_argument(
        "--include-icu-inputs",
        action="store_true",
        help="Also scan icu/inputevents.csv.gz (slower and more verbose)",
    )
    parser.add_argument("--no-gallery", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    linked = build_outputs(
        mimic_root=args.mimic_root,
        transitions_path=args.transitions,
        output_dir=args.output_dir,
        include_icu_inputs=args.include_icu_inputs,
        render_gallery=not args.no_gallery,
    )
    counts = Counter(str(row["linkage_status"]) for row in linked)
    print(
        f"Wrote {len(linked)} linked examples to {args.output_dir} "
        f"({dict(sorted(counts.items()))})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
