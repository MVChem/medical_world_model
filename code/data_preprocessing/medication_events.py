"""Pure predicates for recorded administrations used by CXR cohort filtering.

No prescription or pharmacy order implies administration. eMAR requires an
explicit administration status; ICU requires a delivered, positive medication
segment and the original d_items category. Amounts retain their source units:
neither products nor overlapping eMAR/ICU records are summed or deduplicated.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
import hashlib
import json
import math
from typing import Any


EMAR_ADMINISTRATION_STATUSES = frozenset({"administered", "applied", "started", "restarted", "given"})
ICU_DELIVERED_STATUSES = frozenset({
    "finishedrunning", "changeddose/rate", "changedose/rate", "changed", "stopped", "paused",
})
ICU_MEDICATION_CATEGORIES = frozenset({"medications", "antibiotics"})
DOSE_FIELDS = (
    "parent_field_ordinal", "dose_given", "dose_given_unit", "dose_due", "dose_due_unit",
    "product_amount_given", "product_unit", "product_description", "route", "administration_type",
    "infusion_rate", "infusion_rate_unit", "complete_dose_not_given", "infusion_complete",
)


def normalize_status(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def parse_timestamp(value: Any) -> datetime | None:
    """Accept MIMIC's naive full timestamps; reject date-only/timezone values."""
    if isinstance(value, datetime):
        return value if value.tzinfo is None else None
    text = str(value or "").strip()
    if len(text) < 19:
        return None
    try:
        result = datetime.fromisoformat(text)
    except ValueError:
        return None
    return result if result.tzinfo is None else None


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _identifier(value: Any) -> str:
    return str(value or "").strip()


def _record_id(table: str, row: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(dict(row), sort_keys=True, default=str).encode()).hexdigest()[:20]
    natural = row.get("emar_id") or row.get("orderid") or "row"
    return f"{table}:{natural}:{digest}"


def _base(row: Mapping[str, Any], table: str, name: str, start: datetime, end: datetime) -> dict[str, Any]:
    return {
        "subject_id": _identifier(row.get("subject_id")), "hadm_id": _identifier(row.get("hadm_id")),
        "source_table": table, "record_id": _record_id(table, row), "name": name,
        "start": start, "end": end, "amount": None, "amount_unit": None,
        "rate": None, "rate_unit": None, "route": None,
    }


def emar_event(
    row: Mapping[str, Any], details: Iterable[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Normalize an actual eMAR event, or return a machine-readable rejection.

    Details are optional. Only exact subject_id + emar_id children are used.
    An explicit administered status is sufficient when dose is unquantified;
    reject when every present dose_given is finite and nonpositive. Planned
    dose_due never substitutes for dose_given. Product rows remain separate.
    """
    if not _identifier(row.get("subject_id")):
        return None, "missing_subject"
    status = normalize_status(row.get("event_txt"))
    if status not in EMAR_ADMINISTRATION_STATUSES:
        return None, "not_explicit_administration:" + status
    name = str(row.get("medication") or "").strip()
    if not name:
        return None, "missing_medication_name"
    timestamp = parse_timestamp(row.get("charttime"))
    if timestamp is None:
        return None, "missing_or_invalid_event_time"
    event_id = _identifier(row.get("emar_id"))
    children = [dict(detail) for detail in details or ()
                if event_id and _identifier(detail.get("subject_id")) == _identifier(row.get("subject_id"))
                and _identifier(detail.get("emar_id")) == event_id]
    given = [detail for detail in children
             if detail.get("dose_given") is not None and str(detail["dose_given"]).strip()]
    values = [_number(detail["dose_given"]) for detail in given]
    if values and all(value is not None and value <= 0 for value in values):
        return None, "explicit_nonpositive_dose"
    event = _base(row, "hosp.emar", name, timestamp, timestamp)
    event.update({
        "record_status": row.get("event_txt"), "event_type": "administration",
        "emar_id": event_id or None, "pharmacy_id": row.get("pharmacy_id") or None,
        "dose_details": [{key: detail.get(key) for key in DOSE_FIELDS} for detail in children],
        "dose_basis": "explicit eMAR administration status; dose unquantified",
    })
    if len(given) == 1:
        event.update(amount=given[0]["dose_given"], amount_unit=given[0].get("dose_given_unit") or None,
                     dose_basis="single original dose_given value; no summation")
    routes = {str(detail["route"]).strip() for detail in children if str(detail.get("route") or "").strip()}
    if len(routes) == 1:
        event["route"] = next(iter(routes))
    return event, None


def input_event(
    row: Mapping[str, Any], item: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Normalize delivered ICU medications, excluding fluids/nutrition/blood.

    ``start/end`` are the effective matching interval. A positive finite rate
    establishes a continuous infusion; otherwise the administration matches at
    starttime only. ``source_end`` preserves the original record endpoint.
    Amount describes the whole original record, never a window-prorated dose.
    """
    if not _identifier(row.get("subject_id")):
        return None, "missing_subject"
    status = normalize_status(row.get("statusdescription")).replace(" ", "")
    if status not in ICU_DELIVERED_STATUSES:
        return None, "not_delivered_segment_status:" + status
    amount = _number(row.get("amount"))
    if amount is None or amount <= 0:
        return None, "missing_or_nonpositive_amount"
    if (not item or _identifier(item.get("itemid")) != _identifier(row.get("itemid"))
            or normalize_status(item.get("linksto")) != "inputevents"):
        return None, "missing_or_wrong_item_dictionary"
    if normalize_status(item.get("category")) not in ICU_MEDICATION_CATEGORIES:
        return None, "not_medication_item_category"
    name = str(item.get("label") or "").strip()
    if not name:
        return None, "missing_medication_name"
    start, end = parse_timestamp(row.get("starttime")), parse_timestamp(row.get("endtime"))
    if start is None or end is None or end < start:
        return None, "missing_or_invalid_event_time"
    rate = _number(row.get("rate"))
    continuous = rate is not None and rate > 0 and end > start
    event = _base(row, "icu.inputevents", name, start, end if continuous else start)
    event.update({
        "record_status": row.get("statusdescription"), "source_end": end,
        "event_type": "infusion_segment" if continuous else "administration",
        "amount": row.get("amount"), "amount_unit": row.get("amountuom") or None,
        "rate": row.get("rate") or None, "rate_unit": row.get("rateuom") or None,
        "dose_basis": "delivered amount for complete original ICU segment",
        "itemid": row.get("itemid"), "item_category": item.get("category"),
        "stay_id": row.get("stay_id") or None, "orderid": row.get("orderid") or None,
        "linkorderid": row.get("linkorderid") or None,
    })
    return event, None


def event_overlaps(
    event: Mapping[str, Any], subject_id: Any, start: Any, end: Any, *, hadm_id: Any = None,
) -> bool:
    """Exact patient and inclusive CXR overlap; admission restriction is opt-in.

    The patient-only cohort accepts records from different or missing admissions
    when their documented administration falls between the image endpoints.
    """
    lower, upper = parse_timestamp(start), parse_timestamp(end)
    if lower is None or upper is None or upper < lower:
        raise ValueError("Expected ordered naive CXR acquisition timestamps")
    if not _identifier(subject_id) or _identifier(event.get("subject_id")) != _identifier(subject_id):
        return False
    if hadm_id is not None and (not _identifier(hadm_id)
                               or _identifier(event.get("hadm_id")) != _identifier(hadm_id)):
        return False
    event_start, event_end = parse_timestamp(event.get("start")), parse_timestamp(event.get("end"))
    return bool(event_start is not None and event_end is not None and event_start <= event_end
                and event_start <= upper and event_end >= lower)
