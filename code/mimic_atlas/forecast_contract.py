"""Shared inference-time forecasting prompt contract.

This module intentionally depends only on the Python standard library so both
the standalone MIMIC builder and the training package can use exactly the same
horizon bins and prompt text.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

HORIZON_BINS = ("0-24h", "24-72h", "3-7d", ">7d")
HORIZON_DESCRIPTIONS = {
    "0-24h": "within 24 hours",
    "24-72h": "between 24 and 72 hours",
    "3-7d": "between 3 and 7 days",
    ">7d": "more than 7 days",
}


def horizon_bin_from_elapsed_hours(elapsed_hours: float) -> str:
    """Map a realized follow-up interval to a prespecified coarse bin."""

    if (
        isinstance(elapsed_hours, bool)
        or not isinstance(elapsed_hours, (int, float))
        or not math.isfinite(elapsed_hours)
        or elapsed_hours <= 0
    ):
        raise ValueError("elapsed_hours must be a finite positive number")
    if elapsed_hours <= 24:
        return "0-24h"
    if elapsed_hours <= 72:
        return "24-72h"
    if elapsed_hours <= 168:
        return "3-7d"
    return ">7d"


def validate_horizon_bin(value: Any) -> str:
    if not isinstance(value, str) or value not in HORIZON_BINS:
        raise ValueError(f"horizon_bin must be one of {list(HORIZON_BINS)}")
    return value


def _clean_report_part(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"source_report.{label} must be a string or null")
    value = " ".join(value.split())
    return value or None


def build_coarse_horizon_prompt(
    source_report: Mapping[str, Any] | None, horizon_bin: str
) -> str:
    """Build model text from current report plus a coarse, prespecified horizon."""

    horizon_bin = validate_horizon_bin(horizon_bin)
    if source_report is None:
        source_report = {}
    if not isinstance(source_report, Mapping):
        raise TypeError("source_report must be an object")

    findings = _clean_report_part(source_report.get("findings"), "findings")
    impression = _clean_report_part(source_report.get("impression"), "impression")
    unsectioned = _clean_report_part(
        source_report.get("unsectioned_report"), "unsectioned_report"
    )

    report_lines: list[str] = []
    if findings:
        report_lines.append(f"CURRENT FINDINGS: {findings}")
    if impression:
        report_lines.append(f"CURRENT IMPRESSION: {impression}")
    if unsectioned and not report_lines:
        report_lines.append(f"CURRENT REPORT: {unsectioned}")
    if not report_lines:
        report_lines.append("CURRENT REPORT: unavailable")

    return "\n".join(
        [
            "Infer a compact latent representation of the next observed chest-radiograph state.",
            (
                f"Prediction horizon: {horizon_bin} "
                f"({HORIZON_DESCRIPTIONS[horizon_bin]} after the current study)."
            ),
            (
                "Use only the current image, current report, and prespecified horizon bin. "
                "Represent uncertainty conservatively."
            ),
            *report_lines,
        ]
    )
