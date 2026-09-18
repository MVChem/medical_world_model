"""Validated API query parameters and request bodies."""

from typing import Literal

from fastapi import Query
from pydantic import BaseModel, Field


class TableRequest(BaseModel):
    tables: list[str] = Field(min_length=1, max_length=20)


def clinical_filters(
    scope: Literal["patient", "window", "admission"] = "patient",
    start: str | None = None,
    end: str | None = None,
    hadm_id: str = "",
    q: str = Query("", max_length=200),
):
    return {"scope": scope, "start": start, "end": end, "hadm_id": hadm_id, "q": q}


def patient_filters(
    q: str = Query("", max_length=100),
    split: Literal["all", "train", "validate", "test", "none"] = "all",
    longitudinal: bool = False,
    coverage: Literal["all", "cxr", "matched", "cxr_only", "iv_only"] = "all",
    paired: Literal["all", "pairs", "linked"] = "all",
    sort: Literal["subject", "studies", "pairs", "linked_pairs"] = "subject",
    featured: bool = False,
):
    return {
        "query": q.strip(),
        "split": split,
        "longitudinal": longitudinal,
        "coverage": coverage,
        "paired": paired,
        "sort": sort,
        "featured": featured,
    }


def pair_filters(
    q: str = Query("", max_length=100),
    split: Literal["all", "train", "validate", "test"] = "all",
    linkage: Literal["all", "unique", "unmatched", "ambiguous"] = "all",
):
    return {"query": q.strip(), "split": split, "linkage": linkage}
