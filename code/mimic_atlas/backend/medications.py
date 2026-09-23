"""Read-only APIs for the independently prepared medication-pair selection."""
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from ..medication_cohort import MedicationCohort


def create_medication_router(catalog: MedicationCohort):
    router = APIRouter(prefix="/api/medication-cohort", tags=["Medication pairs"])

    def call(function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except KeyError as exc:
            raise HTTPException(404, "Unknown medication pair") from exc
        except (RuntimeError, OSError, ValueError) as exc:
            raise HTTPException(503, str(exc)) from exc

    @router.get("")
    def summary():
        return catalog.status()

    @router.get("/pairs")
    def pairs(q: str = Query("", max_length=200), subject_id: str = Query("", max_length=32),
              split: Literal["all", "train", "validate", "test", "unassigned"] = "all",
              page: int = Query(1, ge=1), limit: int = Query(25, ge=1, le=100)):
        return call(catalog.search, q=q, subject_id=subject_id, split=split, page=page, limit=limit)

    @router.get("/pairs/{identifier}")
    def pair(identifier: str):
        return call(catalog.detail, identifier)

    @router.get("/pairs/{identifier}/medications")
    def medications(identifier: str, source: Literal["all", "hosp.emar", "icu.inputevents"] = "all",
                    q: str = Query("", max_length=200), page: int = Query(1, ge=1),
                    limit: int = Query(25, ge=1, le=100)):
        return call(catalog.medications, identifier, source=source, q=q, page=page, limit=limit)

    return router
