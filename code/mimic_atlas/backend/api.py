"""Patient, cohort, clinical table, image and export endpoints."""

from __future__ import annotations

import base64
import copy
import csv
import io
import json
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse

from ..clinical_tables import TABLES, filter_events
from ..data import ROOT
from ..exports import ExportLibrary
from ..memory import CacheBusy
from .frontend import offline_html
from .images import image_bytes
from .schemas import TableRequest, clinical_filters, pair_filters, patient_filters


def create_api_router(store):
    def patient_lease(request: Request):
        subject = request.path_params.get("subject_id")
        # Directory and report-path lookups do not allocate patient caches.
        if (
            not subject
            or request.url.path.endswith("/directory")
            or "/reports/" in request.url.path
        ):
            yield
            return
        ready()
        if subject not in store.by_subject and subject not in store.cohort.by_id:
            raise HTTPException(404, "Unknown subject")
        try:
            with store.memory.lease(subject):
                yield
        except CacheBusy as exc:
            raise HTTPException(503, str(exc), headers={"Retry-After": "1"}) from exc

    router = APIRouter(dependencies=[Depends(patient_lease)])

    def ready():
        if store.catalog()["state"] != "ready":
            raise HTTPException(503, store.catalog())

    def patient(pid):
        ready()
        try:
            return store.patient(pid)
        except KeyError:
            raise HTTPException(404, "Unknown subject") from None

    library = ExportLibrary(ROOT / "runs" / "exports")

    @router.get("/api/exports")
    def export_library():
        return {"bundles": library.catalog()}

    @router.get("/api/exports/{identifier}/cases")
    def export_cases(
        identifier: str,
        page: int = Query(1, ge=1),
        limit: int = Query(25, ge=1, le=100),
    ):
        try:
            return library.cases(identifier, page, limit)
        except KeyError:
            raise HTTPException(404, "Unknown export bundle") from None

    @router.get("/api/exports/{identifier}/files/{filename}")
    def export_file(identifier: str, filename: str):
        try:
            path = library.artifact(identifier, filename)
        except KeyError:
            raise HTTPException(404, "Unknown export artifact") from None
        return FileResponse(
            path, filename=filename, media_type="application/octet-stream"
        )

    @router.get("/api/memory")
    def memory_status():
        return store.memory_status()

    @router.get("/api/catalog")
    def catalog():
        return store.catalog()

    @router.get("/api/patients")
    def patients(
        filters: Annotated[dict, Depends(patient_filters)],
        page: int = Query(1, ge=1),
        limit: int = Query(30, ge=1, le=100),
    ):
        ready()
        return store.search(page=page, limit=limit, **filters)

    @router.get("/api/cohort")
    def cohort():
        return store.cohort.catalog()

    @router.get("/api/cohort/pairs")
    def pairs(
        filters: Annotated[dict, Depends(pair_filters)],
        page: int = Query(1, ge=1),
        limit: int = Query(30, ge=1, le=100),
    ):
        rows = store.cohort.filter_pairs(**filters)
        return {
            "rows": rows[(page - 1) * limit : page * limit],
            "total": len(rows),
            "page": page,
            "limit": limit,
            "index_state": store.cohort.status["state"],
        }

    def cohort_csv(rows, filename):
        if store.cohort.status["state"] != "ready":
            raise HTTPException(409, "Wait for whole-cohort linkage")

        def generate():
            if not rows:
                return
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            yield output.getvalue()
            for row in rows:
                output.seek(0)
                output.truncate(0)
                writer.writerow(row)
                yield output.getvalue()

        return StreamingResponse(
            generate(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}.csv"'},
        )

    @router.get("/api/cohort/patients.csv")
    def patients_csv(filters: Annotated[dict, Depends(patient_filters)]):
        return cohort_csv(
            store.cohort.filter_patients(store.cohort.rows, **filters), "mimic-patients"
        )

    @router.get("/api/cohort/pairs.csv")
    def pairs_csv(filters: Annotated[dict, Depends(pair_filters)]):
        return cohort_csv(store.cohort.filter_pairs(**filters), "mimic-pair-candidates")

    @router.get("/api/patients/{subject_id}")
    def detail(subject_id: str, compact: bool = False):
        data = patient(subject_id)
        return store.compact_patient(subject_id) if compact else data

    @router.get("/api/patients/{subject_id}/selection")
    def selection(subject_id: str, transition: str = "", study: str = ""):
        patient(subject_id)
        try:
            return store.selection(subject_id, transition, study)
        except KeyError:
            raise HTTPException(404, "Unknown selection") from None

    @router.get("/api/patients/{subject_id}/clinical-status")
    def clinical_status(subject_id: str):
        return patient(subject_id)["clinical_status"]

    @router.get("/api/patients/{subject_id}/context/{table_name}")
    def context_rows(
        subject_id: str,
        table_name: Literal[
            "admissions",
            "transfers",
            "diagnoses_icd",
            "procedures_icd",
            "icustays",
            "procedureevents",
            "inputevents",
        ],
        page: int = Query(1, ge=1),
        limit: int = Query(25, ge=1, le=100),
        q: str = Query("", max_length=200),
        download: bool = False,
    ):
        data = patient(subject_id)
        if data["clinical_status"]["state"] != "ready":
            raise HTTPException(409, "临床背景加载中")
        if table_name == "inputevents" and not data["icu_inputs"]:
            raise HTTPException(409, "请先加载 ICU 输入事件")
        key = {"diagnoses_icd": "diagnoses", "icustays": "stays"}.get(
            table_name, table_name
        )
        rows = store.tables[subject_id].by_subject[subject_id][key]
        if q:
            rows = [r for r in rows if q.lower() in " ".join(r.values()).lower()]
        if download:
            # Explicit export includes every filtered raw row, independently of paging.
            def generate():
                if not rows:
                    return
                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                writer.writeheader()
                yield output.getvalue()
                for row in rows:
                    output.seek(0)
                    output.truncate(0)
                    writer.writerow(row)
                    yield output.getvalue()

            return StreamingResponse(
                generate(),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f'attachment; filename="mimic-{subject_id}-{table_name}.csv"'
                },
            )
        return {
            "rows": rows[(page - 1) * limit : page * limit],
            "total": len(rows),
            "page": page,
            "limit": limit,
            "fields": list(rows[0]) if rows else [],
        }

    @router.get("/api/images/{dicom_id}")
    def image(
        dicom_id: str,
        size: int = Query(512, ge=128, le=3000),
        quality: int = Query(75, ge=40, le=95),
        format: Literal["jpeg", "webp"] = "jpeg",
    ):
        ready()
        return Response(
            image_bytes(store, dicom_id, size, quality, format),
            media_type=f"image/{format}",
        )

    @router.post("/api/patients/{subject_id}/icu-inputs")
    def load_inputs(subject_id: str):
        patient(subject_id)
        return store.request_inputs(subject_id)

    @router.get("/api/patients/{subject_id}/tables")
    def table_status(subject_id: str):
        patient(subject_id)
        return store.extended.manifest(subject_id)

    @router.get("/api/patients/{subject_id}/directory")
    def patient_directory(subject_id: str):
        ready()
        if not store.index:
            raise HTTPException(409, "Patient index has not been prepared")
        counts = store.index.directory(subject_id)
        if not counts:
            raise HTTPException(404, "Unknown patient")
        return {
            "subject_id": subject_id,
            "tables": counts,
            "prepared_at": store.index.manifest["created_at"],
            "rules": store.index.manifest["cxr"]["rules"],
        }

    @router.get("/api/patients/{subject_id}/reports/{study_id}")
    def raw_report(subject_id: str, study_id: str):
        ready()
        if not store.index:
            raise HTTPException(409, "Patient index has not been prepared")
        report = store.index.reports(subject_id).get(study_id.removeprefix("s"))
        if report is None:
            raise HTTPException(404, "Study does not belong to this patient")
        return {
            "subject_id": subject_id,
            "study_id": study_id.removeprefix("s"),
            **report,
        }

    @router.post("/api/patients/{subject_id}/tables")
    def load_tables(subject_id: str, payload: TableRequest):
        patient(subject_id)
        try:
            return store.request_tables(subject_id, payload.tables)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    # Paging and chart requests do not re-scan the source file.
    @router.get("/api/patients/{subject_id}/tables/{table_name}")
    def table_records(
        subject_id: str,
        table_name: str,
        filters: Annotated[dict, Depends(clinical_filters)],
        page: int = Query(1, ge=1),
        limit: int = Query(50, ge=1, le=200),
        series_id: str = Query("", max_length=100),
    ):
        patient(subject_id)
        try:
            return store.extended.query(
                subject_id,
                table_name,
                page=page,
                limit=limit,
                series_id=series_id,
                **filters,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/api/patients/{subject_id}/tables/{table_name}/export.csv")
    def table_csv(
        subject_id: str,
        table_name: str,
        filters: Annotated[dict, Depends(clinical_filters)],
        series_id: str = Query("", max_length=100),
    ):
        patient(subject_id)
        if table_name not in TABLES:
            raise HTTPException(404, "Unknown table")
        with store.extended.lock:
            key = (subject_id, table_name)
            if store.extended.states.get(key, {}).get("state") != "ready":
                raise HTTPException(409, "Table is not ready")
            try:
                rows = filter_events(
                    store.extended.events[key], series_id=series_id, **filters
                )
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            fields = store.extended.fields[key]

        def generate():
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            yield output.getvalue()
            for row in rows:
                output.seek(0)
                output.truncate(0)
                writer.writerow(row["raw"])
                yield output.getvalue()

        return StreamingResponse(
            generate(),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="mimic-{subject_id}-{table_name}.csv"'
            },
        )

    @router.get("/api/export/{subject_id}.json")
    def export_json(subject_id: str):
        data = patient(subject_id)
        return Response(
            json.dumps(data, ensure_ascii=False, indent=2),
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="mimic-{data["subject_id"]}.json"'
            },
        )

    @router.get("/api/export/{subject_id}.html")
    def export_html(subject_id: str, transition: str):
        data = copy.deepcopy(patient(subject_id))
        if data["clinical_status"]["state"] != "ready":
            raise HTTPException(409, "Wait for MIMIC-IV context before exporting HTML")
        selected = next(
            (r for r in data["transitions"] if r["transition_id"] == transition), None
        )
        if selected is None:
            raise HTTPException(404, "Unknown transition")
        packet = selected["mimic_cxr_transition"]
        ids = {packet[s]["study_id"] for s in ("current_state", "future_state")}
        data["studies"] = [s for s in data["studies"] if s["study_id"] in ids]
        data["transitions"] = [selected]
        data["preferred_transition"] = transition
        for study in data["studies"]:
            for image in study["images"]:
                if image["available"]:
                    encoded = base64.b64encode(
                        image_bytes(store, image["dicom_id"], 1400)
                    ).decode()
                    image["url"] = "data:image/jpeg;base64," + encoded
        catalog = store.catalog()
        catalog["featured"] = [
            r for r in catalog["featured"] if r["subject_id"] == subject_id
        ] or [
            {
                "subject_id": subject_id,
                "split": data["split"],
                "studies": len(data["studies"]),
            }
        ]
        snapshot = json.dumps(
            {
                "catalog": catalog,
                "patient": data,
                "extended": store.extended.export_window(
                    subject_id,
                    packet["current_state"]["image"]["acquisition_timestamp"],
                    packet["future_state"]["image"]["acquisition_timestamp"],
                ),
            },
            ensure_ascii=False,
        ).replace("<", "\\u003c")
        html = offline_html(snapshot)
        return HTMLResponse(
            html,
            headers={
                "Content-Disposition": f'attachment; filename="{transition}.html"'
            },
        )

    return router
