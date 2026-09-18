"""HTML explorer: python -m MIMIC_example.app (run from code/)."""

from __future__ import annotations

import argparse
import base64
import copy
import csv
import io
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field
from starlette.middleware.gzip import GZipMiddleware

from .clinical_tables import TABLES, filter_events
from .data import ROOT, AtlasConfig, AtlasStore


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


def image_bytes(store, dicom_id, size, quality=90, format="jpeg"):
    image = store.images.get(dicom_id)
    if image is None:
        raise HTTPException(404, "Unknown image")
    path = image.image_path.resolve()
    if not path.is_relative_to(store.cxr_root.resolve()) or not path.is_file():
        raise HTTPException(404, "Image is unavailable")
    with Image.open(path) as im:
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format.upper(), quality=quality)
    return buf.getvalue()


def create_app(config=None, *, store=None):
    store = store or AtlasStore(config or AtlasConfig())

    @asynccontextmanager
    async def lifespan(app):
        store.start()
        yield
        store.close()

    app = FastAPI(title="MIMIC Atlas", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)
    app.state.store = store
    app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")

    @app.middleware("http")
    async def local_headers(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    def ready():
        if store.catalog()["state"] != "ready":
            raise HTTPException(503, store.catalog())

    def patient(pid):
        ready()
        try:
            return store.patient(pid)
        except KeyError:
            raise HTTPException(404, "Unknown subject") from None

    @app.get("/")
    def index():
        return FileResponse(ROOT / "web/index.html")

    @app.get("/api/catalog")
    def catalog():
        return store.catalog()

    @app.get("/api/patients")
    def patients(
        filters: Annotated[dict, Depends(patient_filters)],
        page: int = Query(1, ge=1),
        limit: int = Query(30, ge=1, le=100),
    ):
        ready()
        return store.search(page=page, limit=limit, **filters)

    @app.get("/api/cohort")
    def cohort():
        return store.cohort.catalog()

    @app.get("/api/cohort/pairs")
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

    @app.get("/api/cohort/patients.csv")
    def patients_csv(filters: Annotated[dict, Depends(patient_filters)]):
        return cohort_csv(
            store.cohort.filter_patients(store.cohort.rows, **filters), "mimic-patients"
        )

    @app.get("/api/cohort/pairs.csv")
    def pairs_csv(filters: Annotated[dict, Depends(pair_filters)]):
        return cohort_csv(store.cohort.filter_pairs(**filters), "mimic-pair-candidates")

    @app.get("/api/patients/{subject_id}")
    def detail(subject_id: str, compact: bool = False):
        data = patient(subject_id)
        return store.compact_patient(subject_id) if compact else data

    @app.get("/api/patients/{subject_id}/selection")
    def selection(subject_id: str, transition: str = "", study: str = ""):
        patient(subject_id)
        try:
            return store.selection(subject_id, transition, study)
        except KeyError:
            raise HTTPException(404, "Unknown selection") from None

    @app.get("/api/patients/{subject_id}/clinical-status")
    def clinical_status(subject_id: str):
        return patient(subject_id)["clinical_status"]

    @app.get("/api/patients/{subject_id}/context/{table_name}")
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

    @app.get("/api/images/{dicom_id}")
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

    @app.post("/api/patients/{subject_id}/icu-inputs")
    def load_inputs(subject_id: str):
        patient(subject_id)
        return store.request_inputs(subject_id)

    @app.get("/api/patients/{subject_id}/tables")
    def table_status(subject_id: str):
        patient(subject_id)
        return store.extended.manifest(subject_id)

    @app.get("/api/patients/{subject_id}/directory")
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

    @app.get("/api/patients/{subject_id}/reports/{study_id}")
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

    @app.post("/api/patients/{subject_id}/tables")
    def load_tables(subject_id: str, payload: TableRequest):
        patient(subject_id)
        try:
            return store.request_tables(subject_id, payload.tables)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    # Paging and chart requests do not re-scan the source file.
    @app.get("/api/patients/{subject_id}/tables/{table_name}")
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

    @app.get("/api/patients/{subject_id}/tables/{table_name}/export.csv")
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

    @app.get("/api/export/{subject_id}.json")
    def export_json(subject_id: str):
        data = patient(subject_id)
        return Response(
            json.dumps(data, ensure_ascii=False, indent=2),
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="mimic-{data["subject_id"]}.json"'
            },
        )

    @app.get("/api/export/{subject_id}.html")
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
        html = (ROOT / "web/index.html").read_text()
        for asset in ("cohort.js",):
            html = html.replace(
                f'<script src="/static/{asset}" defer></script>',
                "<script>" + (ROOT / "web" / asset).read_text() + "</script>",
            )
        html = html.replace(
            '<link rel="stylesheet" href="/static/explorer.css">',
            "<style>" + (ROOT / "web/explorer.css").read_text() + "</style>",
        )
        html = html.replace(
            '<script src="/static/clinical.js" defer></script>',
            "<script>" + (ROOT / "web/clinical.js").read_text() + "</script>",
        )
        html = html.replace(
            '<link rel="stylesheet" href="/static/style.css">',
            "<style>" + (ROOT / "web/style.css").read_text() + "</style>",
        )
        html = html.replace(
            '<script src="/static/app.js" defer></script>',
            "<script>window.MIMIC_SNAPSHOT="
            + snapshot
            + ";</script><script defer>"
            + (ROOT / "web/app.js").read_text()
            + "</script>",
        )
        return HTMLResponse(
            html,
            headers={
                "Content-Disposition": f'attachment; filename="{transition}.html"'
            },
        )

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    defaults = AtlasConfig()
    parser.add_argument("--cxr-root", type=Path, default=defaults.cxr_root)
    parser.add_argument("--iv-root", type=Path, default=defaults.iv_root)
    parser.add_argument("--pairs", type=Path, default=defaults.pairs_path)
    parser.add_argument(
        "--index-root",
        type=Path,
        help="Prepared patient index; default: latest complete runs/patient_index_*",
    )
    parser.add_argument(
        "--include-icu-inputs",
        action="store_true",
        help="Also scan ICU inputevents (slower)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--preload-tables",
        default="none",
        help="Comma-separated IV tables, all, or none (default: on demand)",
    )
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    log_dir = (
        ROOT / "runs" / ("atlas_" + datetime.now().astimezone().strftime("%Y%m%d"))
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler(log_dir / "server.log"), logging.StreamHandler()],
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    preload = (
        tuple(TABLES)
        if args.preload_tables == "all"
        else ()
        if args.preload_tables == "none"
        else tuple(args.preload_tables.split(","))
    )
    if any(name not in TABLES for name in preload):
        parser.error("Unknown --preload-tables name")
    config = AtlasConfig(
        args.cxr_root,
        args.iv_root,
        args.pairs,
        args.include_icu_inputs,
        preload,
        args.index_root
        or next(
            (
                p.parent
                for p in sorted(
                    (ROOT / "runs").glob("patient_index_*/manifest.json"), reverse=True
                )
            ),
            None,
        ),
    )
    uvicorn.run(create_app(config), host=args.host, port=args.port, log_config=None)


if __name__ == "__main__":
    main()
