"""Run the FastAPI backend and built React application from code/."""

from __future__ import annotations

import argparse
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

from ..clinical_tables import TABLES
from ..data import ROOT, AtlasConfig, AtlasStore
from ..patient_index import VERSION
from .api import create_api_router
from .frontend import DIST
from .vqa import create_vqa_router
from .medications import create_medication_router
from ..medication_cohort import MedicationCohort


def create_app(config=None, *, store=None, medication_cohort=None):
    store = store or AtlasStore(config or AtlasConfig())
    medications = (medication_cohort if isinstance(medication_cohort, MedicationCohort)
                   else MedicationCohort(medication_cohort, expected_cxr_root=store.cxr_root))
    medications.bind_cxr_root(store.cxr_root)

    @asynccontextmanager
    async def lifespan(app):
        store.start()
        try:
            yield
        finally:
            medications.close()
            store.close()

    app = FastAPI(
        title="MIMIC Atlas",
        version="2.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
        swagger_ui_parameters={"tryItOutEnabled": True},
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)
    app.state.store = store
    app.include_router(create_api_router(store))
    app.include_router(create_vqa_router())
    app.state.medication_cohort = medications
    app.include_router(create_medication_router(medications))
    app.mount(
        "/assets",
        StaticFiles(directory=DIST / "assets", check_dir=False),
        name="assets",
    )

    @app.middleware("http")
    async def local_headers(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/", include_in_schema=False)
    def index():
        if not (DIST / "index.html").is_file():
            raise HTTPException(
                503,
                "Build the React frontend: cd frontend && npm ci --include=dev && npm run build",
            )
        return FileResponse(DIST / "index.html")

    return app


def latest_index():
    for path in sorted(
        (ROOT / "runs").glob("patient_index_*/manifest.json"), reverse=True
    ):
        try:
            meta = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if meta.get("state") == "ready" and meta.get("version") == VERSION:
            return path.parent
    return None


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    defaults = AtlasConfig()
    parser.add_argument("--cxr-root", type=Path, default=defaults.cxr_root)
    parser.add_argument("--iv-root", type=Path, default=defaults.iv_root)
    parser.add_argument("--pairs", type=Path, default=defaults.pairs_path)
    parser.add_argument(
        "--medication-cohort", type=Path,
        help="Prepared pair selection v1 directory (default: code/data/medworld_0923)",
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        help="Prepared patient index; default: latest complete runs/patient_index_*",
    )
    parser.add_argument(
        "--include-icu-inputs",
        action="store_true",
        help="Compatibility flag; patient opening automatically loads ICU inputs",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--preload-tables",
        default="none",
        help="Comma-separated IV tables, all, or none (default: on demand)",
    )
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument(
        "--patient-cache-count", type=int, default=defaults.patient_cache_count
    )
    parser.add_argument(
        "--patient-cache-mib", type=int, default=defaults.patient_cache_bytes // 2**20
    )
    parser.add_argument(
        "--image-cache-mib", type=int, default=defaults.image_cache_bytes // 2**20
    )
    parser.add_argument(
        "--cache-idle-seconds", type=float, default=defaults.cache_idle_seconds
    )
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
        args.index_root or latest_index(),
        patient_cache_count=args.patient_cache_count,
        patient_cache_bytes=args.patient_cache_mib * 2**20,
        cache_idle_seconds=args.cache_idle_seconds,
        image_cache_bytes=args.image_cache_mib * 2**20,
    )
    uvicorn.run(create_app(config, medication_cohort=args.medication_cohort),
                host=args.host, port=args.port, log_config=None)


if __name__ == "__main__":
    main()
