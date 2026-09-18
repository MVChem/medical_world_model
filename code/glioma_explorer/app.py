"""Local-only FastAPI service. Run: PYTHONPATH=code python -m glioma_explorer.app."""
import argparse
import csv
import io
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .data import DATA_ROOT, catalog, image_info, patient, slice_png, selected_timepoints

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="Glioma Atlas", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
app.mount("/preview-assets", StaticFiles(directory=DATA_ROOT / "glioma_explorer/runs/full_data", check_dir=False), name="preview-assets")


@app.get("/")
def index():
    return FileResponse(ROOT / "web/index.html")


@app.get("/api/catalog")
def get_catalog():
    return catalog()


@app.get("/api/patient/{dataset}/{pid}")
def get_patient(dataset: Literal["ucsf", "mu"], pid: str, first: int | None = None, second: int | None = None):
    p = patient(dataset, pid)
    if p is None:
        raise HTTPException(404, "Patient not found")
    result = dict(p)
    if p['image_available']:
        try:
            result["image"] = image_info(pid, selected_timepoints(pid, first, second))
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
    return result


@app.get("/api/slice/{pid}.png")
def get_slice(pid: str, sequence: Literal["t1ce", "flair", "t1", "t2"] = "t1ce",
              plane: Literal["axial", "coronal", "sagittal"] = "axial",
              index: int = Query(75, ge=0), timepoint: int = Query(1, ge=1, le=6),
              overlay: bool = True, opacity: float = Query(.45, ge=0, le=1),
              first: int | None = None, second: int | None = None):
    dataset = "mu" if pid.startswith("PatientID_") else "ucsf"
    if not patient(dataset, pid):
        raise HTTPException(404, "Patient not found")
    try:
        data = slice_png(pid, sequence, plane, index, timepoint, overlay, opacity,
                         selected_timepoints(pid, first, second))
    except (ValueError, KeyError) as e:
        raise HTTPException(422, str(e)) from e
    return Response(data, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


@app.get("/api/export/{dataset}.csv")
def export_csv(dataset: Literal["ucsf", "mu"]):
    out = io.StringIO()
    fields = ["id", "age", "sex", "diagnosis", "grade", "timepoints", "gap_days", "image_available"]
    writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in catalog()[dataset]["patients"]:
        safe = {k: ("'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else v)
                for k, v in row.items()}
        writer.writerow(safe)
    return Response("\ufeff" + out.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{dataset}_patients.csv"'})


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
