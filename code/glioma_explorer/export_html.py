"""Build a self-contained HTML with real example slices and both cohort tables."""
import argparse
import base64
import json
import re
from pathlib import Path

from .data import DATA_ROOT, catalog, image_info, patient, slice_png, SEQUENCES

ROOT = Path(__file__).resolve().parent


def export(pid, destination, mu_pid="PatientID_0003"):
    snapshot = {"catalog": catalog(), "patients": {}, "default_patient": pid,
                "examples": {}, "slices": {}}
    for dataset, example_pid in (("ucsf", pid), ("mu", mu_pid)):
        p = dict(patient(dataset, example_pid) or {})
        if not p or not p['image_available']:
            raise ValueError(f"Example MRI unavailable: {example_pid}")
        p["image"] = image_info(example_pid)
        center = p["image"]["default_slices"]["axial"]
        indices = sorted(set(max(0, min(p["image"]["shape"][2] - 1, center + off))
                             for off in range(-20, 21, 2)))
        snapshot['patients'][example_pid] = p
        snapshot['examples'][example_pid] = {"default_index": center, "indices": indices}
        for sequence in SEQUENCES:
            for index in indices:
                for timepoint in p['image']['timepoints']:
                    key = f"{example_pid}:{sequence}:axial:{index}:{timepoint}"
                    image = slice_png(example_pid, sequence, "axial", index, timepoint, True, .45)
                    snapshot["slices"][key] = "data:image/png;base64," + base64.b64encode(image).decode()
            print(f"Rendered {example_pid} {sequence}: {len(indices)} slices × {len(p['image']['timepoints'])} timepoints", flush=True)
    html = (ROOT / "web/index.html").read_text()
    css = (ROOT / "web/style.css").read_text()
    js = (ROOT / "web/app.js").read_text()
    payload = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    html, replaced = re.subn(r'<link\b[^>]*href="/static/style\.css"[^>]*>',
                             lambda _: f'<style>{css}</style>', html)
    if replaced != 1:
        raise ValueError("Could not inline the local stylesheet")
    html = html.replace('<script src="/static/app.js"></script>',
                        '<script>window.__ATLAS_SNAPSHOT__=' + payload + ';</script>\n<script>' + js + '</script>')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html)
    print(f"Saved {destination} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", default="100004")
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "glioma_explorer/runs/glioma_atlas.html")
    parser.add_argument("--examples", action="store_true", help="Export the old two-patient demonstration instead of the complete viewer")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.examples:
        if args.output.name == 'glioma_atlas.html':
            args.output = args.output.with_name('glioma_examples.html')
        export(args.patient, args.output)
    else:
        from .export_full import main
        main(args.workers)
