"""Read original local data; never infer MRI visits from spreadsheet row order."""
from __future__ import annotations

import io
import math
import os
import re
import threading
from collections import Counter
from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to
from PIL import Image

DATA_ROOT = Path(os.environ.get("GLIOMA_DATA_ROOT", "/home/data2/chk/data"))
UCSF_ROOT = DATA_ROOT / "UCSF-ALPTDG"
MU_ROOT = DATA_ROOT / "MU-Glioma-Post"
LABELS = [
    {"id": 1, "name": "NCR", "description": "非增强／坏死核心", "color": "#b8a1e3"},
    {"id": 2, "name": "SNFH", "description": "周围 FLAIR 高信号", "color": "#65c6bb"},
    {"id": 3, "name": "ET", "description": "增强组织", "color": "#efb861"},
    {"id": 4, "name": "RC", "description": "切除腔", "color": "#e291a3"},
]
PLANES = {"axial": 2, "coronal": 1, "sagittal": 0}
SEQUENCES = ("t1ce", "flair", "t1", "t2")
MU_SUFFIXES = {"t1ce": "brain_t1c", "t1": "brain_t1n", "flair": "brain_t2f",
               "t2": "brain_t2w", "seg": "tumorMask"}
MRI_LOCK = threading.RLock()  # Bound peak memory during concurrent slice requests.


def clean(v):
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [clean(x) for x in v]
    if isinstance(v, np.generic):
        return clean(v.item())
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    return v


def number(v):
    # Deliberately exclude numeric strings and unparseable clinical annotations.
    return float(v) if isinstance(v, (int, float, np.number)) and pd.notna(v) else None


def counts(values):
    return [{"name": str(k), "count": n} for k, n in
            Counter("未记录" if v is None or pd.isna(v) else v for v in values).most_common()]


def histogram(values, edges):
    finite = [v for v in values if v is not None and math.isfinite(v)]
    hist, _ = np.histogram(finite, bins=edges)
    return [{"label": f"{int(a)}–{int(b)}", "count": int(n), "from": a, "to": b}
            for a, b, n in zip(edges, edges[1:], hist)]


@lru_cache(maxsize=1)
def catalog():
    if not UCSF_ROOT.is_dir():
        raise FileNotFoundError(f"Extracted UCSF directory missing: {UCSF_ROOT}")
    names = {p.relative_to(UCSF_ROOT).as_posix() for p in UCSF_ROOT.glob('*/*.nii.gz')}
    tables = list(UCSF_ROOT.glob('UCSF_PostopGlioma*.xlsx'))
    if len(tables) != 1:
        raise ValueError(f"Expected one UCSF clinical table, found {len(tables)}")
    with pd.ExcelFile(tables[0]) as x:
        visits = x.parse("UCSF-LPTDG")
        clinical = x.parse("Clinical Info")
    ucsf = []
    for _, row in clinical.iterrows():
        pid = str(int(row.SubjectID))
        v = visits[visits.SubjectID == row.SubjectID].sort_values("Timepoint")
        first = v.iloc[0]
        present = all(f"{pid}/{pid}_time{t}_{s}.nii.gz" in names
                      for t in (1, 2) for s in (*SEQUENCES, "seg"))
        ucsf.append({"id": pid, "age": number(first["Patient Age"]),
                     "sex": clean(first["Patient Sex"]), "diagnosis": clean(row["WHO 2021 Diagnosis"]),
                     "grade": number(row["Grade"]), "idh": clean(row["IDH"]),
                     "timepoints": 2, "gap_days": number(row["Days from 1st scan to 2nd scan"]),
                     "times": [0, number(row["Days from 1st scan to 2nd scan"])],
                     "image_available": present, "scanner": clean(first["Scanner"]),
                     "clinical": clean(row.to_dict())})
    d = pd.read_excel(MU_ROOT / "MU-Glioma-Post_ClinicalData-July2025.xlsx", sheet_name="MU Glioma Post")
    d = d[d.Patient_ID.astype(str).str.startswith("PatientID_")]
    time_cols = [c for c in d.columns if "MRI (Timepoint_" in c]
    mu, gaps, numeric_entries = [], [], 0
    mu_files = {p.relative_to(MU_ROOT).as_posix(): p.stat().st_size
                for p in MU_ROOT.glob('PatientID_*/Timepoint_*/*.nii.gz')}
    for _, row in d.iterrows():
        timeline = [{"timepoint": int(re.search(r"Timepoint_(\d+)", c)[1]), "day": number(row[c])}
                    for c in time_cols if number(row[c]) is not None]
        timeline.sort(key=lambda t: t["day"])
        times = sorted(set(t["day"] for t in timeline))
        numeric_entries += len(timeline)
        pair_gaps = [b - a for a, b in zip(times, times[1:]) if b > a]
        gaps.extend(pair_gaps)
        image_visits = []
        days = {t["timepoint"]: t["day"] for t in timeline}
        for timepoint in range(1, 7):
            stem = f"{row.Patient_ID}/Timepoint_{timepoint}/{row.Patient_ID}_Timepoint_{timepoint}_"
            if all(stem + MU_SUFFIXES[s] + '.nii.gz' in mu_files for s in SEQUENCES):
                image_visits.append({"timepoint": timepoint, "day": days.get(timepoint),
                                     "mask_available": stem + 'tumorMask.nii.gz' in mu_files})
        image_visits.sort(key=lambda t: (t["day"] is None, t["day"] or 0, t["timepoint"]))
        for t in timeline:
            t["image_available"] = any(v["timepoint"] == t["timepoint"] for v in image_visits)
        mu.append({"id": row.Patient_ID, "age": number(row["Age at diagnosis"]),
                   "sex": clean(row["Sex at Birth"]), "diagnosis": clean(row["Primary Diagnosis"]),
                   "grade": number(row["Grade of Primary Brain Tumor"]),
                   "timepoints": len(timeline), "gap_days": float(np.median(pair_gaps)) if pair_gaps else None,
                   "times": times, "timeline": timeline, "image_available": bool(image_visits),
                   "image_visits": image_visits,
                   "clinical": clean(row.to_dict())})
    scanners = pd.read_excel(MU_ROOT / "MR_Scanner_data.xlsx")
    volume_table = pd.ExcelFile(MU_ROOT / "MU-Glioma-Post_Segmentation_Volumes.xlsx")
    volume_summary = []
    for i, sheet in enumerate(volume_table.sheet_names):
        vol = volume_table.parse(sheet)
        arr = pd.to_numeric(vol["Volume (mm^3)"], errors="coerce").dropna() / 1000
        volume_summary.append({"label": "NETC" if i == 0 else LABELS[i]["name"], "rows": len(arr),
                               "median_ml": float(arr.median()),
                               "q1_ml": float(arr.quantile(.25)), "q3_ml": float(arr.quantile(.75))})
    volume_table.close()
    ugaps = [p["gap_days"] for p in ucsf if p["gap_days"] is not None]
    result = {"ucsf": {"name": "UCSF-ALPTDG", "patients": ucsf, "visits": len(visits),
                       "pairs": len(ucsf), "gaps": ugaps, "image_available": True,
                       "file_count": sum(n.endswith(".nii.gz") for n in names),
                       "source": "https://imagingdatasets.ucsf.edu/dataset/2",
                       "storage": "extracted_directory", "storage_path": str(UCSF_ROOT),
                       "license": "非商业使用 · UCSF DUA", "size": "30.18 GB · 已解压", "age_label": "首次扫描年龄"},
              "mu": {"name": "MU-Glioma-Post", "patients": mu, "visits": numeric_entries,
                     "pairs": len(gaps), "gaps": gaps, "image_available": any(p['image_available'] for p in mu),
                     "image_visits": sum(len(p['image_visits']) for p in mu),
                     "mask_visits": sum(v['mask_available'] for p in mu for v in p['image_visits']),
                     "file_count": len(mu_files), "storage_path": str(MU_ROOT),
                     "download_source": "https://huggingface.co/datasets/sbandred/mu-glioma-post-raw",
                     "scanner_rows": len(scanners), "volume_summary": volume_summary,
                     "scanner_vendors": counts(scanners["MR Vendor"]),
                     "source": "https://www.cancerimagingarchive.net/collection/mu-glioma-post/",
                     "license": "CC BY 4.0", "size": f"{sum(mu_files.values()) / 1e9:.2f} GB · 本地影像", "age_label": "诊断时年龄"}}
    for dataset in result.values():
        ps = dataset["patients"]
        dataset.update(patient_count=len(ps), median_gap=float(np.median(dataset["gaps"])),
                       diagnosis=counts(p["diagnosis"] for p in ps), sex=counts(p["sex"] for p in ps),
                       grades=counts(p["grade"] for p in ps),
                       age_histogram=histogram([p["age"] for p in ps], list(range(0, 101, 10))),
                       gap_histogram=histogram(dataset["gaps"], [0, 30, 60, 90, 180, 365, 730, 1500]),
                       timepoint_counts=counts(p["timepoints"] for p in ps))
    return clean(result)


def patient(dataset, pid):
    if dataset not in catalog():
        raise KeyError("Unknown dataset")
    return next((p for p in catalog()[dataset]["patients"] if p["id"] == pid), None)


@lru_cache(maxsize=4)
def load_volume(pid, timepoint, sequence):
    dataset = "mu" if pid.startswith("PatientID_") else "ucsf"
    p = patient(dataset, pid)
    visits = [v["timepoint"] for v in p["image_visits"]] if p and dataset == "mu" else [1, 2]
    if not p or timepoint not in visits or sequence not in (*SEQUENCES, "seg"):
        raise ValueError("Invalid image identifier")
    if dataset == "mu":
        name = f"{pid}/Timepoint_{timepoint}/{pid}_Timepoint_{timepoint}_{MU_SUFFIXES[sequence]}.nii.gz"
        path = MU_ROOT / name
        if sequence == "seg" and not path.is_file():
            return None  # A missing annotation is not an empty tumor mask.
    else:
        path = UCSF_ROOT / f"{pid}/{pid}_time{timepoint}_{sequence}.nii.gz"
    image = nib.load(path)
    canonical = nib.as_closest_canonical(image)
    # Materialize once, then release the file proxy from the cached volume.
    dtype = np.uint8 if sequence == "seg" else np.float32
    return nib.Nifti1Image(np.asarray(canonical.dataobj, dtype=dtype).copy(), canonical.affine, canonical.header)


def align(image, reference, mask=False):
    if image.shape == reference.shape and np.allclose(image.affine, reference.affine, atol=1e-4):
        return image
    return resample_from_to(image, reference, order=0 if mask else 1)


def selected_timepoints(pid, first=None, second=None):
    dataset = "mu" if pid.startswith("PatientID_") else "ucsf"
    p = patient(dataset, pid)
    if not p:
        raise ValueError("Unknown image patient")
    available = [v['timepoint'] for v in p['image_visits']] if dataset == "mu" else [1, 2]
    selected = tuple(t for t in (first, second) if t is not None) if first is not None else tuple(available[:2])
    if not selected or any(t not in available for t in selected) or len(set(selected)) != len(selected):
        raise ValueError("Invalid image timepoint selection")
    return selected


@lru_cache(maxsize=2)
def image_pair(pid, sequence, timepoints=None):
    timepoints = timepoints or selected_timepoints(pid)
    reference = load_volume(pid, timepoints[0], "t1ce")
    images = [align(load_volume(pid, t, sequence), reference) for t in timepoints]
    masks = [load_volume(pid, t, "seg") for t in timepoints]
    masks = [align(m, reference, mask=True) if m is not None else None for m in masks]
    arrays = [np.asarray(img.dataobj, dtype=np.float32) for img in images]
    # MRI intensity units vary between scans/scanners. Window each volume separately;
    # normalized display brightness must not be interpreted as longitudinal change.
    windows = []
    for a in arrays:
        samples = a.ravel()[::19]
        samples = samples[np.isfinite(samples) & (samples > 0)]
        lo, hi = np.percentile(samples, [1, 99.5]) if len(samples) else (0, 1)
        windows.append((float(lo), float(max(hi, lo + 1))))
    return arrays, [np.asarray(m.dataobj, dtype=np.uint8) if m is not None else None for m in masks], reference, windows


@lru_cache(maxsize=12)
def image_info(pid, timepoints=None):
    with MRI_LOCK:
        timepoints = timepoints or selected_timepoints(pid)
        masks = [load_volume(pid, t, "seg") for t in timepoints]
        ref = load_volume(pid, timepoints[0], "t1ce")
        arrays = [np.asarray(align(m, ref, mask=True).dataobj, dtype=np.uint8) if m is not None else None for m in masks]
        source = arrays[0] if arrays[0] is not None else np.zeros(ref.shape, dtype=np.uint8)
        # Source-timepoint label burden selects the initial slice; no future label selection.
        foreground = (source > 0) & (source != 4)
        centers = {plane: int(np.argmax(foreground.sum(axis=tuple(i for i in range(3) if i != axis))))
                   if foreground.any() else source.shape[axis] // 2 for plane, axis in PLANES.items()}
        volumes = []
        for m in masks:
            if m is None:
                volumes.append(None)
                continue
            a = np.asarray(m.dataobj)
            if m.header.get_xyzt_units()[0] != "mm":
                raise ValueError("Cannot report mL without verified millimetre spatial units")
            voxel_mm3 = abs(float(np.linalg.det(m.affine[:3, :3])))
            volumes.append({str(label["id"]): round(float(np.count_nonzero(a == label["id"])) * voxel_mm3 / 1000, 4)
                            for label in LABELS})
        labels = [dict(label) for label in LABELS]
        if pid.startswith("PatientID_"):
            labels[0].update(name="NETC", description="非增强肿瘤核心")
        return {"shape": list(ref.shape), "spacing": list(map(float, ref.header.get_zooms()[:3])),
                "orientation": "RAS+ · neurological", "default_slices": centers,
                "volumes_ml": volumes, "labels": labels, "timepoints": list(timepoints),
                "mask_available": [m is not None for m in masks],
                "label_values": [list(map(int, np.unique(a))) if a is not None else None for a in arrays]}


def slice_array(array, plane, index):
    return np.rot90(np.take(array, index, axis=PLANES[plane]))


def slice_png(pid, sequence, plane, index, timepoint, overlay, opacity, timepoints=None):
    with MRI_LOCK:
        timepoints = timepoints or selected_timepoints(pid)
        slot = timepoints.index(timepoint)
        arrays, masks, ref, windows = image_pair(pid, sequence, timepoints)
        lo, hi = windows[slot]
        axis = PLANES[plane]
        if not 0 <= index < ref.shape[axis]:
            raise ValueError("Slice outside volume")
        a = slice_array(arrays[slot], plane, index)
        gray = np.uint8(np.clip(np.nan_to_num((a - lo) / (hi - lo)), 0, 1) * 255)
        rgb = np.repeat(gray[..., None], 3, axis=-1).astype(float)
        if overlay and masks[slot] is not None:
            m = slice_array(masks[slot], plane, index)
            for label in LABELS:
                color = np.array([int(label["color"][i:i + 2], 16) for i in (1, 3, 5)])
                use = m == label["id"]
                rgb[use] = rgb[use] * (1 - opacity) + color * opacity
        img = Image.fromarray(np.uint8(rgb))
        axes = [i for i in range(3) if i != axis]
        spacing = ref.header.get_zooms()
        # Account for physical pixel aspect ratio, including non-isotropic data.
        width, height = img.width * spacing[axes[0]], img.height * spacing[axes[1]]
        scale = 640 / max(width, height)
        img = img.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.NEAREST)
        out = io.BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()
