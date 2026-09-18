"""Export every patient, original volume and worksheet for a complete static viewer.

The HTML loads compressed display volumes on demand using local JavaScript assets,
including inside the chat's sandboxed HTML preview. No data server is required.
Original NIfTI remain in DATA_ROOT; these caches retain every voxel on the display
grid with the same 8-bit window used by the live viewer, plus discrete mask labels.
"""
import argparse
import base64
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re

import nibabel as nib
import numpy as np
import pandas as pd

from .data import DATA_ROOT, UCSF_ROOT, MU_ROOT, SEQUENCES, MU_SUFFIXES, align, catalog, clean, image_info, load_volume

ROOT = Path(__file__).resolve().parent
OUT = DATA_ROOT / "glioma_explorer/runs"
ASSETS = OUT / "full_data"
VERSION = 1


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    temporary.replace(path)


def script(path, callback, key, value):
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    text = f"window.{callback}({json.dumps(key)},{payload});\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)
    return len(text.encode())


def render_patient(task):
    dataset, p = task
    pid = p['id']
    root = UCSF_ROOT if dataset == 'ucsf' else MU_ROOT
    files = sorted((root / pid).glob('*.nii.gz' if dataset == 'ucsf' else 'Timepoint_*/*.nii.gz'))
    fingerprint = [(f.relative_to(root).as_posix(), f.stat().st_size, f.stat().st_mtime_ns) for f in files]
    directory = ASSETS / dataset / pid
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / 'patient.json'
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        if old.get('version') == VERSION and old.get('fingerprint') == [list(row) for row in fingerprint]:
            if all((ASSETS / f['asset']).is_file() and (ASSETS / f['asset']).stat().st_size == f['asset_bytes'] for f in old['patient']['files']):
                return old['summary']
    timepoints = (1, 2) if dataset == 'ucsf' else tuple(v['timepoint'] for v in p['image_visits'])
    if not timepoints:
        raise ValueError(f'No image visits: {pid}')
    p = dict(p)
    p['image'] = image_info(pid, timepoints)
    reference = load_volume(pid, timepoints[0], 't1ce')
    mm_grids = []
    for f in files:
        header = nib.as_closest_canonical(nib.load(f))
        if header.header.get_xyzt_units()[0] == 'mm':
            mm_grids.append((header.shape, header.affine, f.name))
    p.update(files=[], volume_keys={})
    p['image']['display_reference_timepoint'] = timepoints[0]
    reverse = {value: key for key, value in MU_SUFFIXES.items()}
    for source in files:
        name = source.name
        if dataset == 'ucsf':
            match = re.fullmatch(rf'{pid}_time(\d+)_(.+)\.nii\.gz', name)
            timepoint, sequence = (int(match[1]), match[2]) if match else (None, None)
        else:
            match = re.fullmatch(rf'{pid}_Timepoint_(\d+)_(.+)\.nii\.gz', name)
            timepoint, sequence = int(match[1]), reverse[match[2]]
        kind = 'label' if ('seg' in name or 'tumorMask' in name) else 'signed' if ('subtraction' in name or 't1ce-t1' in name) else 'mri'
        native = nib.as_closest_canonical(nib.load(source))
        if len(native.shape) != 3:
            raise ValueError(f'Invalid 3D image: {source}')
        source_unit = native.header.get_xyzt_units()[0]
        unit_basis = 'original_header'
        if source_unit != 'mm':
            matching = [name for shape, affine, name in mm_grids if shape == native.shape and np.allclose(affine, native.affine, atol=1e-4)]
            if source_unit != 'unknown' or not matching:
                raise ValueError(f'Unresolved spatial unit: {source}')
            unit_basis = 'same_shape_and_affine_as_mm_file:' + matching[0]
        volume = align(native, reference, mask=kind == 'label')
        array = np.asarray(volume.dataobj, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError(f'Nonfinite image values: {source}')
        if kind == 'label':
            values = np.unique(array)
            if values.min() < 0 or values.max() > 255 or not np.equal(values, values.astype(np.uint8)).all():
                raise ValueError(f'Non-byte discrete labels: {source}: {values}')
            window = None
            gray = array.astype(np.uint8)
        else:
            samples = array.ravel()[::19]
            if kind == 'signed':
                nonzero = np.abs(samples[samples != 0])
                limit = max(float(np.percentile(nonzero, 99.5)), 1) if len(nonzero) else 1
                lo, hi = -limit, limit
            else:
                samples = samples[samples > 0]
                lo, hi = np.percentile(samples, [1, 99.5]) if len(samples) else (0, 1)
                hi = max(hi, lo + 1)
            window = [float(lo), float(hi)]
            gray = np.uint8(np.clip((array - lo) / (hi - lo), 0, 1) * 255)
        raw = np.ascontiguousarray(gray).tobytes()
        packed = gzip.compress(raw, compresslevel=3, mtime=0)
        if gzip.decompress(packed) != raw:
            raise ValueError('Display cache round-trip failed')
        key = f'{dataset}/{pid}/{name}'
        asset = f'{dataset}/{pid}/{name}.js'
        description = {'key': key, 'name': name, 'asset': asset, 'kind': kind,
                       'timepoint': timepoint, 'sequence': sequence,
                       'shape': list(gray.shape), 'spacing': list(map(float, reference.header.get_zooms()[:3])),
                       'window': window, 'labels': list(map(int, np.unique(gray))) if kind == 'label' else None,
                       'source_bytes': source.stat().st_size,
                       'source_spatial_unit': source_unit, 'display_unit_basis': unit_basis,
                       'source_path': source.relative_to(root).as_posix(),
                       'voxel_bytes': len(raw), 'voxel_sha256': hashlib.sha256(raw).hexdigest()}
        value = {**description, 'encoding': 'gzip-base64-uint8-c-order', 'data': base64.b64encode(packed).decode()}
        description['asset_bytes'] = script(ASSETS / asset, '__ATLAS_VOLUME__', key, value)
        p['files'].append(description)
        if sequence in (*SEQUENCES, 'seg') and timepoint is not None:
            p['volume_keys'][f'{sequence}:{timepoint}'] = key
    summary = {'dataset': dataset, 'patient': pid, 'timepoints': list(timepoints),
               'original_files_loaded': len(files), 'mri_volumes': sum(f['sequence'] in SEQUENCES for f in p['files']),
               'asset_bytes': sum(f['asset_bytes'] for f in p['files']),
               'metadata': f'{dataset}/{pid}/patient.js'}
    record = {'version': VERSION, 'fingerprint': fingerprint, 'patient': clean(p), 'summary': summary}
    script(directory / 'patient.js', '__ATLAS_PATIENT__', pid, clean(p))
    dump(metadata_path, record)
    return summary


def cell(value):
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return clean(value)


def export_tables():
    result = {}
    for dataset, root in [('ucsf', UCSF_ROOT), ('mu', MU_ROOT)]:
        result[dataset] = []
        for path in sorted(root.glob('*.xlsx')):
            with pd.ExcelFile(path) as workbook:
                for sheet in workbook.sheet_names:
                    # Preserve all original rows, including title/dictionary/header rows.
                    frame = workbook.parse(sheet, header=None)
                    result[dataset].append({'workbook': path.name, 'sheet': sheet,
                                            'rows': [[cell(v) for v in row] for row in frame.itertuples(index=False, name=None)]})
    script(ASSETS / 'tables.js', '__ATLAS_TABLES__', 'all', result)
    return {dataset: [{'workbook': t['workbook'], 'sheet': t['sheet'], 'rows': len(t['rows'])} for t in tables]
            for dataset, tables in result.items()}


def write_html(destination, summaries):
    snapshot = {'mode': 'volumes', 'asset_base': 'full_data/', 'catalog': catalog(),
                'default_patient': '100004', 'patient_index': {r['patient']: r['metadata'] for r in summaries}}
    html = (ROOT / 'web/index.html').read_text()
    html = re.sub(r'<link\b[^>]*href="/static/style\.css"[^>]*>',
                  lambda _: '<style>' + (ROOT / 'web/style.css').read_text() + '</style>', html)
    payload = json.dumps(snapshot, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    js = (ROOT / 'web/full_renderer.js').read_text() + '\n' + (ROOT / 'web/app.js').read_text()
    html = html.replace('<script src="/static/app.js"></script>',
                        '<script>window.__ATLAS_SNAPSHOT__=' + payload + ';</script>\n<script>' + js + '</script>')
    html = html.replace('<script src="/static/full_renderer.js"></script>', '')
    temporary = destination.with_suffix('.html.tmp')
    temporary.write_text(html)
    temporary.replace(destination)


def main(workers):
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    tasks = [(d, p) for d, cohort in catalog().items() for p in cohort['patients']]
    state = {'phase': 'building', 'total_patients': len(tasks), 'completed_patients': 0,
             'original_files_loaded': 0, 'asset_bytes': 0, 'errors': []}
    summaries = []
    dump(OUT / 'full_preview_status.json', state)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(render_patient, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                summaries.append(result)
                state['completed_patients'] += 1
                state['original_files_loaded'] += result['original_files_loaded']
                state['asset_bytes'] += result['asset_bytes']
            except Exception as error:
                state['errors'].append({'dataset': task[0], 'patient': task[1]['id'], 'error': str(error)})
            state['updated_at'] = datetime.now(timezone.utc).isoformat()
            dump(OUT / 'full_preview_status.json', state)
            if state['completed_patients'] % 10 == 0:
                print(json.dumps(state), flush=True)
    if state['errors']:
        state['phase'] = 'failed'
        dump(OUT / 'full_preview_status.json', state)
        raise SystemExit(1)
    sheets = export_tables()
    summaries.sort(key=lambda r: (r['dataset'], r['patient']))
    write_html(OUT / 'glioma_atlas.html', summaries)
    state.update(phase='complete', sheets=sheets, patients=summaries,
                 image_timepoints=sum(len(r['timepoints']) for r in summaries),
                 mri_volumes=sum(r['mri_volumes'] for r in summaries))
    dump(OUT / 'full_preview_status.json', state)
    print(json.dumps({k: v for k, v in state.items() if k not in ('patients', 'sheets')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    main(parser.parse_args().workers)
