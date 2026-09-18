"""Build a deterministic patient-disjoint pilot from local MIMIC-CXR.

Reuse the existing audited metadata, timestamp and split parser. Select pairs
without using target labels. Reports are Findings + Impression only; missing
sections are excluded. Report availability timestamps are NOT present in CXR.
This is therefore a retrospective report-available pilot, not a validated
prospective acquisition-time forecast.
"""
import argparse
import hashlib
from collections import Counter
from pathlib import Path
import sys

import numpy as np

from common import atomic_json, read_config, write_rows

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mimic_atlas.build_mimic_transitions import (
    Audit, BuildConfig, LABEL_COLUMNS, attach_labels, attach_splits,
    find_candidates, load_studies, read_report,
)


def report_text(path):
    sections = read_report(path, 12000)
    return '\n'.join(f'{k.upper()}: {sections[k]}' for k in ('findings', 'impression') if sections.get(k))


def prepare(cfg):
    root, out = Path(cfg['data_root']), Path(cfg['cache'])
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise FileExistsError('Dataset already prepared; select a new cache for a different cohort.')
    audit = Audit()
    studies = load_studies(root, 'frontal', audit)
    attach_splits(root, studies, audit)
    attach_labels(root, studies, audit)
    config = BuildConfig(root, out, num_examples=None, split='all', view='frontal',
                         min_gap_hours=cfg['min_gap_hours'], max_gap_days=cfg['max_gap_days'],
                         one_per_patient=False, min_label_flips=0, render_gallery=False)
    candidates = find_candidates(studies.values(), config, audit)
    indices = [LABEL_COLUMNS.index(k) for k in cfg['findings']]
    observations, pairs = {}, {}
    counts = {}
    for split in ('train', 'validate', 'test'):
        subset = [c for c in candidates if c.source.split == split]
        subset.sort(key=lambda c: hashlib.sha256(f'{cfg["seed"]}:{c.source.subject_id}:{c.source.study_id}:{c.target.study_id}'.encode()).hexdigest())
        limit = cfg['max_train_pairs'] if split == 'train' else cfg['max_eval_pairs']
        rows = []
        for c in subset:
            try:
                texts = [report_text(s.report_path) for s in (c.source, c.target)]
            except OSError:
                audit.increment('pilot_missing_report')
                continue
            if not all(texts):
                audit.increment('pilot_missing_findings_and_impression')
                continue
            ids = []
            for study, img, text in zip((c.source, c.target), (c.source_image, c.target_image), texts):
                key = f'{study.subject_id}_{study.study_id}_{img.dicom_id}'
                ids.append(key)
                observations[key] = dict(id=key, patient=study.subject_id, study=study.study_id,
                    split=split, image=str(img.image_path), report=text, view=img.view,
                    timestamp=img.timestamp.isoformat(), labels=[study.labels[i] for i in indices])
            pair_id = hashlib.sha256((':'.join(ids)).encode()).hexdigest()[:24]
            rows.append(dict(id=pair_id, patient=c.source.subject_id, split=split,
                source=ids[0], target=ids[1], horizon=int(np.searchsorted(cfg['horizon_edges_hours'], c.elapsed_hours, side='left')),
                realized_gap_hours=c.elapsed_hours, view=c.matched_view))
            if len(rows) >= limit:
                break
        pairs[split] = rows
        counts[split] = dict(pairs=len(rows), patients=len({r['patient'] for r in rows}),
                            horizon_counts=dict(Counter(r['horizon'] for r in rows)))
        write_rows(out / f'{split}.jsonl', rows)
    patients = [{r['patient'] for r in pairs[s]} for s in pairs]
    assert all(not patients[i] & patients[j] for i in range(3) for j in range(i))
    used = {r[k] for rows in pairs.values() for r in rows for k in ('source', 'target')}
    obs = [observations[k] for k in sorted(used)]
    write_rows(out / 'observations.jsonl', obs)
    atomic_json(out / 'manifest.json', dict(config=cfg, counts=counts, observations=len(obs), audit=audit.as_dict(),
        report_scope='Findings + Impression; unsectioned reports excluded',
        label_source='Official CheXpert CSV for task training only; NOT CheXbert evaluation',
        report_availability='Unknown. Retrospective report-available pilot; 6h minimum acquisition gap is not proof of report availability.',
        selection='Hash-ordered adjacent same-patient pairs, matched frontal view; no target-label selection',
        stage1='Only selected training patients/studies; image-only report and disease supervision',
        horizon='Requested bin derived from observed gap; exact gap never enters model'))
    print(dict(counts=counts, observations=len(obs)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/pilot.json')
    prepare(read_config(p.parse_args().config))
