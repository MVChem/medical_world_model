"""Versioned original-MIMIC cohort and strictly cutoff-filtered EHR inputs."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('POLARS_MAX_THREADS', '8')
import polars as pl
import numpy as np
from common import atomic_json, digest, load_rows, read_config, write_rows

LABS = ['50813', '50818', '50820', '50821', '50862', '50882', '50902',
        '50912', '50931', '50963', '50971', '50983', '51003', '51006',
        '51221', '51222', '51265', '51301']
CHART = ['220045', '220050', '220051', '220179', '220180', '220210',
         '220277', '220339', '223762', '223834', '223835', '223849', '224685', '226732']


def read_selected(path, key, wanted):
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if r[key] in wanted:
                yield r


def cutoff_rows(observations, events, items):
    """Latest available record per item; both event and recording time required.

    Select by availability, then require occurrence within the preceding 48h.
    Patient and native admission keys are both enforced by the join.
    """
    parsed = events.with_columns(*[
        pl.col(k).str.to_datetime('%Y-%m-%d %H:%M:%S%.f', strict=False).alias('_' + k)
        for k in ('charttime', 'storetime')])
    parsed = parsed.filter(pl.col('_charttime').is_not_null() & pl.col('_storetime').is_not_null())
    parsed = parsed.with_columns(pl.max_horizontal('_charttime', '_storetime').alias('_available'))
    # Deterministic choice for records sharing the exact availability timestamp.
    parsed = parsed.sort('_available', '_charttime', '_source_record')
    left = observations.join(pl.DataFrame({'itemid': items}), how='cross').sort('cutoff')
    joined = left.join_asof(parsed, left_on='cutoff', right_on='_available',
        by=['subject_id', 'hadm_id', 'itemid'], strategy='backward', check_sortedness=False)
    return joined.filter(pl.col('_available').is_not_null() &
        (pl.col('_charttime') >= pl.col('cutoff') - pl.duration(hours=48)))


def extract_ehr(linked, observations, tokenizer, budget):
    points = pl.DataFrame([dict(id=r['id'], subject_id=r['patient'], hadm_id=r['hadm_id'],
        cutoff=datetime.fromisoformat(r['timestamp'])) for r in observations])
    hadms = points['hadm_id'].unique().to_list()
    events_by_obs = defaultdict(lambda: defaultdict(list))
    provenance = {}
    for category, table, dictionary, items in [
        ('ICU observations', 'icu/chartevents', 'icu/d_items', CHART),
        ('Blood laboratory results', 'hosp/labevents', 'hosp/d_labitems', LABS),
    ]:
        path = linked / 'iv' / (table + '.parquet')
        labels = {r['itemid']: r['label'] for r in pl.read_parquet(
            linked / 'iv' / (dictionary + '.parquet'), columns=['itemid', 'label']).to_dicts()}
        frame = (pl.scan_parquet(path).filter(pl.col('hadm_id').is_in(hadms) & pl.col('itemid').is_in(items))
            .select('subject_id', 'hadm_id', 'itemid', 'charttime', 'storetime', 'value', 'valueuom', '_source_record')
            .filter(pl.col('value').is_not_null()).collect(engine='streaming'))
        joined = cutoff_rows(points, frame, items)
        for r in joined.iter_rows(named=True):
            assert r['_charttime'] <= r['cutoff'] and r['_storetime'] <= r['cutoff']
            age = (r['cutoff'] - r['_charttime']).total_seconds() / 3600
            line = f"{labels[r['itemid']]}: {r['value']} {r['valueuom'] or ''} ({age:.1f}h before image)"
            events_by_obs[r['id']][category].append(dict(text=line, table=table,
                source_record=r['_source_record'], itemid=r['itemid'], charttime=r['charttime'],
                storetime=r['storetime'], age_hours=age))
        provenance[table] = dict(path=str(path), metadata_sha256=digest(path.with_suffix('.json')),
            itemids=items, scoped_records=len(frame), selected_records=len(joined))
        print(f'EHR {table}: {len(frame):,} scoped, {len(joined):,} selected', flush=True)
        del frame, joined

    # Keep medication event text verbatim, including held/not-given statuses.
    table = 'hosp/emar'
    path = linked / 'iv' / (table + '.parquet')
    frame = (pl.scan_parquet(path).filter(pl.col('hadm_id').is_in(hadms))
        .select('subject_id', 'hadm_id', 'charttime', 'storetime', 'medication', 'event_txt', '_source_record')
        .filter(pl.col('medication').is_not_null()).collect(engine='streaming'))
    grouped = defaultdict(list)
    for r in frame.iter_rows(named=True):
        try:
            event, recorded = datetime.fromisoformat(r['charttime']), datetime.fromisoformat(r['storetime'])
        except (ValueError, TypeError):
            continue
        grouped[(r['subject_id'], r['hadm_id'])].append((max(event, recorded), event, r))
    for rows in grouped.values():
        rows.sort(key=lambda x: (x[0], x[1], x[2]['_source_record']), reverse=True)
    for obs in observations:
        cutoff = datetime.fromisoformat(obs['timestamp'])
        seen = set()
        for available, event, r in grouped[(obs['patient'], obs['hadm_id'])]:
            if available > cutoff or event < cutoff-timedelta(hours=24) or r['medication'] in seen:
                continue
            seen.add(r['medication'])
            age = (cutoff-event).total_seconds()/3600
            events_by_obs[obs['id']]['Medication records'].append(dict(
                text=f"{r['medication']}: {r['event_txt'] or 'status unspecified'} ({age:.1f}h before image)",
                table=table, source_record=r['_source_record'], charttime=r['charttime'],
                storetime=r['storetime'], age_hours=age))
            if len(seen) == 6:
                break
    provenance[table] = dict(path=str(path), metadata_sha256=digest(path.with_suffix('.json')),
        scoped_records=len(frame), window_hours=24, max_distinct_medications=6)
    print(f'EHR {table}: {len(frame):,} scoped', flush=True)
    audit = []
    coverage = Counter()
    for obs in observations:
        categories = events_by_obs[obs['id']]
        for records in categories.values():
            records.sort(key=lambda r: (r['age_hours'], r['source_record']))
        # Round robin avoids exhausting the context budget on one table.
        ordered = []
        for i in range(max([len(v) for v in categories.values()] or [0])):
            for category in ('ICU observations', 'Blood laboratory results', 'Medication records'):
                if i < len(categories.get(category, [])):
                    ordered.append((category, categories[category][i]))
        lines = ['Recorded history available by the current image time:']
        selected = []
        for category, record in ordered:
            line = category + ' | ' + record['text']
            if len(tokenizer.encode('\n'.join(lines+[line]), add_special_tokens=False)) <= budget:
                lines.append(line)
                selected.append(record)
                coverage[category] += 1
        if not selected:
            lines.append('No selected recent values available; this does not establish normal findings.')
        obs['ehr_text'] = '\n'.join(lines)
        obs['ehr_record_count'] = len(selected)
        audit.append(dict(id=obs['id'], patient=obs['patient'], hadm_id=obs['hadm_id'],
            cutoff=obs['timestamp'], selected_records=selected, candidate_records=len(ordered)))
    return audit, dict(tables=provenance, selected_record_types=dict(coverage),
        observations_with_values=sum(r['ehr_record_count'] > 0 for r in observations),
        serialization='Whole original-value lines; round robin ICU/labs/medication, recent first; no IDs or absolute dates in input',
        cutoff='Native subject_id/hadm_id; occurrence AND storetime <= own image time. 48h labs/chart, 24h medications.',
        limitations='Recording time is an availability proxy; record revision history is unavailable.')


def prepare(cfg):
    os.umask(0o077)
    linked, out = Path(cfg['linked_run']), Path(cfg['cache'])
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists():
        raise FileExistsError('Completed cache exists; use a new version.')
    pairs, original, counts, hashes = {}, {}, {}, {}
    for split in ('train', 'validate', 'test'):
        path = linked/'cohorts'/cfg['linked_tier']/(split+'.jsonl')
        hashes[str(path)] = digest(path)
        rows = load_rows(path)
        rows.sort(key=lambda r: hashlib.sha256(f"{cfg['seed']}:{r['pair_id']}".encode()).hexdigest())
        selected, per_patient = [], Counter()
        for r in rows:
            if split == 'train' and per_patient[r['subject_id']] >= cfg['max_train_pairs_per_patient']:
                continue
            selected.append(r)
            per_patient[r['subject_id']] += 1
            if split == 'train' and len(selected) >= cfg['max_train_pairs']:
                break
        pairs[split] = [dict(id=r['pair_id'], patient=r['subject_id'], split=split,
            source=r['source_image'], target=r['target_image'], hadm_id=r['hadm_id'],
            horizon=int(np.searchsorted(cfg['horizon_edges_hours'], r['realized_gap_hours'], side='left')),
            realized_gap_hours=r['realized_gap_hours'], view=r['view']) for r in selected]
        original.update({r['pair_id']: r for r in selected})
        counts[split] = dict(pairs=len(selected), patients=len(per_patient),
            maximum_pairs_per_patient=max(per_patient.values()), source_pairs=len(rows))
    subjects = [{r['patient'] for r in rows} for rows in pairs.values()]
    assert all(not subjects[i] & subjects[j] for i in range(3) for j in range(i))
    image_ids = {r[k] for rows in pairs.values() for r in rows for k in ('source', 'target')}
    images = {r['dicom_id']: r for r in read_selected(linked/'images.jsonl', 'dicom_id', image_ids)}
    studies = {r['study_id']: r for r in read_selected(linked/'studies.jsonl', 'study_id', {r['study_id'] for r in images.values()})}
    observations = []
    for id in sorted(image_ids):
        image = images[id]
        study = studies[image['study_id']]
        assert len(image['hadm_ids']) == 1 and image['exists'] and study['report']['valid']
        observations.append(dict(id=id, patient=image['subject_id'], study=image['study_id'],
            split=image['split'], image=image['path'], report=study['report']['text'], view=image['view'],
            timestamp=image['timestamp'], hadm_id=image['hadm_ids'][0],
            labels=[study['labels'].get(k, -2) for k in cfg['findings']],
            report_raw_sha256=study['report']['raw_sha256']))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg['qwen'], local_files_only=True)
    audit, ehr = extract_ehr(linked, observations, tokenizer, cfg['ehr_tokens'])
    for split, rows in pairs.items():
        write_rows(out/(split+'.jsonl'), rows)
    write_rows(out/'observations.jsonl', observations)
    write_rows(out/'ehr_audit.jsonl', audit)
    write_rows(out/'selected_pair_audit.jsonl', original.values())
    report_lengths = [len(tokenizer.encode(r['report'], add_special_tokens=False)) for r in observations]
    atomic_json(out/'manifest.json', dict(config=cfg, counts=counts, observations=len(observations),
        cohort_sha256=hashes, ehr=ehr, report_scope='Original cleaned Findings/Impression, no generated text',
        report_token_cap=cfg['report_tokens'], reports_over_token_cap=sum(n>cfg['report_tokens'] for n in report_lengths),
        label_source='Official MIMIC-CXR CheXpert labels only; unknown/uncertain masked',
        qwen_gate_used=False, selection='Deterministic pair hash, train patient cap, no selection on outcomes or Qwen annotations',
        source_target='Only source image/report/EHR and horizon enter prediction; target evidence supplies losses only',
        report_availability='Retrospective report-available assumption; original CXR report publication time unavailable',
        horizon='6–24h / >24–72h bins from adjacent realized gap; no exact gap in model input',
        ehr_audit_sha256=digest(out/'ehr_audit.jsonl')))
    print(json.dumps(dict(counts=counts, observations=len(observations), ehr=ehr), indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    prepare(read_config(p.parse_args().config))
