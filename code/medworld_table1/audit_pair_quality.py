"""CPU-only audit of the existing pilot cohort; exports aggregate statistics only.

Does not select a new cohort, change caches, run a model, or train anything.
Admission linkage is retrospective audit context, not a forecasting input.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from metrics import clinical_metrics


ROOT = Path(__file__).resolve().parent


def rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def table(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', newline='') as handle:
        yield from csv.DictReader(handle)


def time(value):
    return datetime.fromisoformat(value) if value else None


def label_summary(current, target):
    current, target = np.asarray(current), np.asarray(target)
    known_target = np.isin(target, [0, 1])
    joint = np.isin(current, [0, 1]) & known_target
    changes = joint & (current != target)
    changed = changes.any(1)
    observed_stable = joint.any(1) & ~changed
    return dict(
        pairs=len(target), changed_pairs=int(changed.sum()),
        no_observed_change_pairs=int(observed_stable.sum()),
        no_joint_known_pairs=int((~joint.any(1)).sum()),
        no_observed_change_among_joint_known=float(observed_stable.sum()/joint.any(1).sum()) if joint.any() else None,
        known_future_fields=int(known_target.sum()), known_transition_fields=int(joint.sum()),
        no_known_future_pairs=int((~known_target.any(1)).sum()),
        total_fields=int(target.size), change_events=int(changes.sum()),
        target_label_counts={str(v): int((target == v).sum()) for v in [-2, -1, 0, 1]},
        joint_known_fields_per_pair={str(k): v for k, v in sorted(Counter(joint.sum(1).tolist()).items())},
    )


def match_ids(timestamp, intervals):
    return {identifier for identifier, start, end in intervals
            if start is not None and end is not None and start <= timestamp <= end}


def link_category(source, target):
    common = source & target
    if len(common) == 1:
        return 'one_common_interval'
    if len(common) > 1:
        return 'ambiguous_multiple_common_intervals'
    if len(source) == len(target) == 1:
        return 'different_intervals_both_uniquely_linked'
    if not source and not target:
        return 'neither_endpoint_linked'
    if not source or not target:
        return 'one_endpoint_unlinked'
    return 'ambiguous_endpoints_no_common_interval'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT / 'data/pilot')
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/pilot_20260908_8h')
    parser.add_argument('--mimic-iv', type=Path, default=Path('/home/data1/data/MIMIC/mimic-iv-3.1'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.run / 'analysis_20260909/pair_quality_audit.json'
    manifest = json.loads((args.cache / 'manifest.json').read_text())
    cohort = {split: rows(args.cache / f'{split}.jsonl') for split in ['train', 'validate', 'test']}
    observations = {r['id']: r for r in rows(args.cache / 'observations.jsonl')}
    subjects = {r['patient'] for r in observations.values()}
    timestamps = {key: time(r['timestamp']) for key, r in observations.items()}
    admissions, extended, icu = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in table(args.mimic_iv / 'hosp/admissions.csv.gz'):
        if row['subject_id'] not in subjects:
            continue
        start, end, ed = time(row['admittime']), time(row['dischtime']), time(row['edregtime'])
        admissions[row['subject_id']].append((row['hadm_id'], start, end))
        extended[row['subject_id']].append((row['hadm_id'], min(t for t in [start, ed] if t is not None), end))
    for row in table(args.mimic_iv / 'icu/icustays.csv.gz'):
        if row['subject_id'] in subjects:
            icu[row['subject_id']].append((row['stay_id'], time(row['intime']), time(row['outtime'])))
    linkage = {
        name: {key: match_ids(timestamps[key], intervals[r['patient']]) for key, r in observations.items()}
        for name, intervals in [('admission_strict', admissions), ('admission_with_ed_registration', extended), ('icu_stay', icu)]
    }
    wanted = {Path(r['image']).stem for r in observations.values()}
    metadata_path = Path(manifest['config']['data_root']) / 'mimic-cxr-2.0.0-metadata.csv'
    metadata = {r['dicom_id']: r for r in table(metadata_path) if r['dicom_id'] in wanted}
    assert len(metadata) == len(wanted), 'Missing metadata for selected images'
    per_obs_metadata = {key: metadata[Path(r['image']).stem] for key, r in observations.items()}
    result = dict(
        scope='Existing pilot only; CPU CSV/JSON audit; no cohort changes, training, or model inference.',
        definitions={
            'labels': 'Six official CheXpert training labels; -2 unmentioned, -1 uncertain, 0 explicitly absent, 1 present. CheXbert evaluation is audited separately.',
            'no_observed_change': 'At least one jointly known binary field, with no binary flip among those fields. Not adjudicated clinical stability.',
            'admission_strict': 'Same subject_id; both selected-image timestamps inclusively inside [admittime, dischtime].',
            'admission_with_ed_registration': 'Same subject_id; both timestamps inside [min(edregtime, admittime), dischtime], matching the separate example linker.',
            'icu_stay': 'Same subject_id; both timestamps inside [intime, outtime].',
            'unlinked': 'No containing interval in these tables; not evidence of an incorrect pair. ED-only/outpatient/coverage limitations possible.',
            'portable': 'Case-insensitive PORTABLE keyword in nonempty PerformedProcedureStepDescription; keyword absence does not prove standard acquisition.',
            'duplicates': 'Duplicate pair IDs/endpoints and shared observations checked; no image-content/perceptual deduplication performed.',
        },
        provenance={
            'cache': str(args.cache.resolve()), 'mimic_iv': str(args.mimic_iv.resolve()),
            'metadata': str(metadata_path.resolve()),
            'input_sha256': {name: hashlib.sha256((args.cache / name).read_bytes()).hexdigest()
                             for name in ['manifest.json', 'train.jsonl', 'validate.jsonl', 'test.jsonl', 'observations.jsonl']},
        },
        splits={},
    )
    patient_sets = [{r['patient'] for r in group} for group in cohort.values()]
    assert all(not patient_sets[i] & patient_sets[j] for i in range(3) for j in range(i))
    result['patient_disjoint_splits_verified'] = True
    for split, pairs in cohort.items():
        counts = Counter(r['patient'] for r in pairs)
        pc = sorted(counts.values(), reverse=True)
        gaps = [r['realized_gap_hours'] for r in pairs]
        sources, targets = {r['source'] for r in pairs}, {r['target'] for r in pairs}
        acquisition = Counter()
        procedures = Counter()
        for pair in pairs:
            source, target = [observations[pair[side]] for side in ['source', 'target']]
            assert source['patient'] == target['patient'] == pair['patient']
            assert source['view'] == target['view'] == pair['view']
            assert source['split'] == target['split'] == split
            elapsed = (timestamps[pair['target']] - timestamps[pair['source']]).total_seconds()/3600
            assert abs(elapsed-pair['realized_gap_hours']) < 1e-8 and 6 <= elapsed <= 720
            assert int(np.searchsorted([24, 72, 168, 720], elapsed, side='left')) == pair['horizon']
            a, b = [per_obs_metadata[pair[side]] for side in ['source', 'target']]
            assert a['ViewPosition'] == b['ViewPosition'] == pair['view']
            for field, prefix in [('PerformedProcedureStepDescription', 'procedure'), ('PatientOrientationCodeSequence_CodeMeaning', 'orientation')]:
                x, y = (a[field] or '').strip().upper(), (b[field] or '').strip().upper()
                acquisition[prefix+'_both_known' if x and y else prefix+'_at_least_one_missing'] += 1
                if x and y and x != y:
                    acquisition[prefix+'_different_when_both_known'] += 1
                if prefix == 'procedure' and x and y:
                    procedures[x+' -> '+y] += 1
                    acquisition['portable_keyword_both_present' if 'PORTABLE' in x and 'PORTABLE' in y
                                else 'portable_keyword_neither_present' if 'PORTABLE' not in x and 'PORTABLE' not in y
                                else 'portable_keyword_status_mismatch'] += 1
            acquisition['same_calendar_day'] += timestamps[pair['source']].date() == timestamps[pair['target']].date()
        current, target = [[observations[r[side]]['labels'] for r in pairs] for side in ['source', 'target']]
        links = {name: dict(Counter(link_category(matches[r['source']], matches[r['target']]) for r in pairs))
                 for name, matches in linkage.items()}
        horizons = {}
        for horizon in range(4):
            group = [r for r in pairs if r['horizon'] == horizon]
            indices = [i for i, r in enumerate(pairs) if r['horizon'] == horizon]
            horizons[str(horizon)] = dict(pairs=len(group),
                admission_with_ed_registration=dict(Counter(link_category(linkage['admission_with_ed_registration'][r['source']], linkage['admission_with_ed_registration'][r['target']]) for r in group)),
                labels=label_summary([current[i] for i in indices], [target[i] for i in indices]))
        result['splits'][split] = dict(
            pairs=len(pairs), patients=len(counts), views=dict(Counter(r['view'] for r in pairs)),
            horizon_counts=dict(Counter(str(r['horizon']) for r in pairs)),
            gap_hours_quantiles={str(q): float(np.quantile(gaps, q)) for q in [0, .25, .5, .75, .9, .99, 1]},
            patient_pairs=dict(maximum=pc[0], median=float(np.median(pc)), p90=float(np.quantile(pc, .9)),
                top_1pct_patients=math.ceil(len(pc)*.01), top_1pct_pair_share=sum(pc[:math.ceil(len(pc)*.01)])/len(pairs),
                top_10pct_patients=math.ceil(len(pc)*.1), top_10pct_pair_share=sum(pc[:math.ceil(len(pc)*.1)])/len(pairs),
                patients_with_multiple_pairs=sum(x > 1 for x in pc)),
            duplication=dict(duplicate_pair_ids=len(pairs)-len({r['id'] for r in pairs}),
                duplicate_endpoint_pairs=len(pairs)-len({(r['source'], r['target']) for r in pairs}),
                observations_used_as_both_source_and_target=len(sources & targets), unique_observations=len(sources | targets)),
            exact_text=dict(identical_source_target_reports=sum(observations[r['source']]['report'] == observations[r['target']]['report'] for r in pairs),
                unique_target_reports=len({observations[r['target']]['report'] for r in pairs})),
            chexpert_training_labels=label_summary(current, target), admission_and_icu_linkage=links,
            acquisition=dict(acquisition), procedure_pairs=dict(procedures), by_horizon=horizons,
        )
        assert sum(links['admission_with_ed_registration'].values()) == len(pairs)
        assert len(pairs) == manifest['counts'][split]['pairs']
    saved_labels = args.run / 'copy/chexbert_labels.json'
    if saved_labels.exists():
        labels = json.loads(saved_labels.read_text())
        predictions = rows(args.run / 'copy/predictions.jsonl')
        assert [r['id'] for r in predictions] == [r['id'] for r in cohort['test']]
        result['test_chexbert_evaluation_labels'] = label_summary(labels['current'], labels['target'])
        disagreements = {}
        for side, key in [('source', 'current'), ('target', 'target')]:
            cx = np.asarray([observations[r[side]]['labels'] for r in cohort['test']])
            cb = np.asarray(labels[key])
            assert cx.shape == cb.shape
            joint = np.isin(cx, [0, 1]) & np.isin(cb, [0, 1])
            disagreements[side] = dict(total_fields=int(cx.size), all_state_disagreements=int((cx != cb).sum()),
                both_extractors_binary_fields=int(joint.sum()), binary_disagreements=int((joint & (cx != cb)).sum()),
                chexpert_only_binary=int((np.isin(cx, [0, 1]) & ~np.isin(cb, [0, 1])).sum()),
                chexbert_only_binary=int((np.isin(cb, [0, 1]) & ~np.isin(cx, [0, 1])).sum()))
        result['test_chexpert_vs_chexbert'] = disagreements
        # Descriptive, post hoc slices of already-saved predictions. These do
        # not define a replacement benchmark and must not select a checkpoint.
        test = cohort['test']
        matches = linkage['admission_with_ed_registration']
        same_episode = np.asarray([link_category(matches[r['source']], matches[r['target']]) == 'one_common_interval' for r in test])
        same_acquisition = []
        for r in test:
            a, b = [per_obs_metadata[r[side]] for side in ['source', 'target']]
            fields = ['PerformedProcedureStepDescription', 'PatientOrientationCodeSequence_CodeMeaning']
            same_acquisition.append(all((a[f] or '').strip() and (a[f] or '').strip().upper() == (b[f] or '').strip().upper() for f in fields))
        short_gap = np.asarray([6 <= r['realized_gap_hours'] <= 72 for r in test])
        masks = {'all': np.ones(len(test), dtype=bool), 'same_admission_with_ed': same_episode,
                 'same_admission_6_to_72h_same_known_procedure_and_orientation': same_episode & short_gap & np.asarray(same_acquisition, dtype=bool)}
        sliced = {'note': 'Post hoc descriptive slices, using existing predictions only. Different case mix and reference support; not a causal test of data cleaning or a formal replacement table.', 'groups': {}}
        for name, mask in masks.items():
            indices = np.flatnonzero(mask)
            sliced['groups'][name] = dict(pairs=int(mask.sum()), patients=len({test[i]['patient'] for i in indices}),
                reference=label_summary(np.asarray(labels['current'])[mask], np.asarray(labels['target'])[mask]), models={})
        for model in ['copy', 'direct', 'matched', 'ours']:
            directory = args.run / model / ('evaluation' if model != 'copy' else '')
            generated = rows(directory / 'predictions.jsonl')
            lab = json.loads((directory / 'chexbert_labels.json').read_text())
            assert [r['id'] for r in generated] == [r['id'] for r in test]
            assert lab['current'] == labels['current'] and lab['target'] == labels['target']
            scores = np.asarray([r['scores'] for r in generated]) if model != 'copy' else None
            for name, mask in masks.items():
                metrics = clinical_metrics(np.asarray(lab['current'])[mask], np.asarray(lab['target'])[mask],
                    np.asarray(lab['predicted'])[mask], scores[mask] if scores is not None else None, manifest['config']['findings'])
                sliced['groups'][name]['models'][model] = metrics
        result['existing_prediction_slices'] = sliced
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({**{s: {k: v for k, v in r.items() if k not in ['procedure_pairs', 'by_horizon']} for s, r in result['splits'].items()},
                      'test_chexpert_vs_chexbert': result.get('test_chexpert_vs_chexbert'), 'output': str(output)}, indent=2))


if __name__ == '__main__':
    main()
