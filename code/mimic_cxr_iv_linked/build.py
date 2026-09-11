"""Inventory ALL CXR images/studies and link each endpoint to IV episodes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import pairwise
import json
import os
from pathlib import Path
import sys
import time

from common import *

sys.path.insert(0, str(ROOT.parent))
from MIMIC_example.build_mimic_transitions import (
    Audit, LABEL_COLUMNS, attach_labels, attach_splits, load_studies, parse_study_datetime,
)


def run(args):
    os.umask(0o077)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    marker = out / 'inventory_summary.json'
    signature = dict(version=VERSION, cxr=str(args.cxr.resolve()), iv=str(args.iv.resolve()),
        min_gap_hours=args.min_gap_hours, max_gap_hours=args.max_gap_hours,
        sources={str(p.relative_to(base)): fingerprint(p) for base, paths in [
            (args.cxr, ['mimic-cxr-2.0.0-metadata.csv', 'mimic-cxr-2.0.0-split.csv', 'mimic-cxr-2.0.0-chexpert.csv']),
            (args.iv, ['hosp/patients.csv.gz', 'hosp/admissions.csv.gz', 'icu/icustays.csv.gz'])]
            for p in [base / x for x in paths]})
    if marker.exists():
        if json.loads(marker.read_text())['signature'] != signature:
            raise ValueError('Existing inventory has different inputs/options; use a new output directory.')
        print('Inventory already complete; reusing verified input signature.', flush=True)
        return
    started = time.monotonic()
    def log(s):
        print(f'[inventory {time.monotonic()-started:.0f}s] {s}', flush=True)
    audit = Audit()
    studies = load_studies(args.cxr, 'frontal', audit)
    attach_splits(args.cxr, studies, audit)
    attach_labels(args.cxr, studies, audit)
    subjects = {s.subject_id for s in studies.values()}
    patients = {r['subject_id']: r for r in csv_rows(args.iv/'hosp/patients.csv.gz') if r['subject_id'] in subjects}
    strict, extended, icu = defaultdict(list), defaultdict(list), defaultdict(list)
    admissions, stays = {}, {}
    for r in csv_rows(args.iv/'hosp/admissions.csv.gz'):
        if r['subject_id'] not in subjects:
            continue
        a, e, d = map(timestamp, [r['admittime'], r['edregtime'], r['dischtime']])
        begins = [x for x in [a, e] if x is not None]
        admissions[r['hadm_id']] = r
        strict[r['subject_id']].append((r['hadm_id'], a, d))
        extended[r['subject_id']].append((r['hadm_id'], min(begins) if begins else None, d))
    for r in csv_rows(args.iv/'icu/icustays.csv.gz'):
        if r['subject_id'] in subjects:
            stays[r['stay_id']] = r
            icu[r['subject_id']].append((r['stay_id'], timestamp(r['intime']), timestamp(r['outtime'])))
    dump_json(out/'cxr_subjects.json', sorted(subjects))
    # Split mappings are formed once; no quadratic patient/study lookup.
    subject_splits = defaultdict(set)
    for s in studies.values():
        if s.split:
            subject_splits[s.subject_id].add(s.split)
    if any(len(v) > 1 for v in subject_splits.values()):
        raise ValueError('Official patient split overlap.')
    dump_jsonl(out/'patients.jsonl', (dict(**r, cxr_split=next(iter(subject_splits[p]), None)) for p, r in sorted(patients.items())))
    dump_jsonl(out/'admissions.jsonl', admissions.values())
    dump_jsonl(out/'icustays.jsonl', stays.values())
    log(f'{len(studies):,} studies; {len(subjects):,} CXR patients; {len(patients):,} found in IV')
    images, by_study, image_audit = {}, defaultdict(list), Counter()
    for line, r in enumerate(csv_rows(args.cxr/'mimic-cxr-2.0.0-metadata.csv'), 1):
        t = parse_study_datetime(r.get('StudyDate'), r.get('StudyTime'))
        p, st, im = r['subject_id'], r['study_id'], r['dicom_id']
        path = args.cxr/'files'/('p'+p[:2])/('p'+p)/('s'+st)/(im+'.jpg')
        a = containing_ids(t, extended[p])
        z = containing_ids(t, strict[p])
        k = containing_ids(t, icu[p])
        s = studies.get((p, st))
        record = dict(dicom_id=im, subject_id=p, study_id=st, timestamp=iso(t),
            split=s.split if s else None, path=str(path.resolve()), exists=path.is_file(),
            view=r.get('ViewPosition', '').strip().upper(), rows=r.get('Rows'), columns=r.get('Columns'),
            procedure=r.get('PerformedProcedureStepDescription') or None,
            orientation=r.get('PatientOrientationCodeSequence_CodeMeaning') or None,
            patient_in_iv=p in patients, hadm_ids=a, strict_hadm_ids=z, stay_ids=k,
            linkage='ambiguous' if len(a)>1 else 'unique' if a else 'unmatched',
            source_table='mimic-cxr-2.0.0-metadata.csv', source_record=line)
        if im in images:
            raise ValueError('Duplicate DICOM identifier in metadata.')
        images[im] = record
        by_study[(p, st)].append(im)
        image_audit[record['linkage']] += 1
        image_audit['missing_image'] += not record['exists']
    dump_jsonl(out/'images.jsonl', images.values())
    log(f'Linked all {len(images):,} image records, including nonfrontal images')
    obs, report_audit = {}, Counter()
    for i, (key, s) in enumerate(sorted(studies.items())):
        report = clean_report(s.report_path.read_text(errors='replace')) if s.report_path.is_file() else clean_report('')
        report['path'] = str(s.report_path.resolve())
        views = Counter(images[k]['view'] for k in by_study[key])
        record = dict(subject_id=s.subject_id, study_id=s.study_id, split=s.split,
            timestamp=iso(s.timestamp), latest_timestamp=iso(s.latest_image_timestamp),
            image_ids=by_study[key], view_counts=dict(views), multiple_images=len(by_study[key])>1,
            report=report, labels=dict(zip(LABEL_COLUMNS, s.labels)) if s.labels is not None else None,
            selected_frontal={v: im.dicom_id for v, im in s.images_by_view.items()},
            patient_in_iv=s.subject_id in patients)
        obs[key] = record
        report_audit['valid'] += report['valid']
        for flag, value in report['flags'].items():
            report_audit[flag] += bool(value)
        if (i+1) % 50000 == 0:
            log(f'Cleaned {i+1:,} study reports')
    dump_jsonl(out/'studies.jsonl', obs.values())
    timelines = defaultdict(list)
    for s in studies.values():
        timelines[s.subject_id].append(s)
    pairs, rejections = [], Counter()
    rejected_file = (out/'rejected_adjacent_pairs.jsonl').open('w')
    for p in sorted(timelines):
        ordered = sorted(timelines[p], key=lambda x: (x.timestamp, x.study_id))
        ties = Counter(s.timestamp for s in ordered)
        for a, b in pairwise(ordered):
            reasons = []
            if ties[a.timestamp] > 1 or ties[b.timestamp] > 1:
                reasons.append('ambiguous_study_timestamp')
            if a.latest_image_timestamp >= b.timestamp:
                reasons.append('overlapping_acquisition_windows')
            if not a.split or a.split != b.split:
                reasons.append('missing_or_mismatched_split')
            view = next((v for v in ['PA', 'AP'] if v in a.images_by_view and v in b.images_by_view), None)
            if not view:
                reasons.append('no_common_frontal_view')
            for side, s in [('source', a), ('target', b)]:
                if not obs[(p, s.study_id)]['report']['valid']:
                    reasons.append(side+'_report_missing_sections')
            if view:
                ia, ib = [images[s.images_by_view[view].dicom_id] for s in [a, b]]
                gap = (timestamp(ib['timestamp'])-timestamp(ia['timestamp'])).total_seconds()/3600
                if not args.min_gap_hours <= gap <= args.max_gap_hours:
                    reasons.append('gap_outside_window')
                if not ia['exists'] or not ib['exists']:
                    reasons.append('missing_selected_image')
            if reasons:
                rejections.update(reasons)
                rejected_file.write(json.dumps(dict(subject_id=p, source_study=a.study_id, target_study=b.study_id, reasons=reasons))+'\n')
                continue
            status = link_status(ia['hadm_ids'], ib['hadm_ids'])
            same = status == 'same_admission' and p in patients
            hadm = ia['hadm_ids'][0] if same else None
            stay = ia['stay_ids'][0] if same and len(ia['stay_ids']) == len(ib['stay_ids']) == 1 and ia['stay_ids'] == ib['stay_ids'] and stays[ia['stay_ids'][0]]['hadm_id'] == hadm else None
            oid = ':'.join([p, a.study_id, b.study_id, ia['dicom_id'], ib['dicom_id']])
            la, lb = [obs[(p, s.study_id)]['labels'] or {} for s in [a, b]]
            joint = [k for k in FINDINGS if la.get(k) in (0,1) and lb.get(k) in (0,1)]
            short = same and gap <= 72
            acquisition = same_known(ia['procedure'], ib['procedure']) and same_known(ia['orientation'], ib['orientation'])
            pairs.append(dict(pair_id=hashlib.sha256(oid.encode()).hexdigest()[:24], subject_id=p, split=a.split,
                source_study=a.study_id, target_study=b.study_id, source_image=ia['dicom_id'], target_image=ib['dicom_id'],
                source_time=ia['timestamp'], target_time=ib['timestamp'], realized_gap_hours=gap,
                horizon_bin=horizon(gap), horizon_assignment='derived_from_adjacent_observed_gap', view=view,
                patient_in_iv=p in patients, link_status=status, hadm_id=hadm, stay_id=stay,
                same_strict_admission=bool(same and len(ia['strict_hadm_ids']) == len(ib['strict_hadm_ids']) == 1 and ia['strict_hadm_ids'] == ib['strict_hadm_ids']),
                acquisition_matched_known=acquisition,
                tiers=dict(same_admission_6h_30d=same, same_admission_6h_72h=short,
                    same_admission_6h_72h_acquisition_matched=short and acquisition,
                    same_icu_6h_72h=short and stay is not None),
                audit_flags=dict(source_multiple_images=obs[(p,a.study_id)]['multiple_images'],
                    target_multiple_images=obs[(p,b.study_id)]['multiple_images'],
                    same_report=obs[(p,a.study_id)]['report']['text'] == obs[(p,b.study_id)]['report']['text'],
                    missing_chexpert_record=not bool(la and lb),
                    known_joint_finding_count=len(joint), known_future_finding_count=sum(lb.get(k) in (0,1) for k in FINDINGS),
                    observed_binary_change=any(la[k] != lb[k] for k in joint),
                    report_cutoff_verified=False)))
    rejected_file.close()
    dump_jsonl(out/'pairs.jsonl', pairs)
    tier_names = ['same_admission_6h_30d','same_admission_6h_72h','same_admission_6h_72h_acquisition_matched','same_icu_6h_72h']
    summary = dict(signature=signature, sources=dict(cxr=str(args.cxr), iv=str(args.iv)),
        population=dict(cxr_studies=len(studies), cxr_images=len(images), cxr_patients=len(subjects), cxr_patients_in_iv=len(patients)),
        image_linkage=dict(image_audit), report_cleaning=dict(report_audit),
        adjacent_pairs_considered=sum(max(0,len(v)-1) for v in timelines.values()),
        eligible_cxr_pairs=group_summary(pairs), patient_in_iv_pairs=group_summary(r for r in pairs if r['patient_in_iv']),
        pair_link_status=dict(Counter(r['link_status'] for r in pairs)), rejection_reasons_nonexclusive=dict(rejections),
        tiers={name:{split:group_summary(r for r in pairs if r['tiers'][name] and (split=='all' or r['split']==split)) for split in ['all','train','validate','test']} for name in tier_names},
        label_values_not_used_for_selection=True, patient_disjoint_splits_verified=True,
        original_builder_audit=audit.as_dict(), elapsed_seconds=time.monotonic()-started)
    dump_json(marker, summary)
    log(f'Complete: {len(pairs):,} CXR pairs; {summary["tiers"]["same_admission_6h_72h"]["all"]["pairs"]:,} same-admission short pairs')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cxr', type=Path, default=DEFAULT_CXR)
    p.add_argument('--iv', type=Path, default=DEFAULT_IV)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--min-gap-hours', type=float, default=6)
    p.add_argument('--max-gap-hours', type=float, default=720)
    run(p.parse_args())
