"""Count full local CXR/IV longitudinal linkage capacity, without training.

Reuse the pilot's chronological, same-view candidate rules without its size cap.
Output aggregate counts only; no patient-level records or clinical event export.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from time import monotonic

from audit_pair_quality import table, time, match_ids

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mimic_atlas.build_mimic_transitions import (
    Audit, BuildConfig, attach_labels, attach_splits, find_candidates,
    load_studies, read_report,
)


class Group:
    def __init__(self):
        self.count = 0
        self.patients = set()
        self.studies = set()
        self.admissions = set()
        self.views = Counter()
        self.horizons = Counter()

    def add(self, pair, admission=None):
        self.count += 1
        self.patients.add(pair.source.subject_id)
        self.studies.update([pair.source.study_id, pair.target.study_id])
        if admission is not None:
            self.admissions.add(admission)
        self.views[pair.matched_view] += 1
        label = '6-24h' if pair.elapsed_hours <= 24 else '>24-72h' if pair.elapsed_hours <= 72 else '>72-168h' if pair.elapsed_hours <= 168 else '>168-720h'
        self.horizons[label] += 1

    def summary(self):
        return dict(pairs=self.count, patients=len(self.patients), studies=len(self.studies),
                    uniquely_linked_common_admissions=len(self.admissions), views=dict(self.views), horizons=dict(self.horizons))


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxr', type=Path, default=Path('/home/data1/data/MIMIC/MIMIC_CXR'))
    parser.add_argument('--iv', type=Path, default=Path('/home/data1/data/MIMIC/mimic-iv-3.1'))
    parser.add_argument('--output', type=Path, default=root / 'runs/pilot_20260908_8h/analysis_20260909/full_mimic_linkage_audit.json')
    args = parser.parse_args()
    started = monotonic()
    def progress(message):
        print(f'{monotonic()-started:.1f}s {message}', flush=True)
    audit = Audit()
    studies = load_studies(args.cxr, 'frontal', audit)
    attach_splits(args.cxr, studies, audit)
    attach_labels(args.cxr, studies, audit)
    progress(f'CXR metadata loaded: {len(studies)} studies')
    config = BuildConfig(args.cxr, args.output.parent, num_examples=None, split='all', view='frontal',
                         min_gap_hours=6, max_gap_days=30, one_per_patient=False, min_label_flips=0, render_gallery=False)
    candidates = find_candidates(studies.values(), config, audit)
    progress(f'Chronological, same-view candidates before report-section checks: {len(candidates)}')
    subjects = {s.subject_id for s in studies.values()}
    iv_subjects = {r['subject_id'] for r in table(args.iv / 'hosp/patients.csv.gz')}
    admissions, extended, icu = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in table(args.iv / 'hosp/admissions.csv.gz'):
        if r['subject_id'] in subjects:
            start, end, ed = time(r['admittime']), time(r['dischtime']), time(r['edregtime'])
            admissions[r['subject_id']].append((r['hadm_id'], start, end))
            extended[r['subject_id']].append((r['hadm_id'], min(t for t in [start, ed] if t is not None), end))
    for r in table(args.iv / 'icu/icustays.csv.gz'):
        if r['subject_id'] in subjects:
            icu[r['subject_id']].append((r['stay_id'], time(r['intime']), time(r['outtime'])))
    progress('MIMIC-IV patient, admission and ICU intervals loaded')
    # Cache only report-validity booleans and interval identifiers, not report text.
    report_valid, image_links = {}, {}
    def valid_report(study):
        key = (study.subject_id, study.study_id)
        if key not in report_valid:
            try:
                sections = read_report(study.report_path, 12000)
                report_valid[key] = bool(sections.get('findings') or sections.get('impression'))
            except OSError:
                report_valid[key] = False
        return report_valid[key]
    def links(study, image):
        if image.dicom_id not in image_links:
            patient = study.subject_id
            image_links[image.dicom_id] = [match_ids(image.timestamp, intervals[patient]) for intervals in [admissions, extended, icu]]
        return image_links[image.dicom_id]
    names = [
        'cxr_candidates_before_report_sections', 'cxr_valid_image_report_pairs',
        'cxr_valid_pairs_patient_in_iv', 'same_strict_admission_6h_to_30d',
        'same_admission_with_ed_6h_to_30d', 'same_admission_with_ed_6h_to_7d',
        'same_admission_with_ed_6h_to_72h',
        'same_admission_with_ed_6h_to_72h_same_known_procedure_orientation',
        'same_icu_6h_to_30d', 'same_icu_6h_to_72h',
        'same_icu_6h_to_72h_same_known_procedure_orientation',
    ]
    groups = {name: {split: Group() for split in ['all', 'train', 'validate', 'test']} for name in names}
    valid_patients_by_split = defaultdict(set)
    link_status, exclusion = Counter(), Counter()
    def add(name, pair, admission=None):
        for split in ['all', pair.source.split]:
            groups[name][split].add(pair, admission)
    for i, pair in enumerate(candidates):
        add('cxr_candidates_before_report_sections', pair)
        if not (valid_report(pair.source) and valid_report(pair.target)):
            exclusion['report_missing_or_no_findings_impression'] += 1
            continue
        add('cxr_valid_image_report_pairs', pair)
        valid_patients_by_split[pair.source.split].add(pair.source.subject_id)
        if pair.source.subject_id not in iv_subjects:
            exclusion['patient_not_in_iv_patients'] += 1
            continue
        add('cxr_valid_pairs_patient_in_iv', pair)
        sl = links(pair.source, pair.source_image)
        tl = links(pair.target, pair.target_image)
        strict_same = len(sl[0]) == len(tl[0]) == 1 and sl[0] == tl[0]
        same = len(sl[1]) == len(tl[1]) == 1 and sl[1] == tl[1]
        same_icu = len(sl[2]) == len(tl[2]) == 1 and sl[2] == tl[2]
        if same:
            status = 'unique_same_admission_with_ed'
        elif len(sl[1]) > 1 or len(tl[1]) > 1:
            status = 'ambiguous_admission_endpoint'
        elif len(sl[1]) == len(tl[1]) == 1:
            status = 'different_admissions_both_unique'
        elif not sl[1] and not tl[1]:
            status = 'neither_endpoint_linked'
        else:
            status = 'one_endpoint_unlinked'
        link_status[status] += 1
        if strict_same:
            add('same_strict_admission_6h_to_30d', pair, next(iter(sl[0])))
        if same:
            hadm = next(iter(sl[1]))
            add('same_admission_with_ed_6h_to_30d', pair, hadm)
            if pair.elapsed_hours <= 168:
                add('same_admission_with_ed_6h_to_7d', pair, hadm)
            if same_icu:
                add('same_icu_6h_to_30d', pair, hadm)
            if pair.elapsed_hours <= 72:
                add('same_admission_with_ed_6h_to_72h', pair, hadm)
                a, b = pair.source_image, pair.target_image
                same_acquisition = all(x and y and x.strip().upper() == y.strip().upper()
                                       for x, y in [(a.procedure, b.procedure), (a.orientation, b.orientation)])
                if same_acquisition:
                    add('same_admission_with_ed_6h_to_72h_same_known_procedure_orientation', pair, hadm)
                if same_icu:
                    add('same_icu_6h_to_72h', pair, hadm)
                    if same_acquisition:
                        add('same_icu_6h_to_72h_same_known_procedure_orientation', pair, hadm)
        if (i+1) % 10000 == 0:
            progress(f'Processed {i+1}/{len(candidates)} pairs')
    sets = list(valid_patients_by_split.values())
    assert all(not sets[i] & sets[j] for i in range(len(sets)) for j in range(i))
    summaries = {name: {split: group.summary() for split, group in splits.items()} for name, splits in groups.items()}
    for name, splits in summaries.items():
        for key in ['pairs', 'patients', 'studies', 'uniquely_linked_common_admissions']:
            assert splits['all'][key] == sum(splits[s][key] for s in ['train', 'validate', 'test']), (name, key)
    assert sum(link_status.values()) == summaries['cxr_valid_pairs_patient_in_iv']['all']['pairs']
    result = dict(
        scope='Full local CXR corpus, no pilot pair cap; aggregate linkage capacity audit only, no training or new clinical-input dataset.',
        definitions=dict(pair='Strictly adjacent full-timeline studies; exact common AP/PA; selected-image gap 6h to 30d; official patient-disjoint splits; both images available; Findings/Impression required on both reports; official label records required, no target-label value filtering.',
            admission='Same patient; both image timestamps uniquely contained in the same [min(edregtime, admittime), dischtime]. Strict-admission row instead starts at admittime.',
            icu='Same-admission rules plus unique identical ICU stay at both acquisition times.',
            acquisition='Nonempty identical procedure description and patient orientation on both images; AP/PA already matched.',
            effective='Structurally eligible image-report-clinical episode linkage, not clinically adjudicated transitions; per-feature pre-current EHR availability, report availability, image near-duplicates and intervention quality are not established.'),
        sources=dict(cxr=str(args.cxr.resolve()), mimic_iv=str(args.iv.resolve()),
                     script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        population=dict(cxr_studies=len(studies), cxr_patients=len(subjects),
            cxr_patients_in_iv_patients=len(subjects & iv_subjects), cxr_patients_not_in_iv=len(subjects-iv_subjects)),
        groups=summaries, admission_link_status_of_valid_pairs=dict(link_status), exclusions_after_candidates=dict(exclusion),
        original_builder_audit=audit.as_dict(), patient_disjoint_splits_verified=True,
        elapsed_seconds=monotonic()-started,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    progress(f'Aggregate results saved to {args.output}')
    print(json.dumps({'population': result['population'], 'groups': {k:v['all'] for k,v in summaries.items()}, 'admission_link_status': dict(link_status)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
