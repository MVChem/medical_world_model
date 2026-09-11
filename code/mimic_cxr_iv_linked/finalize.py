"""Materialize versioned cohorts, input/target contracts, counts and review cases."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import os
from pathlib import Path

from common import *


def main(args):
    os.umask(0o077)
    out=args.out
    required=['inventory_summary.json','iv_summary.json','image_qc_summary.json','availability_summary.json']
    for filename in required:
        if not (out/filename).exists():
            raise FileNotFoundError(f'Finish prerequisite: {filename}')
    inventory=json.loads((out/'inventory_summary.json').read_text())
    pairs=list(read_jsonl(out/'pairs.jsonl'))
    selected={r[k] for r in pairs for k in ['source_image','target_image']}
    images={r['dicom_id']:r for r in read_jsonl(out/'images.jsonl') if r['dicom_id'] in selected}
    needed={(r['subject_id'],r[k]) for r in pairs for k in ['source_study','target_study']}
    studies={(r['subject_id'],r['study_id']):r for r in read_jsonl(out/'studies.jsonl') if (r['subject_id'],r['study_id']) in needed}
    qc={r['dicom_id']:r for r in read_jsonl(out/'image_qc.jsonl') if r['dicom_id'] in selected}
    clinical={r['dicom_id']:r for r in read_jsonl(out/'clinical_availability.jsonl')}
    assert set(qc)==selected,'Incomplete endpoint image QC'
    duplicate_groups=defaultdict(list)
    for im,r in qc.items():
        if r.get('valid'):
            duplicate_groups[r['pixel_sha256']].append(im)
    cross_patient,cross_split=set(),set()
    duplicate_rows=[]
    for digest,ims in duplicate_groups.items():
        if len(ims)<2:
            continue
        subjects={images[i]['subject_id'] for i in ims}
        splits={images[i]['split'] for i in ims}
        if len(subjects)>1:
            cross_patient.update(ims)
        if len(splits)>1:
            cross_split.update(ims)
        duplicate_rows.append(dict(pixel_sha256=digest,dicom_ids=ims,subjects=len(subjects),splits=sorted(splits)))
    dump_jsonl(out/'exact_duplicate_groups.jsonl',duplicate_rows)
    flags=Counter()
    for r in pairs:
        a,b=[qc[r[k]] for k in ['source_image','target_image']]
        valid=a['valid'] and b['valid']
        exact=bool(valid and a['pixel_sha256']==b['pixel_sha256'])
        distance=(int(a['dhash64'],16)^int(b['dhash64'],16)).bit_count() if valid else None
        dubious=bool({r['source_image'],r['target_image']} & cross_patient)
        overlap=bool({r['source_image'],r['target_image']} & cross_split)
        clean=valid and not exact and not dubious and not (a.get('almost_constant') or b.get('almost_constant'))
        r['image_qc']=dict(both_decodable=valid,identical_decoded_pixels=exact,dhash_hamming=distance,
            near_duplicate_review=distance is not None and distance<=2,cross_patient_exact_duplicate=dubious,
            cross_split_exact_duplicate=overlap,passed_automatic_image_qc=clean)
        r['current_clinical_availability']=clinical.get(r['source_image'])
        pre=bool(r['current_clinical_availability'] and r['current_clinical_availability']['has_recorded_clinical_history'])
        labs_vitals=bool(r['current_clinical_availability'] and r['current_clinical_availability']['has_labs_and_icu_chart'])
        short=r['tiers']['same_admission_6h_72h']
        strict=r['tiers']['same_admission_6h_72h_acquisition_matched']
        r['tiers'].update(same_admission_6h_72h_image_qc=short and clean,
            same_admission_6h_72h_image_qc_prior_ehr=short and clean and pre,
            same_admission_6h_72h_acquisition_matched_image_qc=strict and clean,
            same_admission_6h_72h_acquisition_matched_image_qc_prior_ehr=strict and clean and pre,
            same_admission_6h_72h_image_qc_prior_labs_icu_chart=short and clean and labs_vitals)
        flags.update(k for k,v in r['image_qc'].items() if v is True)
    dump_jsonl(out/'linked_pairs.jsonl',pairs)
    tier_names=list(pairs[0]['tiers'])
    counts={}
    for tier in tier_names:
        folder=out/'cohorts'/tier
        folder.mkdir(parents=True,exist_ok=True)
        group=[r for r in pairs if r['tiers'][tier]]
        counts[tier]={split:group_summary(r for r in group if split=='all' or r['split']==split) for split in ['all','train','validate','test']}
        for split in ['train','validate','test']:
            dump_jsonl(folder/(split+'.jsonl'),(r for r in group if r['split']==split))
    # Full link database is separate from the narrowly whitelisted forecast views.
    primary='same_admission_6h_72h_image_qc_prior_ehr'
    rows=[r for r in pairs if r['tiers'][primary]]
    contracts=out/'forecast_views'
    contracts.mkdir(exist_ok=True)
    for split in ['train','validate','test']:
        current,image_only,targets=[],[],[]
        for r in rows:
            if r['split']!=split:
                continue
            a=images[r['source_image']]
            b=images[r['target_image']]
            sa=studies[(r['subject_id'],r['source_study'])]
            sb=studies[(r['subject_id'],r['target_study'])]
            context=r['current_clinical_availability']
            base=dict(pair_id=r['pair_id'],current_image=a['path'],current_cutoff=r['source_time'],
                horizon_bin=r['horizon_bin'],clinical_history_counts=context['history_counts'],
                clinical_query=dict(subject_id=r['subject_id'],hadm_id=r['hadm_id'],cutoff=r['source_time'],
                    policy='event_time AND storetime <= cutoff; use explicit table-field allowlists'),
                contract='Current evidence only; identifiers are join metadata, not model tokens.')
            image_only.append(dict(**base,current_report=None,report_policy='withheld: availability unknown'))
            current.append(dict(**base,current_report=sa['report']['text'],
                report_policy='retrospective_report_available_assumption; not a verified prospective input'))
            targets.append(dict(pair_id=r['pair_id'],future_image=b['path'],future_report=sb['report']['text'],
                future_labels=sb['labels'],future_timestamp=r['target_time'],realized_gap_hours=r['realized_gap_hours']))
        dump_jsonl(contracts/(split+'_inputs_image_ehr.jsonl'),image_only)
        dump_jsonl(contracts/(split+'_inputs_report_assumed.jsonl'),current)
        dump_jsonl(contracts/(split+'_targets.jsonl'),targets)
    anchors=json.loads((ROOT.parent/'MIMIC_example/representative_pairs_10.json').read_text())['pairs']
    lookup={(r['subject_id'],r['source_study'],r['target_study']):r for r in pairs}
    regression=[]
    for a in anchors:
        key=(a['subject_id'],a['source_study_id'],a['target_study_id'])
        r=lookup.get(key)
        regression.append(dict(**a,found_in_6h_30d_pairs=bool(r),pair_id=r['pair_id'] if r else None,
            link_status=r['link_status'] if r else None,tiers=r['tiers'] if r else None,
            absence_note=None if r else 'Outside configured window or other eligibility condition; see rejected_adjacent_pairs.jsonl'))
    dump_json(out/'example_regression.json',regression)
    review=[]
    for a in regression:
        if a['pair_id']:
            r=lookup[(a['subject_id'],a['source_study_id'],a['target_study_id'])]
            review.append((r,'existing_example_'+a['category']))
    # Additional deterministic structural examples; no future-label filtering.
    cases=[('new_short_linked',lambda r:r['tiers'][primary]),
        ('different_admissions',lambda r:r['link_status']=='different_admissions'),
        ('unmatched',lambda r:r['link_status'] in ['one_unlinked','neither_linked']),
        ('acquisition_uncertain',lambda r:r['tiers']['same_admission_6h_72h'] and not r['acquisition_matched_known'])]
    present={r['pair_id'] for r,_ in review}
    for name,rule in cases:
        candidates=sorted((r for r in pairs if rule(r) and r['pair_id'] not in present),key=lambda r:hashlib.sha256(('42:'+r['pair_id']).encode()).hexdigest())
        for r in candidates[:2]:
            review.append((r,name));present.add(r['pair_id'])
    dump_jsonl(out/'review_cases.jsonl',(dict(pair=r,selection=why,
        source=dict(image=images[r['source_image']],study=studies[(r['subject_id'],r['source_study'])]),
        target=dict(image=images[r['target_image']],study=studies[(r['subject_id'],r['target_study'])])) for r,why in review))
    summary=dict(version=VERSION,population=inventory['population'],tiers=counts,automatic_image_qc=dict(flags),
        primary_candidate=primary,primary_candidate_definition='same admission including preceding ED registration; 6-72h; decodable nonconstant endpoint images; no exact within-pair or cross-patient pixel duplicate; >=1 timestamp-qualified clinical record before current CXR',
        exact_duplicate_groups=len(duplicate_rows),cross_patient_duplicate_images=len(cross_patient),cross_split_duplicate_images=len(cross_split),
        clinical_tables=json.loads((out/'iv_summary.json').read_text()),
        label_coverage={tier:dict(pairs_with_any_known_future_label=sum(r['audit_flags']['known_future_finding_count']>0 for r in pairs if r['tiers'][tier]),
            pairs_with_joint_known_labels=sum(r['audit_flags']['known_joint_finding_count']>0 for r in pairs if r['tiers'][tier])) for tier in [primary,'same_admission_6h_72h']},
        report_available_time_verified_pairs=0,semantic_adjudication_completed=False,
        limitations=['Report content availability timestamps are unavailable; report-conditioned view is retrospective.',
            'Same-study report may rely on multiple images or prior studies; semantic image/report agreement is not adjudicated.',
            'Clinical storetime is a record-availability proxy, not complete record version history.',
            'Near duplicate dHash flags do not establish duplicate acquisitions and do not exclude stable follow-ups.',
            'No disease-change selection; missing/uncertain weak labels stay unknown.',
            'Horizon bins are assigned from observed adjacent gaps; not a prespecified target-window sampling design.'])
    dump_json(out/'summary.json',summary)
    labels={'same_admission_6h_30d':'同住院 6h–30d','same_admission_6h_72h':'同住院 6–72h',
        'same_admission_6h_72h_acquisition_matched':'同住院短期＋采集字段已知一致',
        'same_icu_6h_72h':'同 ICU 6–72h','same_admission_6h_72h_image_qc':'同住院短期＋自动图像质控',
        primary:'同住院短期＋图像质控＋当前前临床记录',
        'same_admission_6h_72h_acquisition_matched_image_qc':'采集一致＋自动图像质控',
        'same_admission_6h_72h_acquisition_matched_image_qc_prior_ehr':'采集一致＋图像质控＋当前前临床记录',
        'same_admission_6h_72h_image_qc_prior_labs_icu_chart':'同住院短期＋图像质控＋当前前化验及 ICU chart'}
    text=['# 全量 MIMIC-CXR + MIMIC-IV 连接与清洗','',
        '完成结构连接、报告规则清洗、候选图像解码/去重检查、临床时间可用性分层；未完成逐例临床语义判读。','',
        '| 条件 | 总配对数 | 患者数 | Train | Validate | Test |','|---|---:|---:|---:|---:|---:|']
    for tier,by in counts.items():
        text.append('| '+labels[tier]+' | '+' | '.join(str(x) for x in [by['all']['pairs'],by['all']['patients'],by['train']['pairs'],by['validate']['pairs'],by['test']['pairs']])+' |')
    text+=['','各行存在包含或交集关系，不能相加。同住院包括该次入院前急诊。',
        '“当前前临床记录”要求原始 subject_id/hadm_id 匹配，事件时间和 storetime 均不晚于当前 CXR；不代表所有临床变量齐全。',
        'ICU chart 包括生命体征、呼吸支持及其他 ICU chart 项，不能直接等同于完整生命体征面板。','',
        f'原始 CXR：{inventory["population"]["cxr_studies"]:,} studies / {inventory["population"]["cxr_images"]:,} images；{inventory["population"]["cxr_patients_in_iv"]:,} 位患者连接到 IV。',
        f'IV 全部 {summary["clinical_tables"]["total_tables"]} 张本地表已扫描；包括临床表的患者筛选与完整字典表。',
        f'像素完全相同分组 {len(duplicate_rows)}；跨患者重复图像 {len(cross_patient)}；跨 split 重复图像 {len(cross_split)}。',
        '', '查看 `review.html` 的原始病例；`example_regression.json` 对照既有 MIMIC_example；`cohorts/` 保存各子集。',
        '`iv/` 是回顾性关联数据库，不能整体拼成当前输入；`forecast_views/` 将当前输入和未来目标分文件保存。',
        '当前报告的可用时间仍未知：`inputs_report_assumed` 使用回顾性报告可用假设；`inputs_image_ehr` 不放入当前报告。','',
        '未按疾病是否变化筛选，稳定病例保留。未用 Qwen 改写报告或伪造临床标签。']
    (out/'summary.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({tier:by['all'] for tier,by in counts.items()},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
