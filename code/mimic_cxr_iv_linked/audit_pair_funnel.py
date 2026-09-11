"""Explain image/study/pair denominators and quantify alternative pair rules."""
import argparse
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path
from common import read_jsonl, dump_json, timestamp


def run(root):
    images = {r['dicom_id']: {k:r[k] for k in ('timestamp','view','exists')} for r in read_jsonl(root/'images.jsonl')}
    timelines = defaultdict(list)
    for r in read_jsonl(root/'studies.jsonl'):
        timelines[r['subject_id']].append({k:r[k] for k in ('subject_id','study_id','timestamp','latest_timestamp','selected_frontal','split')} | {'valid_report':r['report']['valid']})
    stages = Counter()
    alternatives = Counter()
    reasons = Counter()
    stage_studies = defaultdict(set)
    common_fail = Counter()
    gap_fail = Counter()
    def add(name,a,b):
        stages[name] += 1
        stage_studies[name].update((a['study_id'],b['study_id']))
    for rows in timelines.values():
        rows.sort(key=lambda r:(r['timestamp'],r['study_id']))
        ties = Counter(r['timestamp'] for r in rows)
        for a,b in pairwise(rows):
            add('01_all_adjacent_study_pairs',a,b)
            ordered = (ties[a['timestamp']] == ties[b['timestamp']] == 1 and a['latest_timestamp'] < b['timestamp'])
            valid_split = bool(a['split']) and a['split'] == b['split']
            if not ordered or not valid_split:
                reasons['ambiguous_or_overlapping_time_or_split'] += 1
                continue
            add('02_unambiguous_order_and_split',a,b)
            va,vb = a['selected_frontal'],b['selected_frontal']
            common = next((v for v in ('PA','AP') if v in va and v in vb),None)
            both_frontal = bool(va and vb)
            reports = a['valid_report'] and b['valid_report']
            if both_frontal:
                v1 = common or next(v for v in ('PA','AP') if v in va)
                v2 = common or next(v for v in ('PA','AP') if v in vb)
                gap = (timestamp(images[vb[v2]]['timestamp'])-timestamp(images[va[v1]]['timestamp'])).total_seconds()/3600
                if reports and gap > 0:
                    alternatives['both_frontal_valid_reports_any_positive_gap'] += 1
                    if gap <= 720:
                        alternatives['both_frontal_valid_reports_0h_30d'] += 1
                    if 6 <= gap <= 720:
                        alternatives['both_frontal_valid_reports_6h_30d_allow_ap_pa'] += 1
                if common and reports and gap > 0:
                    alternatives['common_view_valid_reports_any_positive_gap'] += 1
                    if gap <= 720:
                        alternatives['common_view_valid_reports_0h_30d'] += 1
            if not common:
                common_fail['ap_pa_mismatch' if both_frontal else 'one_or_both_no_AP_PA'] += 1
                continue
            add('03_common_AP_or_PA',a,b)
            if not reports:
                reasons['missing_supported_report_sections_after_view_rule'] += 1
                continue
            add('04_both_supported_report_sections',a,b)
            if not 6 <= gap <= 720:
                gap_fail['under_6h' if gap < 6 else 'over_30d'] += 1
                continue
            add('05_gap_6h_30d',a,b)
            if not (images[va[common]]['exists'] and images[vb[common]]['exists']):
                reasons['missing_selected_image'] += 1
                continue
            add('06_selected_images_exist',a,b)
    pairs = list(read_jsonl(root/'pairs.jsonl'))
    assert stages['06_selected_images_exist'] == len(pairs)
    result = dict(
        image_denominators={'all_images':len(images),'AP_PA_images':sum(r['view'] in ('AP','PA') for r in images.values()),
            'image_view_counts':dict(Counter(r['view'] for r in images.values())),
            'unique_images_in_selected_pairs':len({r[k] for r in pairs for k in ('source_image','target_image')})},
        study_denominators={'all_studies':sum(map(len,timelines.values())),'all_patients':len(timelines),
            'patients_one_study':sum(len(r)==1 for r in timelines.values()),
            'patients_multiple_studies':sum(len(r)>1 for r in timelines.values()),
            'studies_with_AP_PA':sum(bool(s['selected_frontal']) for r in timelines.values() for s in r)},
        sequential_pair_funnel={k:{'pairs':v,'unique_studies':len(stage_studies[k])} for k,v in sorted(stages.items())},
        common_view_exclusions=dict(common_fail),other_sequential_exclusions=dict(reasons),gap_exclusions_after_reports_and_view=dict(gap_fail),
        alternative_rule_pair_counts=dict(alternatives),
        interpretation='Rule sensitivity counts; these are not VLM labels or final data-quality exclusions. All original images/reports remain in the inventory. Counts with relaxed rules are adjacent-study pairs, not all combinations.')
    dump_json(root/'pair_funnel.json',result)
    lines=['# 图像、检查与纵向配对的统计口径','',
        '原始图像没有被删除。377,110 张是全库图像数；99,695 张是当前配对两端使用的不同图像数。',
        '每次检查可以有多张图像；同一中间检查可以同时作为上一对的随访和下一对的当前检查。',
        '', '| 依次应用的规则 | 剩余相邻检查对 | 涉及不同检查 |','|---|---:|---:|']
    for k,v in result['sequential_pair_funnel'].items():
        lines.append(f'| {k} | {v["pairs"]:,} | {v["unique_studies"]:,} |')
    lines += ['', '## 放宽规则的敏感性计数','', '| 规则 | 对数 |','|---|---:|']
    lines += [f'| {k} | {v:,} |' for k,v in sorted(alternatives.items())]
    lines += ['', '这些是配对设计的限制，不是模型判定图像不合格。AP/PA 不一致、短间隔或长间隔可单独分层；无标准报告标题不等于原始报告缺失。当前不把规则外数据自动称作坏数据。','', '详细排除原因及图像视角数见 pair_funnel.json。']
    (root/'pair_funnel.md').write_text('\n'.join(lines)+'\n')
    print(__import__('json').dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    run(p.parse_args().run)
