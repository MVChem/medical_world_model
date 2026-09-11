"""Descriptive whole-cohort/partial coverage statistics for VLM weak labels."""
import argparse
from collections import Counter,defaultdict
import csv
import html
import json
from pathlib import Path
import sqlite3
import numpy as np
from common import read_jsonl,dump_json,dump_jsonl
from vlm_protocol import ENUMS
from vlm_triage import triage,VERSION as TRIAGE_VERSION


def run(manifest_path,results,out):
    out.mkdir(parents=True,exist_ok=True)
    config=json.loads((results/'config.json').read_text())
    manifest={r['pair_id']:r for r in read_jsonl(manifest_path)}
    if config['scope']=='strict': manifest={k:r for k,r in manifest.items() if r['original_strict_subset']}
    if config['train_only']: manifest={k:r for k,r in manifest.items() if r['split']=='train'}
    if config['limit']:
        import hashlib
        keys=sorted(manifest,key=lambda k:hashlib.sha256(('20260909:'+k).encode()).hexdigest())[:config['limit']]
        manifest={k:manifest[k] for k in keys}
    db=sqlite3.connect(f'file:{results/"results.sqlite"}?mode=ro',uri=True)
    records={}
    for pid,label,flags,usage,completed in db.execute('SELECT pair_id,label,flags,usage,completed_at FROM results'):
        assert pid in manifest,'Unexpected output pair'
        records[pid]={'label':json.loads(label),'review_flags':json.loads(flags),'usage':json.loads(usage),'completed_at':completed}
        records[pid]['review_routing']=triage(records[pid]['label'],records[pid]['review_flags'])
    attempts=db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
    db.close()
    assert len(records)<=len(manifest)
    def groups(r):
        yield 'all'
        yield 'split/'+r['split']
        yield 'gap/'+r['horizon_bin']
        yield 'view/'+('matched' if r['same_view'] else 'AP_PA_mismatch')
        yield 'iv_link/'+r['link_status']
        if r['original_strict_subset']: yield 'original_70318_subset'
        for k,v in r['tiers'].items():
            if v: yield 'tier/'+k
    expected=Counter()
    members=defaultdict(list)
    for pid,r in manifest.items():
        for g in groups(r):
            expected[g]+=1
            if pid in records: members[g].append(pid)
    stats={}
    for g,n in sorted(expected.items()):
        ids=members[g]
        hist={k:Counter(records[i]['label'][k] for i in ids) for k in ENUMS}
        primary=hist['primary_class']
        assessable=primary['changed']+primary['stable']
        gaps=[manifest[i]['realized_gap_hours'] for i in ids]
        routed=Counter(records[i]['review_routing']['screening_class'] for i in ids)
        stats[g]={'expected_pairs':n,'completed_pairs':len(ids),'unprocessed_or_failed_pairs':n-len(ids),
            'coverage_percent':100*len(ids)/n,'completed_unique_patients':len({manifest[i]['subject_id'] for i in ids}),
            'primary_classes':{k:{'n':primary[k],'percent_of_completed':100*primary[k]/len(ids) if ids else None,
                'percent_of_expected':100*primary[k]/n} for k in ENUMS['primary_class']},
            'conservative_screening':{k:{'n':routed[k],'percent_of_completed':100*routed[k]/len(ids) if ids else None}
                for k in ('changed','stable','needs_review')},
            'changed_percent_among_assessable':100*primary['changed']/assessable if assessable else None,
            'attribute_counts':{k:dict(v) for k,v in hist.items()},
            'review_flag_counts':dict(Counter(f for i in ids for f in records[i]['review_flags'])),
            'gap_hours_quartiles_completed':np.percentile(gaps,[25,50,75]).tolist() if gaps else None}
    assert sum(v['n'] for v in stats['all']['primary_classes'].values())==len(records)
    summary={'complete':len(records)==len(manifest),'expected_pairs':len(manifest),'completed_pairs':len(records),
        'request_failure_attempts':attempts,'model':config['model_snapshot'],'protocol_sha256':config['protocol_sha256'],
        'interpretation':'Descriptive counts of model-generated weak labels. Not a clinical accuracy evaluation. Partial results use completed-pair denominators and must not be reported as final whole-cohort prevalence. Overlapping tier counts must not be summed. No causal inference.',
        'groups':stats,'review_routing_version':TRIAGE_VERSION,
        'image_report_cross_table':dict(Counter(records[i]['label']['image_assessment']+'/'+records[i]['label']['report_assessment'] for i in records))}
    dump_json(out/'statistics.json',summary)
    with (out/'statistics.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['group','expected_pairs','completed_pairs','label_type','class','count','percent_completed','percent_expected'])
        for g,s in stats.items():
            for k,v in s['primary_classes'].items():
                writer.writerow([g,s['expected_pairs'],s['completed_pairs'],'raw_model',k,v['n'],v['percent_of_completed'],v['percent_of_expected']])
            for k,v in s['conservative_screening'].items():
                writer.writerow([g,s['expected_pairs'],s['completed_pairs'],'conservative_review_routing',k,v['n'],v['percent_of_completed'],100*v['n']/s['expected_pairs']])
    dump_jsonl(out/'pair_labels.jsonl',({'pair_id':pid,'subject_id':manifest[pid]['subject_id'],'split':manifest[pid]['split'],
        'source_study':manifest[pid]['source_study'],'target_study':manifest[pid]['target_study'],
        'original_strict_subset':manifest[pid]['original_strict_subset'],**records[pid]} for pid in sorted(records)))
    lines=['# Qwen3.5-9B 纵向胸片初筛统计','',f'状态：{"全部完成" if summary["complete"] else "部分结果，尚未跑完"}。已完成 {len(records):,} / {len(manifest):,} 对。',
        '', '每对输入两张原始胸片及两份未截断的 Findings/Impression；视觉处理器保持比例缩放，每图最多 1,048,576 像素。',
        '以下是模型弱标签的描述性统计，不是人工真值或模型准确率。“稳定”包括持续异常；“无法判断”独立保留。',
        '', '| 分组 | 已完成／应处理 | 有变化 | 稳定 | 无法判断 |', '|---|---:|---:|---:|---:|']
    show=['all','original_70318_subset','tier/same_admission_6h_72h_image_qc_prior_ehr','view/matched','view/AP_PA_mismatch']
    show += sorted(k for k in stats if k.startswith('gap/'))
    for g in show:
        if g not in stats: continue
        s=stats[g]
        cells=[]
        for k in ENUMS['primary_class']:
            v=s['primary_classes'][k]
            cells.append(f'{v["n"]:,} ({v["percent_of_completed"]:.1f}%)' if v['percent_of_completed'] is not None else '—')
        lines.append(f'| {g} | {s["completed_pairs"]:,} / {s["expected_pairs"]:,} | '+' | '.join(cells)+' |')
    lines+=['','## 保守的初筛分流（不覆盖模型原始输出）','',
        '只有图像、报告与总体分类一致，且无已有复核标记的对才暂归入变化／稳定；其余进入待复核。“待复核”不是判定数据不可用，也不代表临床上一定无法判断。',
        '', '| 分组 | 暂定变化 | 暂定稳定 | 待复核 |','|---|---:|---:|---:|']
    for g in show:
        if g not in stats: continue
        cells=[]
        for k,v in stats[g]['conservative_screening'].items():
            cells.append(f'{v["n"]:,} ({v["percent_of_completed"]:.1f}%)' if v['percent_of_completed'] is not None else '—')
        lines.append('| '+g+' | '+' | '.join(cells)+' |')
    lines+=['','百分比以该组已完成对数为分母。各子组可能重叠，不相加。好转、恶化、混合变化、管线变化、拍摄干扰、图文一致性及 split 详见 statistics.json / statistics.csv。',
        '未处理／失败与模型输出“无法判断”是不同状态；均不算作稳定。image_assessment 与报告同次输入，不是盲法的 image-only 实验。']
    (out/'statistics.md').write_text('\n'.join(lines)+'\n')
    # Fixed hash-order review examples from every primary class and flagged cases.
    chosen=[]
    for label in ENUMS['primary_class']:
        chosen += sorted(i for i in records if records[i]['label']['primary_class']==label)[:12]
    chosen += sorted(i for i in records if records[i]['review_flags'])[:12]
    example_subjects={'12137189','11137177','16254738','16562665'}
    chosen += sorted(i for i in records if manifest[i]['subject_id'] in example_subjects)
    chosen=list(dict.fromkeys(chosen))
    assets=out/'review_assets'
    assets.mkdir(exist_ok=True)
    h=['<!doctype html><meta charset="utf-8"><title>VLM screening review</title><style>body{font:16px system-ui;margin:24px;max-width:1600px}article{border-top:2px solid #777;margin-top:30px;padding-top:15px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}img{width:100%;height:600px;object-fit:contain;background:#000}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>',
       f'<h1>Qwen3.5-9B 初筛复核</h1><p>{len(records):,}/{len(manifest):,} 对已完成。原图 + 原报告 + 模型弱标签；按类别选例，不代表随机样本。</p>']
    for pid in chosen:
        r=manifest[pid]
        h.append('<article><h2>'+html.escape(pid)+' · '+html.escape(records[pid]['label']['primary_class'])+'</h2>')
        h.append('<p>'+html.escape(f"{r['source_view']} → {r['target_view']}, {r['realized_gap_hours']:.2f} h; {r['link_status']}")+'</p><div class="pair">')
        for side in ('source','target'):
            link=assets/(r[side+'_image']+'.jpg')
            if not link.exists() and not link.is_symlink(): link.symlink_to(r[side+'_path'])
            h.append('<div><img src="review_assets/'+link.name+'"><pre>'+html.escape(r[side+'_report'])+'</pre></div>')
        h.append('</div><pre>'+html.escape(json.dumps(records[pid],ensure_ascii=False,indent=2))+'</pre></article>')
    (out/'review.html').write_text('\n'.join(h))
    print(json.dumps({'complete':summary['complete'],'expected':len(manifest),'completed':len(records),'classes':stats['all']['primary_classes']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    run(a.manifest,a.results,a.out)
