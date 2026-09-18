"""Local original-image/report gallery with timestamp-separated clinical rows."""
import argparse
from collections import defaultdict
from datetime import timedelta
import html
import json
import os
from pathlib import Path

os.environ.setdefault('POLARS_MAX_THREADS','8')
import polars as pl

from availability import EVENT_TABLES,dt
from common import dump_json,read_jsonl,timestamp

SAFE_FIELDS={
    'labs':['labevent_id','itemid','value','valuenum','valueuom','flag'],
    'medication_administration':['emar_id','medication','event_txt'],
    'icu_chart':['itemid','value','valuenum','valueuom'],
    'icu_input':['itemid','rate','rateuom','statusdescription'],
    'icu_procedure':['itemid','location','locationcategory','statusdescription'],
    'icu_output':['itemid','value','valueuom'],
}


def load_context(out,cases):
    """A few traceable recent rows for review, not an unbounded clinical prompt."""
    contexts={c['pair']['pair_id']:dict(before_current={},interval_retrospective={},late_recorded_prior_events={},episode_linkage={}) for c in cases}
    ids={i for c in cases for side in ['source','target'] for i in c[side]['image']['hadm_ids']}
    admission_rows={r['hadm_id']:r for r in read_jsonl(out/'admissions.jsonl') if r['hadm_id'] in ids}
    stay_rows=[r for r in read_jsonl(out/'icustays.jsonl') if r['hadm_id'] in ids]
    transfers=pl.scan_parquet(out/'iv/hosp/transfers.parquet').filter(pl.col('hadm_id').is_in(sorted(ids))).collect().to_dicts()
    for c in cases:
        pair=c['pair'];key=pair['pair_id']
        linked_ids=set(c['source']['image']['hadm_ids']+c['target']['image']['hadm_ids'])
        before,after=timestamp(pair['source_time']),timestamp(pair['target_time'])
        relevant=[]
        for r in transfers:
            start,end=timestamp(r['intime']),timestamp(r['outtime'])
            if r['hadm_id'] in linked_ids and start is not None and start<=after and (end is None or end>=before):
                relevant.append(r)
        contexts[key]['episode_linkage']=dict(role='retrospective_episode_audit; discharge/end fields are not current inputs',
            admissions=[admission_rows[k] for k in sorted(linked_ids) if k in admission_rows],
            icu_stays=[r for r in stay_rows if r['hadm_id'] in linked_ids],interval_transfers=relevant)
    codebooks={}
    for name,file in [('labs','hosp/d_labitems'),('icu','icu/d_items')]:
        codebooks[name]={r['itemid']:r['label'] for r in pl.read_parquet(out/'iv'/(file+'.parquet'),columns=['itemid','label']).iter_rows(named=True)}
    for name,(table,ev_col) in EVENT_TABLES.items():
        fields=['subject_id','hadm_id',ev_col,'storetime','_source_record']+SAFE_FIELDS[name]
        frame=(pl.scan_parquet(out/'iv'/(table+'.parquet')).select(fields)
            .filter(pl.col('hadm_id').is_in(sorted(ids)))
            .with_columns(dt(ev_col).alias('_event'),dt('storetime').alias('_recorded'))
            .collect(engine='streaming'))
        grouped=defaultdict(list)
        for r in frame.iter_rows(named=True):
            grouped[(r['subject_id'],r['hadm_id'])].append(r)
        for c in cases:
            pair=c['pair'];key=pair['pair_id']
            before,after=timestamp(pair['source_time']),timestamp(pair['target_time'])
            source_links=c['source']['image']['hadm_ids']
            # Ambiguous sources deliberately receive no inferred clinical context.
            if len(source_links)!=1:
                continue
            rows=grouped.get((pair['subject_id'],source_links[0]),[])
            past,interval,late=[],[],[]
            for raw in rows:
                event,recorded=raw['_event'],raw['_recorded']
                if event is None or event<before-timedelta(hours=24) or event>after:
                    continue
                value={k:raw[k] for k in fields if k not in ['subject_id','hadm_id']}
                value['source_table']=table
                item=raw.get('itemid')
                if item:
                    value['label']=codebooks['labs' if name=='labs' else 'icu'].get(item)
                if recorded is not None and max(event,recorded)<=before:
                    past.append(value)
                elif event>before:
                    interval.append(value)
                else:
                    late.append(value)
            sort=lambda r:(r[ev_col],r['storetime'],r['_source_record'])
            contexts[key]['before_current'][name]=dict(window='last 24h by occurrence; event/storetime <= cutoff',count=len(past),recent_rows=sorted(past,key=sort)[-6:])
            contexts[key]['interval_retrospective'][name]=dict(window='occurrence in (source,target]; no assumption it was known at source',count=len(interval),recent_rows=sorted(interval,key=sort)[-6:])
            contexts[key]['late_recorded_prior_events'][name]=dict(count=len(late),recent_rows=sorted(late,key=sort)[-3:])
        print(f'[review] linked recent {name} records',flush=True)
    return contexts


def render(args):
    os.umask(0o077)
    cases=list(read_jsonl(args.out/'review_cases.jsonl'))
    contexts=load_context(args.out,cases)
    dump_json(args.out/'review_clinical_context.json',contexts)
    assets=args.out/'review_assets';assets.mkdir(exist_ok=True)
    cards=[]
    for i,c in enumerate(cases,1):
        p=c['pair'];panels=[]
        for side,title in [('source','当前'),('target','真实随访')]:
            obs=c[side];im=obs['image'];s=obs['study']
            link=assets/(im['dicom_id']+'.jpg')
            if not link.exists():link.symlink_to(im['path'])
            fields={'study_id':s['study_id'],'dicom_id':im['dicom_id'],'time':im['timestamp'],'view':im['view'],
                'procedure':im['procedure'],'orientation':im['orientation'],'hadm_ids':im['hadm_ids'],'stay_ids':im['stay_ids'],
                'study_views':s['view_counts'],'report_flags':s['report']['flags']}
            report=s['report']['text']
            panels.append(f'<div class="endpoint"><h3>{title}</h3><a href="review_assets/{im["dicom_id"]}.jpg"><img loading="lazy" src="review_assets/{im["dicom_id"]}.jpg"></a><p class="report">{html.escape(report)}</p><details><summary>图文来源与质量标记</summary><pre>{html.escape(json.dumps(fields,ensure_ascii=False,indent=2))}</pre></details></div>')
        context=contexts[p['pair_id']]
        details=''.join(f'<details><summary>{name}</summary><pre>{html.escape(json.dumps(context[key],ensure_ascii=False,indent=2))}</pre></details>' for key,name in [
            ('episode_linkage','住院 / ICU / 科室时间线（回顾性连接审计）'),
            ('before_current','检查前已记录的临床事件（最近 24h，节选）'),('interval_retrospective','两次检查之间的回顾性事件（不是当前输入）'),
            ('late_recorded_prior_events','发生较早但未能确认在 cutoff 前记录的事件')])
        tiers=', '.join(k for k,v in p['tiers'].items() if v)
        cards.append(f'<section><h2>{i}. {html.escape(c["selection"])} · {p["realized_gap_hours"]:.1f}h · {html.escape(p["link_status"])}</h2><p>patient {p["subject_id"]} / {p["split"]} / pair {p["pair_id"]}</p><p>{html.escape(tiers or "保留在审计库，未进入同住院子集")}</p><div class="pair">{"".join(panels)}</div>{details}</section>')
    summary=json.loads((args.out/'summary.json').read_text())
    count_rows=[]
    for tier,title in [('same_admission_6h_30d','同住院，6h–30d'),('same_admission_6h_72h','同住院，6–72h'),
        ('same_admission_6h_72h_image_qc_prior_ehr','同住院短期＋图像质控＋当前前临床记录'),
        ('same_admission_6h_72h_acquisition_matched_image_qc_prior_ehr','再要求采集字段已知一致')]:
        numbers=summary['tiers'][tier]
        count_rows.append('<tr><td>'+title+'</td>'+''.join('<td>'+str(x)+'</td>' for x in [numbers['all']['pairs'],numbers['all']['patients'],numbers['train']['pairs'],numbers['validate']['pairs'],numbers['test']['pairs']])+'</tr>')
    counts='<section><h2>实际可用规模分层</h2><table><thead><tr><th>条件</th><th>配对</th><th>患者</th><th>Train</th><th>Validate</th><th>Test</th></tr></thead><tbody>'+''.join(count_rows)+'</tbody></table><p>同住院含该次入院前急诊；各行有包含关系，不能相加。临床可用性是时间戳代理，报告可用时间和图文语义尚未逐例审定。</p></section>'
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>MIMIC-CXR + IV 全量连接审阅</title><style>
body{font:16px/1.55 system-ui,sans-serif;max-width:1260px;margin:30px auto;padding:0 20px;color:#183244;background:#f4f7fa}section{background:white;border:1px solid #d9e2eb;border-radius:12px;padding:20px;margin:25px 0}h2{font-size:20px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}img{width:100%;height:390px;object-fit:contain;background:#111}.report{white-space:pre-wrap}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}summary{cursor:pointer;font-weight:600;padding:8px}details{border-top:1px solid #ddd;margin-top:10px}@media(max-width:700px){.pair{grid-template-columns:1fr}}
table{width:100%;border-collapse:collapse}td,th{text-align:left;border-bottom:1px solid #ddd;padding:8px}
</style><h1>MIMIC-CXR + MIMIC-IV 连接样本审阅</h1><p>包含既有 appendix/mimic_atlas 参照及按结构条件确定性抽取的新例子。用于审阅，不用于估计随机噪声率。图像为原始 JPG 文件；报告只作规则分节和空白规范化，没有模型改写。</p><p>临床事件按当前时点前已记录、区间回顾性、延迟记录分别展示。报告可用时间未知；ICD 出院编码等未进入当前事件列表。</p>'''+counts+''.join(cards)+'</html>'
    (args.out/'review.html').write_text(page)
    print(f'[review] {len(cases)} cases: {args.out/"review.html"}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    render(p.parse_args())
