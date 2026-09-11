"""Read-only cohort/label/time audit; Qwen annotations are descriptive only."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
import sys
from zoneinfo import ZoneInfo

import numpy as np
from common import ROOT, atomic_json, digest, load_rows, read_config
sys.path.insert(0,str(ROOT.parent/'mimic_cxr_iv_linked'))
from vlm_triage import triage

FIELDS = ['primary_class','direction','stable_state','image_assessment',
          'report_assessment','device_change','technical_confound','confidence']


def summarize(rows, labels):
    gaps=np.array([r['realized_gap_hours'] for r in rows])
    observed=Counter()
    qwen=Counter(); routed=Counter(); stable_state=Counter(); flags=Counter(); reasons=Counter()
    cross=Counter(); attributes={k:Counter() for k in FIELDS}
    joint_fields=future_fields=0
    no_future=0
    coverage_by_table=Counter()
    for row in rows:
        audit=row['audit_flags']
        category=('no_comparable_field' if audit['known_joint_finding_count']==0 else
                  'observed_change' if audit['observed_binary_change'] else 'no_observed_change')
        observed[category]+=1
        joint_fields+=audit['known_joint_finding_count']
        future_fields+=audit['known_future_finding_count']
        no_future+=audit['known_future_finding_count']==0
        for table,n in (row.get('current_clinical_availability') or {}).get('history_counts',{}).items():
            if n:coverage_by_table[table]+=1
        record=labels.get(row['pair_id'])
        if record:
            label,review=record['label'],record['routing']
            qwen[label['primary_class']]+=1
            routed[review['screening_class']]+=1
            flags.update(record['flags']);reasons.update(review['reasons'])
            if label['primary_class']=='stable':stable_state[label['stable_state']]+=1
            for field in FIELDS:attributes[field][label[field]]+=1
            cross[category+'/'+label['primary_class']]+=1
    source_views=Counter(row['view'] for row in rows)
    pairs_by_patient=Counter(r['subject_id'] for r in rows)
    return dict(pairs=len(rows),patients=len(pairs_by_patient),admissions=len({r['hadm_id'] for r in rows if r.get('hadm_id')}),
        unique_images=len({r[k] for r in rows for k in ['source_image','target_image']}),
        unique_studies=len({r[k] for r in rows for k in ['source_study','target_study']}),
        split_counts=dict(Counter(r['split'] for r in rows)),views=dict(source_views),
        maximum_pairs_per_patient=max(pairs_by_patient.values()),
        gap_hours=dict(zip(['min','p05','p25','median','p75','p95','max'],np.quantile(gaps,[0,.05,.25,.5,.75,.95,1]).tolist())),
        horizon_counts=dict(Counter(r['horizon_bin'] for r in rows)),
        gap_histogram={'edges':list(range(0,75,3)),'counts':np.histogram(gaps,bins=list(range(0,75,3)))[0].tolist(),
            'scope':'0–72h plot only; longer gaps excluded from histogram, retained in quantiles'},
        official_chexpert=dict(categories=dict(observed),joint_known_fields=joint_fields,
            future_known_fields=future_fields,total_fields=6*len(rows),pairs_without_known_future_field=no_future),
        prior_event_table_coverage=dict(coverage_by_table),
        qwen=dict(completed=sum(qwen.values()),missing=len(rows)-sum(qwen.values()),classes=dict(qwen),
            routing=dict(routed),stable_subtypes_within_primary_stable=dict(stable_state),
            attributes={k:dict(v) for k,v in attributes.items()},review_flags=dict(flags),
            routing_reasons=dict(reasons),official_label_cross_table=dict(cross)))


def audit(cfg,out):
    os.umask(0o077)
    out.mkdir(parents=True,exist_ok=True)
    linked=Path(cfg['linked_run']);cache=Path(cfg['cache'])
    vlm=linked/'vlm_qwen35_9b/full'
    connection=sqlite3.connect('file:'+str((vlm/'results.sqlite').resolve())+'?mode=ro',uri=True)
    labels={}
    for pid,label_text,flag_text in connection.execute('SELECT pair_id,label,flags FROM results'):
        label=json.loads(label_text);flags=json.loads(flag_text)
        labels[pid]=dict(label={k:label[k] for k in FIELDS},flags=flags,routing=triage(label,flags))
    failures=[dict(pair_id=pid,error=error,at=at,max_tokens=tokens) for pid,error,at,tokens in connection.execute(
        'SELECT a.pair_id,a.error,a.at,a.generation_max_tokens FROM attempts a '
        'WHERE a.pair_id NOT IN (SELECT pair_id FROM results) '
        'AND a.id=(SELECT max(b.id) FROM attempts b WHERE b.pair_id=a.pair_id)')]
    connection.close()
    rows=load_rows(linked/'linked_pairs.jsonl')
    tier_names=['same_admission_6h_30d','same_admission_6h_72h',
        'same_admission_6h_72h_image_qc_prior_ehr',cfg['linked_tier']]
    scopes={'original_70318':summarize(rows,labels)}
    for name in tier_names:
        selected=[r for r in rows if r['tiers'].get(name)]
        scopes[name]=summarize(selected,labels)
        for split in ['train','validate','test']:
            scopes[name+'/'+split]=summarize([r for r in selected if r['split']==split],labels)
    used={r['id'] for split in ['train','validate','test'] for r in load_rows(cache/(split+'.jsonl'))}
    selected=[r for r in rows if r['pair_id'] in used]
    assert len(selected)==len(used)
    scopes['actual_all']=summarize(selected,labels)
    for split in ['train','validate','test']:
        scopes['actual_'+split]=summarize([r for r in selected if r['split']==split],labels)
    observations=load_rows(cache/'observations.jsonl')
    obs_lookup={r['id']:r for r in observations}
    ehr_obs={}
    ehr_records={r['id']:r for r in load_rows(cache/'ehr_audit.jsonl')}
    for split in ['train','validate','test']:
        subset=[r for r in observations if r['split']==split]
        inputs=load_rows(cache/(split+'.jsonl'))
        source=[obs_lookup[r['source']] for r in inputs]
        ehr_obs[split]=dict(observations=len(subset),with_ehr_values=sum(r['ehr_record_count']>0 for r in subset),
            source_pairs_with_ehr_values=sum(r['ehr_record_count']>0 for r in source),
            source_pairs=len(source),
            source_pairs_table_coverage={table:sum(any(e['table']==table for e in ehr_records[r['id']]['selected_records']) for r in source)
                for table in ['icu/chartevents','hosp/labevents','hosp/emar']})
    # Direct verification of the official-label classification on the actual subset.
    for split in ['train','validate','test']:
        check=Counter()
        for pair in load_rows(cache/(split+'.jsonl')):
            source,target=[np.array(obs_lookup[pair[k]]['labels']) for k in ['source','target']]
            known=np.isin(source,[0,1]) & np.isin(target,[0,1])
            category='no_comparable_field' if not known.any() else 'observed_change' if (source[known]!=target[known]).any() else 'no_observed_change'
            check[category]+=1
        assert dict(check)==scopes['actual_'+split]['official_chexpert']['categories']
    stats=json.loads((vlm/'summary/statistics.json').read_text())
    assert len(labels)==stats['completed_pairs']
    assert sum(stats['groups']['all']['primary_classes'][k]['n'] for k in ['changed','stable','indeterminate'])==len(labels)
    stage=load_rows(ROOT/'runs/linked_20260909_overnight/ours/metrics.jsonl')
    step_counts=Counter(r['stage'] for r in stage)
    exposure=dict(stage1_optimizer_steps=step_counts[1],stage2_optimizer_steps=step_counts[2],
        stage1_observation_presentations=step_counts[1]*cfg['batch_size']*cfg['gradient_accumulation'],
        stage2_pair_presentations=step_counts[2]*cfg['batch_size']*cfg['gradient_accumulation'],
        unique_training_observations=ehr_obs['train']['observations'],unique_training_pairs=scopes['actual_train']['pairs'],
        sampling='Deterministic no-replacement epoch permutations; all selected observations/pairs seen at least once')
    result=dict(as_of=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        strict_training_pool_with_current_four_pair_cap=sum(min(n,4) for n in Counter(
            r['subject_id'] for r in rows if r['split']=='train' and r['tiers'].get(cfg['linked_tier'])).values()),
        population=json.loads((linked/'summary.json').read_text())['population'],
        inventory={k:v for k,v in json.loads((linked/'pair_funnel.json').read_text()).items() if k in ['image_denominators','study_denominators','alternative_rule_pair_counts']},
        qwen_global={k:v for k,v in stats.items() if k!='groups'},qwen_global_counts=stats['groups']['all'],
        unresolved_failures=failures,scopes=scopes,ehr_serialized=ehr_obs,training_exposure=exposure,
        provenance={str(p):digest(p) for p in [linked/'linked_pairs.jsonl',vlm/'summary/statistics.json',cache/'manifest.json',cache/'train.jsonl',cache/'observations.jsonl']},
        definitions=dict(official_observed_stable='At least one of six official CheXpert findings has known binary labels at both times; no change in those comparable fields. Other fields may be unknown, and severity can still change.',
            qwen_stable='Model judged no meaningful disease-related interval change from the paired images/reports; not adjudicated truth. Stable can contain persistent abnormalities.',
            review='Routing is for manual review, not automatic exclusion or proof of unusable data.',
            training='Qwen labels were not used for training, label balancing or sample selection.',
            population='Image, study, pair and patient counts are different denominators; cohort scopes overlap.'))
    atomic_json(out/'statistics.json',result)
    plot(result,out)
    for name in ['actual_train','actual_validate','actual_test',cfg['linked_tier']+'/train']:
        s=scopes[name]
        print(name,json.dumps({k:s[k] for k in ['pairs','patients','unique_images','gap_hours','horizon_counts','official_chexpert']},ensure_ascii=False),flush=True)
        print('qwen',json.dumps(s['qwen'],ensure_ascii=False),flush=True)
    print('ehr',ehr_obs,'exposure',exposure,flush=True)


def plot(result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=result['scopes']['actual_train'];q=data['qwen'];raw=data['official_chexpert']['categories']
    fig,axes=plt.subplots(1,3,figsize=(15,4.2))
    hist=data['gap_histogram'];edges=np.array(hist['edges'])
    axes[0].bar(edges[:-1],hist['counts'],width=np.diff(edges),align='edge',color='#477ca8',edgecolor='white')
    axes[0].axvline(data['gap_hours']['median'],color='#c86d32',ls='--',label=f"Median {data['gap_hours']['median']:.1f}h")
    axes[0].set(xlabel='Time from current to next CXR (hours)',ylabel='Training pairs',title='Actual training set: 6,000 pairs')
    axes[0].legend(frameon=False)
    labels=['Observed change','No observed change','No comparable field'];counts=[raw.get(k,0) for k in ['observed_change','no_observed_change','no_comparable_field']]
    for ax,title,names,values in [(axes[1],'Original six-finding labels',labels,counts),
        (axes[2],'Qwen raw labels (reference only)',['Changed','Stable','Indeterminate'],[q['classes'].get(k,0) for k in ['changed','stable','indeterminate']])]:
        bars=ax.barh(names,values,color=['#d28444','#477ca8','#aaa'])
        ax.invert_yaxis();ax.set(xlim=(0,6000),xlabel='Training pairs',title=title)
        for bar,value in zip(bars,values):ax.text(value+60,bar.get_y()+bar.get_height()/2,f'{value:,} ({value/60:.2f}%)',va='center',fontsize=9)
    fig.suptitle('Different definitions of change; neither is an adjudicated clinical change label',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(out/'training_data_overview.png',dpi=170);fig.savefig(out/'training_data_overview.pdf');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();audit(read_config(a.config),a.out)
