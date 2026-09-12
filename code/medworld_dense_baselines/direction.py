"""Reference-fixed persistent direction evaluation on the CIG gold pair cohort."""
import argparse
import collections
import csv
import datetime as dt
import json
import time
import numpy as np
from common import *

FINDINGS=['atelectasis','cardiomegaly','consolidation','edema','pleural_effusion','pneumothorax']
SCOPES=['left','right','overall']
MAPPING={'atelectasis':'atelectasis','enlarged cardiac silhouette':'cardiomegaly',
    'consolidation':'consolidation','pulmonary edema/hazy opacity':'edema',
    'pleural effusion':'pleural_effusion','pneumothorax':'pneumothorax'}
DIR={'improved':'improved','no change':'stable','worsened':'worsened'}

def fields(r):
    finding=MAPPING.get(r['label_name'])
    if not finding:return None
    region=r['bbox'].lower()
    scope='overall' if finding=='cardiomegaly' else 'left' if region.startswith('left ') else 'right' if region.startswith('right ') else 'overall'
    return finding,scope

def prepare(run):
    out=run/'direction';out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():return
    def read(name):return list(csv.DictReader((GOLD/name).open(),delimiter='\t'))
    comparisons=read('gold_object_comparison_with_coordinates.txt')
    attrs=(read('gold_object_attribute_with_coordinates.txt')+read('gold_baseline_object_attribute_with_coordinate.txt')
           +read('gold_comparison_relations_500pts_500studies2nd.txt'))
    assertions=collections.defaultdict(set)
    for r in attrs:
        key=fields(r)
        if key:assertions[(r['image_id'].removesuffix('.dcm'),)+key].add(r['context'])
    gold=collections.defaultdict(lambda:collections.defaultdict(set));reject=collections.Counter()
    original_pairs={(r['previous_image_id'],r['current_image_id']) for r in comparisons}
    for r in comparisons:
        key=fields(r)
        if not key:reject['outside_six_findings']+=1;continue
        prev,cur=r['previous_image_id'],r['current_image_id']
        if r['comparison'] not in DIR:reject['mixed_or_conflicting_comparison']+=1;continue
        # Both ends must be explicitly positive; improvement can otherwise mean resolution.
        if assertions[(prev,)+key]!={'yes'} or assertions[(cur,)+key]!={'yes'}:
            reject['not_unambiguously_present_both_ends']+=1;continue
        gold[prev,cur][key].add(DIR[r['comparison']])
    wanted={iid for pair in original_pairs for iid in pair};images={}
    with (LINKED/'images.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            if r['dicom_id'] in wanted:images[r['dicom_id']]=r
    wanted_studies={m['study_id'] for m in images.values()};studies={}
    with (LINKED/'studies.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            if r['study_id'] in wanted_studies:studies[r['study_id']]=r
    inputs=[];refs=[]
    for prev,cur in sorted(original_pairs):
        if prev not in images or cur not in images:reject['pair_missing_metadata']+=1;continue
        source,target=images[prev],images[cur]
        if not source['exists'] or not source.get('timestamp') or not target.get('timestamp'):
            reject['missing_source_or_time']+=1;continue
        hours=(dt.datetime.fromisoformat(target['timestamp'])-dt.datetime.fromisoformat(source['timestamp'])).total_seconds()/3600
        if not 6<=hours<=720:reject['outside_6h_30d_horizon']+=1;continue
        report=studies[source['study_id']]['report']
        if not report['valid']:reject['invalid_source_report']+=1;continue
        labels={k:next(iter(v)) for k,v in gold[prev,cur].items() if len(v)==1}
        reject['conflicting_direction_fields']+=sum(len(v)>1 for v in gold[prev,cur].values())
        # Prefer lateral fields, do not count the same bilateral finding again overall.
        for finding in FINDINGS:
            if (finding,'left') in labels or (finding,'right') in labels:labels.pop((finding,'overall'),None)
        if not labels:reject['pair_without_persistent_direction_target']+=1;continue
        pid=source['subject_id'];assert pid==target['subject_id']
        pairid=prev+'__'+cur
        inputs.append(dict(id=pairid,subject_id=pid,source_image=source['path'],source_report=report['text'],
            horizon_hours=round(hours,4)))
        refs.append(dict(id=pairid,subject_id=pid,source_id=prev,target_id=cur,
            labels=[dict(finding=f,scope=s,direction=d) for (f,s),d in sorted(labels.items())]))
    write_rows(out/'inputs.jsonl',inputs);write_rows(out/'references.jsonl',refs)
    counts=collections.Counter((l['finding'],l['scope'],l['direction']) for r in refs for l in r['labels'])
    atomic(out/'manifest.json',dict(original_gold_pairs=len(original_pairs),eligible_pairs=len(inputs),
        fields=sum(counts.values()),patients=len({r['subject_id'] for r in inputs}),
        coverage={'/'.join(k):v for k,v in counts.items()},rejected=dict(reject),
        inputs_sha256=digest(out/'inputs.jsonl'),references_sha256=digest(out/'references.jsonl'),
        horizons_hours=dict(min=min((r['horizon_hours'] for r in inputs),default=None),
                            max=max((r['horizon_hours'] for r in inputs),default=None)),
        scope='future direction forecast; source image/report and requested horizon only, no target image/report/EHR',
        cohort='separate from 297-pair forecast cohort; same eligible gold subset for all six models',
        metric='macro F1 across supported finding/scope/direction classes; fixed persistent-positive reference mask; invalid/missing predictions count as FN',
        sources={str(GOLD/f):digest(GOLD/f) for f in ['gold_object_comparison_with_coordinates.txt',
            'gold_object_attribute_with_coordinates.txt','gold_baseline_object_attribute_with_coordinate.txt',
            'gold_comparison_relations_500pts_500studies2nd.txt']}))
    print('direction',len(inputs),'pairs',sum(counts.values()),'fields',flush=True)

def parse(text):
    # Exact output schema. Never silently remap missing answers to stable.
    try:
        value=json.loads(text.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip())
        if isinstance(value,dict):value=value.get('directions')
        if not isinstance(value,list) or len(value)!=6:return None
        if any(not isinstance(r,list) or len(r)!=3 for r in value):return None
        allowed={'improved','stable','worsened','absent','unknown'}
        if any(v not in allowed for row in value for v in row):return None
        return value
    except (ValueError,TypeError):return None

def score(run,mid):
    refs=read_rows(run/'direction/references.jsonl')
    responses=read_rows(run/mid/'direction_responses.jsonl')
    preds={r['id']:parse(r['text']) for r in responses}
    supported={(l['finding'],l['scope'],l['direction']) for r in refs for l in r['labels']}
    counts={k:[0,0,0] for k in supported};invalid=0;missing=0
    for r in refs:
        pp=preds.get(r['id']);invalid+=pp is None;missing+=r['id'] not in preds
        for label in r['labels']:
            f,s,d=label['finding'],label['scope'],label['direction']
            pred=pp[FINDINGS.index(f)][SCOPES.index(s)] if pp is not None else 'invalid'
            if pred==d:counts[f,s,d][0]+=1
            else:
                counts[f,s,d][2]+=1
                if (f,s,pred) in supported:counts[f,s,pred][1]+=1
    per={}
    for k,(tp,fp,fn) in sorted(counts.items()):
        per['/'.join(k)]=dict(tp=tp,fp=fp,fn=fn,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    atomic(run/mid/'direction_metrics.json',dict(pairs=len(refs),responses=len(responses),invalid=invalid,missing=missing,
        macro_f1=float(np.mean([v['f1'] for v in per.values()])),per_class=per,
        references_sha256=digest(run/'direction/references.jsonl'),complete=missing==0))

def predict(run,mid,model=None,proc=None):
    import torch
    from PIL import Image
    from features import load_vlm
    spec=load_model_spec(mid)
    if model is None:model,proc=load_vlm(spec)
    out=run/mid;out.mkdir(exist_ok=True)
    path=out/'direction_responses.jsonl'
    seen={r['id'] for r in read_rows(path)} if path.exists() else set()
    rows=read_rows(run/'direction/inputs.jsonl')
    device=model.get_input_embeddings().weight.device
    for i,r in enumerate(rows):
        if r['id'] in seen:continue
        # Report truncation uses one common word boundary, independent of each tokenizer.
        source_report=' '.join(r['source_report'].split()[:220])
        prompt=(f'Forecast the chest radiograph findings {r["horizon_hours"]:.2f} hours after this source examination. '
            'Only the source image and report are supplied. Do not describe an observed follow-up image. '
            'Predict change relative to the source for six findings in this exact row order: '
            +', '.join(FINDINGS)+'. Each row contains [left, right, overall]. '
            'Each entry must be one of improved, stable, worsened, absent, unknown. '
            'Return only a JSON array of six arrays, three strings each, with no explanation. '
            'Source report: '+source_report)
        with Image.open(r['source_image']) as im:
            im=im.convert('RGB');im.thumbnail((512,512),Image.Resampling.BICUBIC)
            messages=[{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':prompt}]}]
            inputs=proc.apply_chat_template(messages,tokenize=True,return_dict=True,return_tensors='pt',
                add_generation_prompt=True,enable_thinking=False).to(device)
        started=time.time()
        with torch.inference_mode():
            output=model.generate(**inputs,max_new_tokens=256,do_sample=False,use_cache=True)
        text=proc.decode(output[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
        with path.open('a') as f:f.write(json.dumps(dict(id=r['id'],text=text,seconds=time.time()-started))+'\n')
        print('direction',mid,i+1,'/',len(rows),flush=True)
    score(run,mid)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','predict','score'])
    p.add_argument('--run',type=Path,default=DEFAULT_RUN);p.add_argument('--model')
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.run)
    elif a.mode=='predict':predict(a.run,a.model)
    else:score(a.run,a.model)
