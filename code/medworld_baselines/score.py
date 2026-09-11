"""Score completed task panels with real clinical extractors; preserve unavailable cells."""
import argparse
import os
import sys
import time
from base import *
from stats import probability_metrics,qa_metrics

def main(args):
    os.umask(0o077)
    run=args.run.resolve();out=run/args.model/args.split
    records=rows(out/'responses.jsonl')
    done={r['key']:r for r in records if r.get('ok')}
    result=read(out/'metrics.json') if (out/'metrics.json').exists() else dict(model=args.model,split=args.split)
    result.update(updated=time.time(),protocol_sha256=digest(run/'protocol.json'),errors=sum(not r.get('ok') for r in records))
    def flush():atomic(out/'metrics.json',result)
    def completed(task,expected,findings=None):
        selected=[]
        for r in expected:
            if findings:
                keys=[task+'|'+r['id']+'|'+f for f in findings]
                if not all(k in done for k in keys):return None
                selected.append([done[k]['probability'] for k in keys])
            else:
                k=task+'|'+r['id']+'|'
                if k not in done:return None
                selected.append(done[k])
        return selected
    refs1=rows(run/'cohort'/f'table1_references_{args.split}.jsonl')
    refs2={r['id']:r for r in rows(run/'cohort'/f'table2_references_{args.split}.jsonl')}
    inputs2=rows(run/'cohort'/f'table2_inputs_{args.split}.jsonl')
    qa=rows(run/'cohort'/f'qa_references_{args.split}.jsonl')
    cls2=[r for r in inputs2 if r['classification']]
    rep2=[r for r in inputs2 if r['report_generation']]
    p1=completed('table1_prob',refs1,FUTURE_FINDINGS)
    p2=completed('table2_prob',cls2,CURRENT_FINDINGS)
    g1=completed('table1_report',refs1)
    g2=completed('table2_report',rep2)
    gq=completed('qa',qa)
    if p2 is not None:
        result['table2_classification']=probability_metrics([refs2[r['id']]['labels'] for r in cls2],p2,CURRENT_FINDINGS)
        write_rows(out/'classification_predictions.jsonl',[dict(id=r['id'],scores=p,labels=refs2[r['id']]['labels']) for r,p in zip(cls2,p2)])
    if gq is not None:
        qrecords=[dict(**q,text=g['text']) for q,g in zip(qa,gq)]
        result['derived_qa']=qa_metrics(qrecords,read(run/'vocabulary.json'))
        result['derived_qa']['groups']={name:qa_metrics([r for r in qrecords if r['subset']==name],read(run/'vocabulary.json')) for name in ['whole','region']}
        write_rows(out/'qa_predictions.jsonl',qrecords)
    flush()
    if args.quick:return
    # Transformers 4 is required by the existing RadGraph release; isolate it before import.
    sys.path.insert(0,str(T1/'metric_vendor'))
    helper=Path(__file__).parent/'legacy_table1'
    sys.path.insert(0,str(helper if helper.exists() else T1))
    from clinical import CheXbert
    from metrics import clinical_metrics
    import numpy as np
    import torch
    cache=run/'reference_labels'
    cache.mkdir(exist_ok=True)
    refpath=cache/f'{args.split}.json'
    if refpath.exists():
        labels=read(refpath)
        if labels['protocol_sha256']!=digest(run/'protocol.json'):raise ValueError('Reference label cache protocol differs')
    else:
        extractor=CheXbert(device=args.device)
        labels=dict(protocol_sha256=digest(run/'protocol.json'),provenance=extractor.provenance,
            current=extractor.labels([r['current_report'] for r in refs1],FUTURE_FINDINGS),
            future=extractor.labels([r['target_report'] for r in refs1],FUTURE_FINDINGS),
            report2=extractor.labels([refs2[r['id']]['report'] for r in rep2],CURRENT_FINDINGS))
        atomic(refpath,labels);del extractor;torch.cuda.empty_cache()
    result.setdefault('table1',{})
    if p1 is not None:
        result['table1'].update(probability_metrics(labels['future'],p1,FUTURE_FINDINGS))
        result['table1']['probability_threshold_transition']=clinical_metrics(labels['current'],labels['future'],(np.asarray(p1)>=.5).astype(int),p1,FUTURE_FINDINGS)['transition_f1']
        write_rows(out/'forecast_probabilities.jsonl',[dict(id=r['id'],scores=p,target=y) for r,p,y in zip(refs1,p1,labels['future'])])
    for name,generated,references,findings,truth in [
        ('table1',g1,[r['target_report'] for r in refs1],FUTURE_FINDINGS,labels['future']),
        ('table2_report',g2,[refs2[r['id']]['report'] for r in rep2],CURRENT_FINDINGS,labels['report2'])]:
        if generated is None:continue
        section=result.setdefault(name,{})
        hypotheses=[g['text'] for g in generated]
        predpath=out/(name+'_chexbert.json')
        predhash=hashlib.sha256(json.dumps(hypotheses).encode()).hexdigest()
        if predpath.exists() and read(predpath)['predictions_sha256']==predhash:
            predicted=read(predpath)['labels']
        else:
            extractor=CheXbert(device=args.device);predicted=extractor.labels(hypotheses,findings)
            atomic(predpath,dict(labels=predicted,provenance=extractor.provenance,predictions_sha256=predhash))
            del extractor;torch.cuda.empty_cache()
        clinical=clinical_metrics(labels['current'] if name=='table1' else truth,truth,predicted,None,findings)
        section.update(report_n=len(generated),chexbert_f1=clinical['chexbert_f1'],clinical_details=clinical,
            empty_reports=sum(not t.strip() for t in hypotheses),truncated_reports=sum(g['finish_reason']=='length' for g in generated),
            unique_reports=len(set(hypotheses)))
        if name=='table1':section['transition_f1']=clinical['transition_f1']
        flush()
        if not args.skip_radgraph and section.get('radgraph_f1') is None:
            try:
                from radgraph import F1RadGraph
                scorer=F1RadGraph(reward_level='all',model_type='radgraph-xl',cuda=0 if args.device=='cuda' else -1,
                    model_cache_dir=str(T1/'weights/radgraph'))
                # Bounded chunks preserve per-report scores and permit durable resume.
                rgpath=out/(name+'_radgraph.json')
                rg=read(rgpath) if rgpath.exists() else dict(predictions_sha256=predhash,per_report=[])
                if rg['predictions_sha256']!=predhash:raise ValueError('RadGraph predictions changed')
                for start in range(len(rg['per_report']),len(hypotheses),24):
                    reward,rewards,_,_=scorer(hyps=hypotheses[start:start+24],refs=references[start:start+24])
                    partial=rewards[1]
                    if len(partial)!=len(hypotheses[start:start+24]):raise ValueError('RadGraph per-report count differs')
                    rg['per_report'].extend(float(v) for v in partial);atomic(rgpath,rg)
                section.update(radgraph_f1=float(np.mean(rg['per_report'])),radgraph_status='radgraph-xl partial F1, official radgraph 0.1.18')
                del scorer;torch.cuda.empty_cache()
            except Exception as e:
                section.update(radgraph_f1=None,radgraph_status=f'{type(e).__name__}: {e}')
            flush()
    result['table1'].update(direction_f1=None,direction_status='Reference direction labels not adjudicated')
    result['updated']=time.time();flush()
    print(args.model,{k:{a:b for a,b in v.items() if a in ['ap','auroc','brier','ece','transition_f1','radgraph_f1','chexbert_f1','micro_f1','exact_match']} for k,v in result.items() if isinstance(v,dict)},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--model',required=True)
    p.add_argument('--split',choices=['test','validate'],default='test');p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--quick',action='store_true');p.add_argument('--skip-radgraph',action='store_true');main(p.parse_args())
