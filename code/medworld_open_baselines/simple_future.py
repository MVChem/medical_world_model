"""Training-split finding transition prior and Copy Current table baselines."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--project',type=Path,required=True)
    args=p.parse_args();project=args.project.resolve();run=args.run.resolve()
    os.environ['MEDWORLD_PROJECT']=str(project)
    sys.path.insert(0,str(project/'code/medworld_baselines'))
    from base import atomic, rows, write_rows, digest, FUTURE_FINDINGS
    from stats import probability_metrics
    original=project/'code/medworld_baselines/runs/raw_models_20260911'
    cache=project/'code/medworld_table1/data/linked_20260913_16k'
    run.mkdir(parents=True,exist_ok=True)
    for name in ['protocol.json','vocabulary.json']:
        shutil.copy2(original/name,run/name)
    shutil.copytree(original/'cohort',run/'cohort',dirs_exist_ok=True)
    shutil.copytree(original/'reference_labels',run/'reference_labels',dirs_exist_ok=True)
    obs={r['id']:r for r in rows(cache/'observations.jsonl')}
    train=rows(cache/'train.jsonl');test=rows(cache/'test.jsonl')
    if {r['patient'] for r in train}&{r['patient'] for r in test}:
        raise ValueError('Train/test patient overlap')
    # Laplace alpha=1 is fixed a priori. Unknown source/target labels never count
    # as observed negatives; source-unknown predictions back off to train priors.
    counts=np.ones((4,len(FUTURE_FINDINGS),2,2),dtype=np.float64)
    marginal=np.ones((4,len(FUTURE_FINDINGS),2),dtype=np.float64)
    for pair in train:
        source=np.asarray(obs[pair['source']]['labels'])
        target=np.asarray(obs[pair['target']]['labels'])
        for j,y in enumerate(target):
            if y not in [0,1]:continue
            marginal[pair['horizon'],j,int(y)]+=1
            if source[j] in [0,1]:counts[pair['horizon'],j,int(source[j]),int(y)]+=1
    transition=counts[:,:,:,1]/counts.sum(-1)
    base=marginal[:,:,1]/marginal.sum(-1)
    model_path=run/'finding_prior/training.json'
    atomic(model_path,dict(status='complete',method='Finding-transition prior',n_train=len(train),
        fit='horizon x finding x source label, Laplace alpha=1, target unknowns excluded',
        source_status='prepared source CheXpert labels; privileged structured source-status interface',
        counts=counts.tolist(),marginal_counts=marginal.tolist(),
        train_sha256=digest(cache/'train.jsonl'),observations_sha256=digest(cache/'observations.jsonl'),
        source_sha256=digest(__file__),gpu_hours=0,created=time.time()))
    # This prior's source labels are dataset annotations, not VLM predictions.
    # Retain that distinction in every report; no continuous-score Copy Current.
    references=rows(run/'cohort/table1_references_test.jsonl')
    bypair={r['id']:r for r in test}
    labels=json.loads((run/'reference_labels/test.json').read_text())
    predictions=[];copy=[];probabilities=[]
    for ref in references:
        pair=bypair[ref['id']];source=obs[pair['source']]
        probs=[float(transition[pair['horizon'],j,int(y)] if y in [0,1] else base[pair['horizon'],j])
               for j,y in enumerate(source['labels'])]
        probabilities.append(probs)
        for finding,prob in zip(FUTURE_FINDINGS,probs):
            predictions.append(dict(key='table1_prob|'+pair['id']+'|'+finding,task='table1_prob',
                id=pair['id'],finding=finding,ok=True,probability=prob))
        copy.append(dict(key='table1_report|'+pair['id']+'|',task='table1_report',id=pair['id'],
                         finding=None,ok=True,text=ref['current_report'],finish_reason='stop'))
    write_rows(run/'finding_prior/test/responses.jsonl',predictions)
    write_rows(run/'copy_current/test/responses.jsonl',copy)
    result=probability_metrics(labels['future'],probabilities,FUTURE_FINDINGS)
    atomic(run/'finding_prior/test/metrics.json',dict(model='finding_prior',status='complete',
        table1=result,source_status_interface='prepared structured source labels',n_train=len(train)))
    for method in ['finding_prior','copy_current']:
        atomic(run/method/'inference_finished.json',dict(status='complete',complete=True,updated=time.time()))
    atomic(run/'models.json',[dict(id=m,label=m) for m in ['finding_prior','copy_current']])
    atomic(run/'simple_future_ready.json',dict(status='complete',n_train=len(train),n_test=len(references),
        checkpoint_selection='not applicable',fit_scope='train only; fixed smoothing; no validation/test fitting'))
    print(json.dumps(dict(n_train=len(train),n_test=len(references),prior=result)))

if __name__=='__main__':main()
