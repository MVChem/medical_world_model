"""Reference-masked probability metrics and strict set-answer evaluation."""
import json
import math
import re
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

def probability_metrics(labels, probabilities, findings, bins=10):
    y=np.asarray(labels);p=np.asarray(probabilities,dtype=float)
    if y.shape!=p.shape or y.ndim!=2 or y.shape[1]!=len(findings):
        raise ValueError('Probability/label shapes differ')
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Probabilities must be finite and in [0,1]')
    per={}
    for k,name in enumerate(findings):
        mask=np.isin(y[:,k],[0,1]);a=y[mask,k];b=p[mask,k]
        both=bool((a==1).any() and (a==0).any())
        details=dict(n=int(mask.sum()),positives=int((a==1).sum()),negatives=int((a==0).sum()),
            ap=float(average_precision_score(a,b)) if both else None,
            auroc=float(roc_auc_score(a,b)) if both else None,
            brier=float(np.mean((b-a)**2)) if len(a) else None)
        bucket=np.minimum((b*bins).astype(int),bins-1)
        reliability=[]
        for j in range(bins):
            selected=bucket==j
            reliability.append(dict(lower=j/bins,upper=(j+1)/bins,n=int(selected.sum()),
                mean_probability=float(b[selected].mean()) if selected.any() else None,
                positive_fraction=float(a[selected].mean()) if selected.any() else None))
        details['ece']=sum(v['n']/len(a)*abs(v['mean_probability']-v['positive_fraction']) for v in reliability if v['n']) if len(a) else None
        details['bins']=reliability;per[name]=details
    result={name:float(np.mean([d[name] for d in per.values() if d[name] is not None]))
            if any(d[name] is not None for d in per.values()) else None for name in ['ap','auroc','brier','ece']}
    result.update(n=len(y),per_finding=per,rank_supported=[k for k,v in per.items() if v['ap'] is not None],
        calibration_supported=[k for k,v in per.items() if v['brier'] is not None])
    return result

def canonical(s):
    return re.sub(r'\s+',' ',s.strip().lower())

def parse_answer(text):
    text=re.sub(r'^```(?:json)?\s*|\s*```$','',text.strip())
    try:
        answer=json.loads(text)
        if not isinstance(answer,list) or not all(isinstance(s,str) and s.strip() for s in answer):raise ValueError()
        return set(canonical(s) for s in answer),True
    except (ValueError,TypeError):return {'__invalid_output__'},False

def qa_metrics(records, vocabulary):
    tp=fp=fn=exact=invalid=oov=0
    for r in records:
        predicted,valid=parse_answer(r['text']);target=set(r['answer'])
        tp+=len(predicted&target);fp+=len(predicted-target);fn+=len(target-predicted)
        exact+=int(valid and predicted==target);invalid+=int(not valid)
        oov+=int(valid and bool(predicted-set(vocabulary)))
    return dict(n=len(records),exact_match=exact/len(records) if records else None,
        micro_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,tp=tp,fp=fp,fn=fn,
        invalid_outputs=invalid,oov_outputs=oov,empty_reference_sets=sum(not r['answer'] for r in records))
