"""Existing official frozen CheXbert and RadGraph; failures remain explicit."""
import sys
from pathlib import Path
root=Path(__file__).resolve().parent
pilot=root.parent/'medworld_table1'
sys.path.insert(0,str(pilot/'vendor'))
sys.path.insert(0,str(pilot))
sys.path.insert(0,str(pilot/'metric_vendor'))
import argparse
import json
import numpy as np
import torch
from common import atomic_json,load_rows
from clinical import CheXbert


def main(source,output):
    rows=load_rows(source)
    refs=[r['reference'] for r in rows];hyps=[r['generated'] for r in rows]
    names=['Atelectasis','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum','Fracture',
           'Lung Lesion','Lung Opacity','Pleural Effusion','Pleural Other','Pneumonia','Pneumothorax','Support Devices']
    result=dict(n=len(rows))
    try:
        extractor=CheXbert()
        y=np.array(extractor.labels(refs,names));p=np.array(extractor.labels(hyps,names))
        per={}
        for k,name in enumerate(names):
            valid=(y[:,k]==0)|(y[:,k]==1)
            a=y[valid,k]==1;b=p[valid,k]==1
            tp=int((a&b).sum());fp=int((~a&b).sum());fn=int((a&~b).sum())
            per[name]=dict(positive=int(a.sum()),valid=int(valid.sum()),tp=tp,fp=fp,fn=fn,
                           f1=2*tp/(2*tp+fp+fn) if a.sum() else None)
        f1s=[v['f1'] for v in per.values() if v['f1'] is not None]
        result['chexbert']=dict(macro_positive_f1=float(np.mean(f1s)) if f1s else None,per_class=per,
            provenance=extractor.provenance,reference_labels=y.tolist(),prediction_labels=p.tolist())
        del extractor;torch.cuda.empty_cache()
    except Exception as e:result['chexbert_error']=f'{type(e).__name__}: {e}'
    atomic_json(output,result)
    try:
        from radgraph import F1RadGraph
        scorer=F1RadGraph(reward_level='all',model_type='radgraph-xl',cuda=0,model_cache_dir=str(pilot/'weights/radgraph'))
        reward,_,_,_=scorer(hyps=hyps,refs=refs)
        result['radgraph']=dict(partial_f1=float(reward[1]),model='radgraph-xl',metric='RG_ER/partial report mean')
    except Exception as e:result['radgraph_error']=f'{type(e).__name__}: {e}'
    atomic_json(output,result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.input,a.output)
