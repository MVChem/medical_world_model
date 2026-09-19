"""Score generated reports with the existing frozen CheXbert protocol, without feature caches."""
import argparse
import json
from pathlib import Path
import sys
from ..config import PROJECT
from ..gpu import acquire_gpu
from .data import FINDINGS, _rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--predictions',required=True);p.add_argument('--out',required=True);p.add_argument('--gpu',default='auto')
    a=p.parse_args();lock,device=acquire_gpu(a.gpu)
    try:
        clinical_root=PROJECT/'code/medworld_table1'
        sys.path.insert(0,str(clinical_root))
        from clinical import CheXbert
        from metrics import clinical_metrics
        rows=_rows(Path(a.predictions));scorer=CheXbert(device=device)
        truth=scorer.labels([r['reference'] for r in rows],FINDINGS)
        predicted=scorer.labels([r['prediction'] for r in rows],FINDINGS)
        metrics=clinical_metrics(truth,truth,predicted,None,FINDINGS)
        result={'n':len(rows),'chexbert_f1':metrics['chexbert_f1'],'details':metrics,'provenance':scorer.provenance,
                'input':str(Path(a.predictions).resolve()),'note':'Fixed reference mask; generated text scored without cached labels/features.'}
        Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'n':len(rows),'chexbert_f1':result['chexbert_f1']}),flush=True)
    finally:
        if lock is not None:lock.close()


if __name__=='__main__':main()
