"""Use precisely the reference four-task scoring code for the no-slot model."""
import argparse
import json
from pathlib import Path
import torch
from bootstrap import digest,atomic_json
from noslots_networks import NoSlotsModel
from slot44_corpus import Slot44Corpus
from slot44_evaluation import evaluate


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--split',choices=['validate','test'],default='validate');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--qa-limit',type=int,default=128)
    a=p.parse_args()
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    cfg=dict(saved['cfg'],evaluation_checkpoint=str(a.checkpoint.resolve()),evaluation_step=saved['step'],
             evaluation_split=a.split,evaluation_limit=a.limit,diagnosis_eval_count=a.qa_limit)
    for key in ['selection','qa_manifest','answer_vocabulary']:
        assert digest(Path(cfg[key]))==saved['signatures'][key], f'{key} changed since training'
    model=NoSlotsModel(cfg).cuda();model.load_compact(saved['model']);del saved
    corpus=Slot44Corpus(cfg,model.tokenizer)
    metrics=evaluate(model,corpus,a.out)
    metrics['model_kind']='Qwen3.5-0.8B full token readout, no learned state slots'
    metrics['scope']['disease_recognition']='same image/report inputs, uncompressed multimodal token prefix, same queries and scoring'
    metrics['scope']['checkpoint_selection']='optimizer step matched to reference checkpoint'
    atomic_json(a.out/'evaluation/metrics.json',metrics)
