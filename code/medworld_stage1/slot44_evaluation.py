"""Four-task evaluation without ablations; preview uses validation patients only."""
import argparse
import json
import re
import time
from pathlib import Path
import numpy as np
import torch
from skimage.metrics import structural_similarity
from bootstrap import atomic_json, write_rows, ORGANS, digest
from evaluation import classification_metrics, save_dense_figure


def canonical(text):
    return re.sub(r'\s+', ' ', text.strip().lower())


def parse_list(text, vocabulary):
    raw = text.strip()
    if raw.startswith('```'):
        raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw)
    try:
        answer = json.loads(raw)
        if not isinstance(answer,list) or not all(isinstance(s,str) and s.strip() for s in answer):
            raise ValueError('Expected JSON array of nonempty strings')
        parsed = sorted({canonical(x) for x in answer})
        return parsed, True, sorted(set(parsed)-set(vocabulary))
    except (ValueError, TypeError):
        return ['__unparseable_output__'], False, []


def set_metrics(rows, vocabulary):
    if not rows:
        return {'n':0}
    per = {}
    total_tp = total_fp = total_fn = 0
    for r in rows:
        p, y = set(r['parsed']), set(r['answer'])
        total_tp += len(p&y); total_fp += len(p-y); total_fn += len(y-p)
    def prf(tp,fp,fn):
        p=tp/(tp+fp) if tp+fp else 0.
        r=tp/(tp+fn) if tp+fn else 0.
        return dict(precision=p,recall=r,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    for label in vocabulary:
        tp=sum(label in r['parsed'] and label in r['answer'] for r in rows)
        fp=sum(label in r['parsed'] and label not in r['answer'] for r in rows)
        fn=sum(label not in r['parsed'] and label in r['answer'] for r in rows)
        per[label]=dict(**prf(tp,fp,fn),support=tp+fn,predicted=tp+fp)
    supported=[v for v in per.values() if v['support']>0]
    empty=[r for r in rows if not r['answer']]
    return dict(n=len(rows),patients=len({r['subject_id'] for r in rows}),images=len({r['image_id'] for r in rows}),
        micro=prf(total_tp,total_fp,total_fn),
        macro={k:float(np.mean([v[k] for v in per.values()])) for k in ['precision','recall','f1']},
        macro_on_supported_labels={k:float(np.mean([v[k] for v in supported])) if supported else None
                                   for k in ['precision','recall','f1']},
        macro_definition='Unweighted average over the fixed disease/finding ontology, zero_division=0',
        per_label=per,exact_match=float(np.mean([r['parse_valid'] and set(r['parsed'])==set(r['answer']) for r in rows])),
        invalid_outputs=sum(not r['parse_valid'] for r in rows),oov_outputs=sum(bool(r['oov']) for r in rows),
        generation_limit_reached=sum(r['generation_limit_reached'] for r in rows),
        empty_answers=len(empty),empty_exact_match=float(np.mean([r['parse_valid'] and not r['parsed'] for r in empty])) if empty else None)


def select_pool(corpus, split, task, limit):
    pool=corpus.pools[split,task]
    if task=='diagnosis':
        # Fixed, balanced interleaving of whole-image and regional questions.
        groups={k:[i for i in pool if corpus.qa[i]['subset']==k] for k in ['whole','region']}
        rng=np.random.default_rng(20260911)
        groups={k:rng.permutation(v).tolist() for k,v in groups.items()}
        pool=[]
        for j in range(max(map(len,groups.values()),default=0)):
            for g in groups.values():
                if j<len(g):pool.append(g[j])
    return pool[:limit] if limit else pool


@torch.no_grad()
def evaluate(model, corpus, run):
    out=run/'evaluation';out.mkdir(parents=True,exist_ok=True)
    model.eval()
    cfg=model.cfg
    split=cfg.get('evaluation_split','test')
    limit=cfg.get('evaluation_limit',0)
    vocabulary=json.loads(Path(cfg['answer_vocabulary']).read_text())['labels']
    result=dict(checkpoint=cfg.get('evaluation_checkpoint','checkpoint_final.pt'),step=cfg.get('evaluation_step'),
        split=split,started=time.time(),qa_source=corpus.selection['qa_source'],
        scope={'classification':'current-report-assisted clinical readout, masked official labels',
               'disease_recognition':'query-conditioned disease/finding sets; current report enters encoder only',
               'segmentation':'agreement with frozen CXAS pseudo labels, not human ground truth',
               'sr':'antialiased bicubic x2 per dimension, HR long edge <=512; valid rectangles only',
               'checkpoint_selection':'fixed-time preview on validation; final checkpoint for final test'})
    def flush():
        atomic_json(out/'metrics.json',result)
    pool=select_pool(corpus,split,'classification',limit)
    ys=[];scores=[];records=[]
    for start in range(0,len(pool),8):
        ii=pool[start:start+8];b=corpus.batch(ii,'classification')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            s=model.route(model.encode(b,'classification'),'classification')
            p=model.classification(s).float().sigmoid().cpu().tolist()
        y=b['labels'].cpu().tolist();ys.extend(y);scores.extend(p)
        records.extend(dict(image_id=corpus.rows[i]['id'],labels=a,scores=v) for i,a,v in zip(ii,y,p))
    result['classification']=classification_metrics(ys,scores)
    result['classification']['label_coverage']=corpus.label_coverage[split]
    write_rows(out/'classification.jsonl',records);flush()
    print('evaluated classification',len(pool),flush=True)
    pool=select_pool(corpus,split,'segmentation',limit);records=[]
    for start in range(0,len(pool),8):
        ii=pool[start:start+8];b=corpus.batch(ii,'segmentation')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            s=model.route(model.encode(b,'segmentation'),'segmentation')
            logits,_=model.segmentation(b['seg_image'],s)
        valid=b['seg_mask'];target=(b['seg_probs']>.5).float()*valid
        pred=(logits.float().sigmoid()>.5).float()*valid
        inter=(pred*target).sum((2,3));total=pred.sum((2,3))+target.sum((2,3))
        dice=(2*inter+1e-6)/(total+1e-6);iou=(inter+1e-6)/(total-inter+1e-6)
        for j,i in enumerate(ii):
            records.append(dict(image_id=corpus.rows[i]['id'],dice=dice[j].cpu().tolist(),iou=iou[j].cpu().tolist()))
            if start+j<4:
                save_dense_figure(out/f'seg_{start+j:02d}.png',
                    [b['seg_image'][j,0].cpu().numpy(),target[j].amax(0).cpu().numpy(),pred[j].amax(0).cpu().numpy()],
                    ['Current image','CXAS masks (union)','Predicted masks (union)'])
                for k,organ in enumerate(ORGANS):
                    save_dense_figure(out/f'seg_{start+j:02d}_{k}.png',
                        [b['seg_image'][j,0].cpu().numpy(),target[j,k].cpu().numpy(),pred[j,k].cpu().numpy()],
                        ['Current image',f'CXAS {organ}',f'Predicted {organ}'])
    result['segmentation']=dict(n=len(records),mean_dice=float(np.mean([r['dice'] for r in records])),
        mean_iou=float(np.mean([r['iou'] for r in records])),
        organs={o:dict(dice=float(np.mean([r['dice'][j] for r in records])),iou=float(np.mean([r['iou'][j] for r in records])))
                for j,o in enumerate(ORGANS)})
    write_rows(out/'segmentation.jsonl',records);flush()
    print('evaluated segmentation',len(pool),flush=True)
    pool=select_pool(corpus,split,'sr',limit);records=[]
    for start in range(0,len(pool),8):
        ii=pool[start:start+8];b=corpus.batch(ii,'sr')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            s=model.route(model.encode(b,'sr'),'sr')
            pred=model.sr(b['lr'],s).float().clamp(0,1)
        for j,i in enumerate(ii):
            row=corpus.rows[i];y,x,h,w=row['box']
            target=b['hr'][j,0,y:y+h,x:x+w].cpu().numpy()
            p=pred[j,0,y:y+h,x:x+w].cpu().numpy()
            mse=float(np.mean((p-target)**2))
            records.append(dict(image_id=row['id'],hr_size=[h,w],lr_size=row['lr_size'],view=row['view'],
                psnr=float(-10*np.log10(max(mse,1e-12))),ssim=float(structural_similarity(target,p,data_range=1.))))
            if start+j<4:
                lr=b['lr'][j,0,y//2:(y+h)//2,x//2:(x+w)//2].cpu().numpy()
                save_dense_figure(out/f'sr_{start+j:02d}.png',[lr,p,target,np.abs(p-target)],['LR (x2)','Prediction','HR','Absolute error'])
    result['sr']=dict(n=len(records),psnr=float(np.mean([r['psnr'] for r in records])),ssim=float(np.mean([r['ssim'] for r in records])))
    write_rows(out/'sr.jsonl',records);flush()
    print('evaluated sr',len(pool),flush=True)
    qa_limit=cfg.get('diagnosis_eval_count',0)
    pool=select_pool(corpus,split,'diagnosis',qa_limit);records=[]
    for start in range(0,len(pool),4):
        ii=pool[start:start+4];b=corpus.batch(ii,'diagnosis')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            s=model.encode(b,'diagnosis')
            texts,truncated=model.diagnosis.generate(s,b['query_ids'],b['query_mask'],cfg['generation_tokens'])
        for qi,text,trunc in zip(ii,texts,truncated):
            q=corpus.qa[qi];parsed,ok,oov=parse_list(text,vocabulary)
            records.append(dict(q,generated=text,parsed=parsed,parse_valid=ok,oov=oov,generation_limit_reached=trunc))
        write_rows(out/'disease_recognition.jsonl',records)
        print('disease lists',len(records),len(pool),flush=True)
    result['disease_recognition']={k:set_metrics(rr,vocabulary) for k,rr in
        [('all',records),('whole',[r for r in records if r['subset']=='whole']),('region',[r for r in records if r['subset']=='region'])]}
    result['completed']=time.time();flush()
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--split',choices=['validate','test'],default='validate');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--qa-limit',type=int,default=128)
    a=p.parse_args()
    from slot44_networks import Slot44Model
    from slot44_corpus import Slot44Corpus
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    cfg=dict(saved['cfg'],evaluation_checkpoint=str(a.checkpoint),evaluation_step=saved['step'],
             evaluation_split=a.split,evaluation_limit=a.limit,diagnosis_eval_count=a.qa_limit)
    for key in ['selection','qa_manifest','answer_vocabulary']:
        assert digest(Path(cfg[key]))==saved['signatures'][key],f'{key} changed since training'
    model=Slot44Model(cfg).cuda();model.load_compact(saved['model']);del saved
    corpus=Slot44Corpus(cfg,model.tokenizer)
    evaluate(model,corpus,a.out)
