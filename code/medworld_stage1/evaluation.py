"""Held-out task scores, state ablations, frozen probes, and reviewable examples."""
import json
import subprocess
import sys
from pathlib import Path
import bootstrap
from bootstrap import *
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score,roc_auc_score
from skimage.metrics import structural_similarity


def classification_metrics(labels,scores):
    y=np.asarray(labels);p=np.asarray(scores)
    per={}
    for k,name in enumerate(FINDINGS):
        valid=(y[:,k]==0)|(y[:,k]==1)
        truth=y[valid,k];pred=p[valid,k]
        pos=int((truth==1).sum());neg=int((truth==0).sum())
        per[name]=dict(positive=pos,negative=neg,n=int(valid.sum()),
            auroc=float(roc_auc_score(truth,pred)) if pos and neg else None,
            auprc=float(average_precision_score(truth,pred)) if pos and neg else None)
    return dict(n=len(y),per_class=per,macro_auroc=mean_available([v['auroc'] for v in per.values()]),
                macro_auprc=mean_available([v['auprc'] for v in per.values()]))


def mean_available(values):
    x=[v for v in values if v is not None]
    return float(np.mean(x)) if x else None


def save_dense_figure(path,columns,titles):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(columns),figsize=(4*len(columns),4),squeeze=False)
    for ax,a,title in zip(axes[0],columns,titles):
        ax.imshow(a,cmap='gray',vmin=0,vmax=1)
        ax.set_title(title);ax.axis('off')
    fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)


@torch.no_grad()
def shuffled_state(model,corpus,task,state,pool,start):
    if len(state)>1:return state.roll(1,0)
    other=pool[(start+1)%len(pool)]
    return model.encode(corpus.batch([other],task),task)


@torch.no_grad()
def evaluate(model,corpus,run):
    model.eval()
    out=run/'evaluation';out.mkdir(exist_ok=True)
    result=dict(checkpoint='checkpoint_final.pt',classification={},segmentation={},sr={},diagnosis={},
        scope=dict(segmentation='agreement with frozen CXAS pseudo labels',sr=f'linear scale={model.cfg["scale"]}; pixel factor={model.cfg["scale"]**2}',
        state_ablation='inference intervention; null is not a separately trained image-only model',
        classification='report-derived study labels; state includes current report',diagnosis='multimodal report reconstruction'))
    eval_split=model.cfg.get('evaluation_split','test')
    result['scope']['evaluation_split']=eval_split
    state_sets={}
    for split in ['validate','test']:
        actual_split=eval_split if split=='test' else split
        pool=corpus.pools[actual_split,'classification'][:model.cfg.get('evaluation_limit',1000000)];states=[];ys=[];scores={v:[] for v in ['actual','null','shuffled']}
        for start in range(0,len(pool),4):
            ii=pool[start:start+4];b=corpus.batch(ii,'classification')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                s=model.encode(b,'classification')
                sh=shuffled_state(model,corpus,'classification',s,pool,start)
                for variant,z in [('actual',s),('null',torch.zeros_like(s)),('shuffled',sh)]:
                    scores[variant].extend(model.classification(z).float().sigmoid().cpu().tolist())
            states.append(s.float().cpu());ys.extend(b['labels'].cpu().tolist())
        state_sets[split]=(torch.cat(states),torch.tensor(ys))
        result['classification'][split]={v:classification_metrics(ys,p) for v,p in scores.items()}
        write_rows(out/f'classification_{split}.jsonl',[dict(id=corpus.rows[i]['id'],labels=y,
            **{v:scores[v][j] for v in scores}) for j,(i,y) in enumerate(zip(pool,ys))])
        atomic_json(out/'metrics.json',result)
    result['slot_diagnostics']={}
    for split,(states,_) in state_sets.items():
        norm=F.normalize(states.float(),dim=-1)
        gram=torch.einsum('bsd,btd->bst',norm,norm)
        offdiag=~torch.eye(8,dtype=torch.bool)
        flattened=norm.flatten(1)
        centered=flattened-flattened.mean(0,keepdim=True)
        singular=torch.linalg.svdvals(centered)
        mass=singular.square()/singular.square().sum().clamp_min(1e-12)
        effective_rank=float(torch.exp(-(mass*mass.clamp_min(1e-12).log()).sum()))
        result['slot_diagnostics'][split]=dict(n=len(states),within_image_slot_cosine=float(gram[:,offdiag].mean()),
            between_patient_cosine=float(F.cosine_similarity(flattened,flattened.roll(1,0)).mean()),
            centered_effective_rank=effective_rank,
            interpretation='descriptive redundancy/variation checks; cosine alone does not establish collapse')
    pool=corpus.pools[eval_split,'segmentation'][:model.cfg.get('evaluation_limit',1000000)]
    seg_records=[]
    for start in range(0,len(pool),4):
        ii=pool[start:start+4];b=corpus.batch(ii,'segmentation')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            s=model.encode(b,'segmentation');sh=shuffled_state(model,corpus,'segmentation',s,pool,start)
            variants={v:model.segmentation(b['seg_image'],z)[0].float().sigmoid() for v,z in
                [('actual',s),('null',torch.zeros_like(s)),('shuffled',sh)]}
        target=(b['seg_probs']>.5).float()*b['seg_mask']
        records=[dict(id=corpus.rows[i]['id']) for i in ii]
        for name,pred in variants.items():
            p=(pred>.5).float()*b['seg_mask']
            inter=(p*target).sum((2,3));total=p.sum((2,3))+target.sum((2,3))
            dice=(2*inter+1e-6)/(total+1e-6);iou=(inter+1e-6)/(total-inter+1e-6)
            for j,r in enumerate(records):r[name]=dict(dice=dice[j].cpu().tolist(),iou=iou[j].cpu().tolist())
        seg_records.extend(records)
        if start<8:
            for j,i in enumerate(ii):
                image=b['seg_image'][j,0].cpu().numpy()
                # Separate mask panels retain projected overlaps without invented colors.
                save_dense_figure(out/f'seg_{start+j:02d}.png',[image,*target[j].cpu().numpy(),*variants['actual'][j].cpu().numpy()],
                    ['CXR',*[f'Teacher {o}' for o in ORGANS],*[f'Student {o}' for o in ORGANS]])
    for v in ['actual','null','shuffled']:
        result['segmentation'][v]=dict(n=len(seg_records),organs=ORGANS,
            dice=np.mean([r[v]['dice'] for r in seg_records],axis=0).tolist(),
            iou=np.mean([r[v]['iou'] for r in seg_records],axis=0).tolist())
        result['segmentation'][v]['mean_dice']=float(np.mean(result['segmentation'][v]['dice']))
    write_rows(out/'segmentation.jsonl',seg_records);atomic_json(out/'metrics.json',result)
    pool=corpus.pools[eval_split,'sr'][:model.cfg.get('evaluation_limit',1000000)];sr_records=[]
    for start in range(0,len(pool),4):
        ii=pool[start:start+4];b=corpus.batch(ii,'sr')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            s=model.encode(b,'sr');sh=shuffled_state(model,corpus,'sr',s,pool,start)
            variants={v:model.sr(b['lr'],z).float().clamp(0,1) for v,z in
                [('actual',s),('null',torch.zeros_like(s)),('shuffled',sh)]}
        variants['bicubic']=F.interpolate(b['lr'],b['hr'].shape[-2:],mode='bicubic',align_corners=False).clamp(0,1)
        for j,i in enumerate(ii):
            row=corpus.rows[i];y,x,h,w=row['box']
            target=b['hr'][j,0,y:y+h,x:x+w].cpu().numpy()
            rec=dict(id=row['id'],view=row['view'],hr_size=[h,w],lr_size=row['lr_size'])
            for name,p in variants.items():
                a=p[j,0,y:y+h,x:x+w].cpu().numpy()
                mse=float(np.mean((a-target)**2))
                rec[name]=dict(psnr=float(-10*np.log10(max(mse,1e-12))),ssim=float(structural_similarity(target,a,data_range=1.)))
            sr_records.append(rec)
            if start+j<8:
                save_dense_figure(out/f'sr_{start+j:02d}.png',
                    [variants['bicubic'][j,0,y:y+h,x:x+w].cpu().numpy(),variants['actual'][j,0,y:y+h,x:x+w].cpu().numpy(),target],
                    ['LR bicubic','Slot-conditioned SR','HR target'])
    for v in ['actual','null','shuffled','bicubic']:
        result['sr'][v]=dict(n=len(sr_records),psnr=float(np.mean([r[v]['psnr'] for r in sr_records])),
                            ssim=float(np.mean([r[v]['ssim'] for r in sr_records])))
    write_rows(out/'sr.jsonl',sr_records);atomic_json(out/'metrics.json',result)
    pool=corpus.pools[eval_split,'diagnosis'][:model.cfg['diagnosis_eval_count']];reports=[]
    for start in range(0,len(pool),2):
        ii=pool[start:start+2];b=corpus.batch(ii,'diagnosis')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            s=model.encode(b,'diagnosis')
            generated=model.diagnosis.generate(s,model.cfg['generation_tokens'])
            if start<16:
                sh=shuffled_state(model,corpus,'diagnosis',s,pool,start)
                shuffled=model.diagnosis.generate(sh,model.cfg['generation_tokens'])
            else:shuffled=[None]*len(ii)
        for j,i in enumerate(ii):
            reports.append(dict(id=corpus.rows[i]['id'],subject_id=corpus.rows[i]['subject_id'],
                reference=corpus.rows[i]['report'],training_target=model.tokenizer.decode(corpus.targets[i],skip_special_tokens=True),
                generated=generated[j],shuffled_generated=shuffled[j]))
        write_rows(out/'diagnosis.jsonl',reports)
        print('generation',len(reports),len(pool),flush=True)
    texts=[r['generated'] for r in reports]
    from collections import Counter
    freq=Counter(texts)
    comparable=[r for r in reports if r['shuffled_generated'] is not None]
    result['diagnosis']=dict(n=len(reports),unique=len(freq),unique_fraction=len(freq)/len(reports),
        largest_identical_group=max(freq.values()),empty=sum(not s.strip() for s in texts),
        shuffle_changed_fraction=sum(r['generated']!=r['shuffled_generated'] for r in comparable)/len(comparable))
    atomic_json(out/'metrics.json',result)
    result['frozen_probe']=frozen_probe(model,corpus,state_sets)
    atomic_json(out/'metrics.json',result)
    # A separate interpreter is required for the existing RadGraph dependency versions.
    torch.cuda.empty_cache()
    try:
        with (out/'clinical.log').open('w') as log:
            p=subprocess.run([sys.executable,str(ROOT/'clinical_score.py'),'--input',str(out/'diagnosis.jsonl'),
                              '--output',str(out/'clinical.json')],stdout=log,stderr=subprocess.STDOUT,timeout=900)
        if p.returncode:raise RuntimeError(f'clinical worker exited {p.returncode}; see clinical.log')
        result['diagnosis']['clinical']=json.loads((out/'clinical.json').read_text())
    except Exception as e:
        result['diagnosis']['clinical_error']=f'{type(e).__name__}: {e}'
        if (out/'clinical.json').exists():
            result['diagnosis']['clinical']=json.loads((out/'clinical.json').read_text())
    atomic_json(out/'metrics.json',result)
    make_local_report(run,result)
    return result


@torch.no_grad()
def frozen_probe(model,corpus,state_sets):
    pool=corpus.pools['train','classification'][:model.cfg['probe_train_count']]
    xs=[];ys=[]
    for start in range(0,len(pool),8):
        ii=pool[start:start+8];b=corpus.batch(ii,'classification')
        with torch.autocast('cuda',dtype=torch.bfloat16):s=model.encode(b,'classification')
        xs.append(s.float().cpu());ys.append(b['labels'].cpu())
    def features(s):return F.layer_norm(s.float(),(1024,)).flatten(1).cuda()
    x=features(torch.cat(xs));y=torch.cat(ys).cuda()
    gen=torch.Generator(device='cuda').manual_seed(701)
    with torch.enable_grad():
        torch.manual_seed(701)
        head=torch.nn.Linear(8*1024,len(FINDINGS)).cuda()
        opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.1)
        from networks import masked_classification
        for step in range(model.cfg['probe_steps']):
            idx=torch.randint(len(x),(128,),device='cuda',generator=gen)
            opt.zero_grad(set_to_none=True)
            loss=masked_classification(head(x[idx]),y[idx],corpus.pos_weight.cuda())
            loss.backward();opt.step()
    result=dict(train_images=len(pool),updates=model.cfg['probe_steps'],input='all 8 frozen slots flattened after layer norm')
    for split,(s,y0) in state_sets.items():
        pred=head(features(s)).sigmoid().cpu().tolist()
        result[split]=classification_metrics(y0.tolist(),pred)
    return result


def make_local_report(run,result):
    c=result['classification']['test']['actual'];g=result['segmentation']['actual'];s=result['sr']['actual'];b=result['sr']['bicubic']
    text=f'''# Four-task Stage 1 result

Checkpoint: `checkpoint_final.pt`. Base: Qwen3.5-0.8B; shared multimodal slots: 8×1024.

| Task | Metric | Final |
|---|---|---:|
| Classification | macro AUPRC | {c['macro_auprc']} |
| Classification | macro AUROC | {c['macro_auroc']} |
| Segmentation | CXAS agreement Dice | {g['mean_dice']} |
| SR | PSNR / SSIM | {s['psnr']:.4f} / {s['ssim']:.4f} |
| Bicubic | PSNR / SSIM | {b['psnr']:.4f} / {b['ssim']:.4f} |

Full scores, support, inference state ablations, frozen probe and clinical text metrics: [metrics.json](evaluation/metrics.json).
Generated reports: [diagnosis.jsonl](evaluation/diagnosis.jsonl). Dense figures: `evaluation/seg_*.png` and `evaluation/sr_*.png`.

Segmentation scores measure imitation of a teacher, not human-mask accuracy. Classification and diagnosis states include reports.
SR uses synthetic degradation, retains aspect ratio, caps HR long edge at 512 and excludes padding from scores.
'''
    (run/'REPORT.md').write_text(text)
