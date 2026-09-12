"""Matched-epoch independent dense heads, fixed cohorts and final-epoch evaluation."""
import argparse
import hashlib
import json
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity
from common import *
from features import downsample
from heads import DenseHead,GroundingHead,box_iou,segmentation_loss

class Corpus:
    def __init__(self,run,mid,task,variant):
        self.rows=read_rows(run/'data/observations.jsonl');self.task=task;self.variant=variant
        self.images=np.load(run/'data/images.npy',mmap_mode='r')
        self.lr_images=np.load(run/'data/lr_images.npy',mmap_mode='r')
        self.hidden=np.load(run/mid/f'{"lr" if task=="sr" else "hr"}_hidden.npy',mmap_mode='r')
        self.human=np.load(run/'data/human_masks.npy',mmap_mode='r')
        self.pseudo=np.load(OLD/'seg_probs.npy',mmap_mode='r')
        if variant!='image':
            self.features=np.load(run/'data'/f'vjepa_{"lr" if task=="sr" else "hr"}.npy',mmap_mode='r')
        done=np.load(run/mid/'features_done.npy')
        self.queries=read_rows(run/'data/grounding.jsonl')
        self.vocab=json.loads((run/'data/query_vocab.json').read_text())
        self.pool={}
        for split in ['train','validate','test','human_test']:
            if task=='grounding':
                self.pool[split]=[i for i,r in enumerate(self.queries) if r['split']==split]
            else:
                self.pool[split]=[i for i,r in enumerate(self.rows) if r['split']==split and task in r['tasks']]
        required=sorted({self.queries[i]['index'] if task=='grounding' else i for ids in self.pool.values() for i in ids})
        assert done[required,1 if task=='sr' else 0].all(),'incomplete VLM cache'
        if variant!='image':
            vdone=np.load(run/'data/vjepa_done.npy')
            assert vdone[required,1 if task=='sr' else 0].all(),'incomplete V-JEPA cache'

    def batch(self,ids):
        ii=[self.queries[i]['index'] for i in ids] if self.task=='grounding' else ids
        def gpu(a):return torch.from_numpy(np.array(a,copy=True)).cuda().float()
        image=gpu(self.images[ii])[:,None]/255
        hidden=gpu(self.hidden[ii])
        if not torch.isfinite(hidden).all():raise ValueError('nonfinite hidden')
        size=512 if self.task=='sr' else 256
        mask=torch.zeros((len(ii),1,size,size),device='cuda')
        for b,i in enumerate(ii):
            y,x,h,w=self.rows[i]['box'];scale=512//size
            y,x,h,w=[v//scale for v in [y,x,h,w]]
            mask[b,:,y:y+h,x:x+w]=1
        if self.task=='grounding':
            visual=F.interpolate(image,(256,256),mode='area')
            target=gpu(np.array([self.queries[i]['xyxy'] for i in ids],np.float32))
            query=torch.tensor([self.vocab.index(self.queries[i]['phrase']) for i in ids],device='cuda')
            return visual,hidden,target,mask,query
        if self.task=='sr':
            target=image
            visual=gpu(self.lr_images[ii])[:,None]/255
        else:
            visual=F.interpolate(image,(256,256),mode='area')
            target=gpu(np.stack([self.human[self.rows[i]['human_index']] if self.rows[i]['kind']=='montgomery'
                                else self.pseudo[self.rows[i]['old_index']] for i in ii]))
        if self.variant!='image':visual=gpu(self.features[ii])
        return visual,hidden,target,mask,None

def objective(task,pred,target,mask):
    if task=='segmentation':return segmentation_loss(pred[:,:target.shape[1]],target,mask)
    if task=='sr':return (((pred-target).abs()*mask).sum((1,2,3))/mask.sum((1,2,3)).clamp_min(1)).mean()
    return F.smooth_l1_loss(pred,target)+1-box_iou(pred,target).mean()

def forward(model,b):
    x,h,t,mask,q=b
    return model(x,h,q) if q is not None else model(x,h)

@torch.inference_mode()
def validation(model,corpus,batch_size):
    model.eval();total=0.;n=0
    ids=corpus.pool['validate']
    for start in range(0,len(ids),batch_size):
        ii=ids[start:start+batch_size];b=corpus.batch(ii)
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=forward(model,b)
        total+=float(objective(corpus.task,pred,b[2],b[3]))*len(ii);n+=len(ii)
    return total/max(n,1)

def bootstrap(records,keys):
    # Cluster bootstrap: anatomical boxes can share a patient.
    by={}
    for r in records:by.setdefault(r['subject_id'],[]).append(r)
    groups=list(by.values());rng=np.random.default_rng(SEED)
    means={k:[] for k in keys}
    for _ in range(1000):
        sample=[r for i in rng.integers(0,len(groups),len(groups)) for r in groups[i]]
        for k in keys:means[k].append(np.mean([r[k] for r in sample]))
    return {k:np.quantile(v,[.025,.975]).tolist() for k,v in means.items()}

@torch.inference_mode()
def evaluate(model,corpus,out,batch_size,epochs):
    model.eval();results={}
    for split in ['test']+(['human_test'] if corpus.task=='segmentation' else []):
        records=[];ids=corpus.pool[split]
        for start in range(0,len(ids),batch_size):
            ii=ids[start:start+batch_size];b=corpus.batch(ii)
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=forward(model,b).float()
            target,mask=b[2],b[3]
            if corpus.task=='segmentation':
                pp=(pred[:,:target.shape[1]].sigmoid()>.5).float()*mask
                tt=(target>.5).float()*mask
                dd=(2*(pp*tt).sum((2,3))+1e-8)/(pp.sum((2,3))+tt.sum((2,3))+1e-8)
            elif corpus.task=='grounding':ious=box_iou(pred,target)
            for j,i in enumerate(ii):
                r=corpus.queries[i] if corpus.task=='grounding' else corpus.rows[i]
                rec=dict(id=r['id'],subject_id=r['subject_id'])
                if corpus.task=='segmentation':
                    rec.update(dice=float(dd[j].mean()),per_organ=dd[j].cpu().tolist())
                elif corpus.task=='grounding':
                    rec.update(phrase=r['phrase'],iou=float(ious[j]),acc50=float(ious[j]>=.5),
                        prediction=pred[j].cpu().tolist(),target=target[j].cpu().tolist())
                else:
                    y,x,h,w=r['box']
                    a=pred[j,0,y:y+h,x:x+w].clamp(0,1).cpu().numpy()
                    t=target[j,0,y:y+h,x:x+w].cpu().numpy()
                    mse=float(np.mean((a-t)**2))
                    # skimage default 7x7 uniform SSIM, data range 1; full valid ROI, no border shave.
                    rec.update(psnr=float(-10*np.log10(max(mse,1e-12))),
                        ssim=float(structural_similarity(t,a,data_range=1.)))
                records.append(rec)
        assert len(records)==len(ids)>0,'fixed test cohort lost samples'
        keys=['dice'] if corpus.task=='segmentation' else ['iou','acc50'] if corpus.task=='grounding' else ['psnr','ssim']
        write_rows(out/f'{split}_per_sample.jsonl',records)
        results[split]=dict(n=len(records),patients=len({r['subject_id'] for r in records}),
            **{k:float(np.mean([r[k] for r in records])) for k in keys},ci95=bootstrap(records,keys))
        if corpus.task=='segmentation':
            results[split]['per_organ']=np.mean([r['per_organ'] for r in records],axis=0).tolist()
    atomic(out/'metrics.json',dict(task=corpus.task,variant=corpus.variant,epochs=epochs,
        checkpoint='final epoch; no test-based checkpoint selection',metrics=results,
        cohort_sha256=digest(out.parents[1]/'data/observations.jsonl')))
    return results

def train(run,mid,task,variant,epochs=20,batch_size=8,microbatch=4):
    torch.set_num_threads(4)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False
    corpus=Corpus(run,mid,task,variant)
    out=run/mid/f'{task}_{variant}';out.mkdir(parents=True,exist_ok=True)
    contract=dict(model=mid,task=task,variant=variant,epochs=epochs,batch_size=batch_size,seed=SEED,
        learning_rate=3e-4,weight_decay=.01,optimizer='AdamW',backbones_frozen=True,
        train_ids_sha256=hashlib.sha256(json.dumps(corpus.pool['train']).encode()).hexdigest(),
        cohort_sha256=digest(run/'data/observations.jsonl'),
        heads_sha256=digest(Path(__file__).parent/'heads.py'))
    if (out/'contract.json').exists():assert json.loads((out/'contract.json').read_text())==contract
    atomic(out/'contract.json',contract)
    if (out/'metrics.json').exists():return
    model=(GroundingHead(len(corpus.vocab)) if task=='grounding' else DenseHead(task,variant)).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    initial_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    atomic(out/'initialization.json',dict(state_sha256=initial_hash,parameters=sum(p.numel() for p in model.parameters()),
        microbatch=microbatch,effective_batch=batch_size))
    first=0;step=0
    if (out/'checkpoint.pt').exists():
        ck=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);first=ck['epoch'];step=ck['step']
    started=time.time()
    for epoch in range(first,epochs):
        model.train();total=0.;n=0
        ids=np.random.default_rng(SEED+epoch).permutation(corpus.pool['train']).tolist()
        lr=3e-4*(.1+.9*.5*(1+np.cos(np.pi*epoch/epochs)))
        for g in optimizer.param_groups:g['lr']=lr
        for start in range(0,len(ids),batch_size):
            chunk=ids[start:start+batch_size];optimizer.zero_grad(set_to_none=True)
            for s in range(0,len(chunk),microbatch):
                ii=chunk[s:s+microbatch];b=corpus.batch(ii)
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=forward(model,b)
                loss=objective(task,pred,b[2],b[3])
                if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
                (loss*len(ii)/len(chunk)).backward();total+=float(loss.detach())*len(ii);n+=len(ii)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();step+=1
            if step%50==0:
                atomic(out/'progress.json',dict(epoch=epoch+1,epochs=epochs,step=step,samples_in_epoch=n,
                    training_loss=total/n,seconds=time.time()-started,status='training'))
        val=validation(model,corpus,microbatch)
        record=dict(epoch=epoch+1,step=step,train_loss=total/n,validate_loss=val,learning_rate=lr,
            order_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),seconds=time.time()-started)
        with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,step=step),out/'checkpoint.tmp.pt')
        os.replace(out/'checkpoint.tmp.pt',out/'checkpoint.pt')
        if epoch+1 in [5,10,20]:
            torch.save(dict(model=model.state_dict(),epoch=epoch+1),out/f'epoch_{epoch+1}.pt')
        atomic(out/'progress.json',dict(**record,epochs=epochs,status='epoch_complete'))
        print(mid,task,variant,record,flush=True)
    evaluate(model,corpus,out,microbatch,epochs)
    atomic(out/'progress.json',dict(epoch=epochs,epochs=epochs,step=step,status='complete',seconds=time.time()-started))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT_RUN)
    p.add_argument('--model',required=True);p.add_argument('--task',choices=['segmentation','sr','grounding'],required=True)
    p.add_argument('--variant',choices=['image','vjepa','vjepa_adapter'],default='image')
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--microbatch',type=int,default=4)
    a=p.parse_args();train(a.run,a.model,a.task,a.variant,a.epochs,a.batch_size,a.microbatch)
