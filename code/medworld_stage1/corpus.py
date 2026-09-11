import json
from pathlib import Path
import bootstrap
from bootstrap import *
import numpy as np
import torch
import torch.nn.functional as F
from cache import downsample


class Corpus:
    def __init__(self,cfg,tokenizer):
        self.cfg=cfg
        self.tokenizer=tokenizer
        root=Path(cfg['cache'])
        manifest=json.loads((root/'manifest.json').read_text())
        assert manifest['scale']==cfg['scale'],'SR scale differs from cached LR generation'
        for name in ['features.json','segmentation.json']:
            meta=json.loads((root/name).read_text())
            assert meta['observations_sha256']==manifest['observations_sha256'],f'{name}: observation mismatch'
        self.rows=load_rows(root/'observations.jsonl')
        self.images=np.load(root/'images.npy',mmap_mode='r')
        self.hr_features=np.load(root/'hr_features.npy',mmap_mode='r')
        self.lr_features=np.load(root/'lr_features.npy',mmap_mode='r')
        self.seg_probs=np.load(root/'seg_probs.npy',mmap_mode='r')
        self.seg_valid=np.load(root/'seg_valid_qc.npy')
        assert len(self.rows)==manifest['valid_count']==len(self.hr_features)==len(self.lr_features)==len(self.seg_valid)
        assert self.hr_features.shape==self.lr_features.shape==(len(self.rows),64,768)
        assert self.seg_probs.shape==(len(self.rows),3,256,256)
        all_tokens=tokenizer([r['report'] for r in self.rows],add_special_tokens=False)['input_ids']
        self.tokens=[x[:cfg['report_tokens']] for x in all_tokens]
        self.targets=[x[:cfg['target_tokens']]+[tokenizer.eos_token_id] for x in all_tokens]
        self.truncation={s:{'input':sum(len(x)>cfg['report_tokens'] for r,x in zip(self.rows,all_tokens) if r['split']==s),
                            'target':sum(len(x)>cfg['target_tokens'] for r,x in zip(self.rows,all_tokens) if r['split']==s)}
                         for s in ['train','validate','test']}
        self.pools={(s,t):[i for i,r in enumerate(self.rows) if r['split']==s and r['tasks'][t]
                          and (t!='segmentation' or self.seg_valid[i])]
                    for s in ['train','validate','test'] for t in cfg['tasks']}
        labels=np.array([self.rows[i]['labels'] for i in self.pools['train','classification']])
        self.pos_weight=torch.tensor(np.clip((labels==0).sum(0)/np.maximum((labels==1).sum(0),1),.25,10),dtype=torch.float32)
        self.label_coverage={}
        for split in ['train','validate','test']:
            a=np.asarray([self.rows[i]['labels'] for i in self.pools[split,'classification']])
            self.label_coverage[split]={name:{str(v):int((a[:,j]==v).sum()) for v in [-2,-1,0,1]}
                                        for j,name in enumerate(FINDINGS)}
        self.missing_report={f'{s}/{t}':sum(not self.rows[i]['report'].strip() for i in pool)
                             for (s,t),pool in self.pools.items()}
        self.permutations={}
    def pad(self,seqs,left=False):
        width=max(1,max(map(len,seqs)))
        ids=torch.full((len(seqs),width),self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,dtype=torch.long)
        mask=torch.zeros_like(ids)
        for i,s in enumerate(seqs):
            if s:
                sl=slice(width-len(s),width) if left else slice(0,len(s))
                ids[i,sl]=torch.tensor(s)
                mask[i,sl]=1
        return ids,mask
    def sample(self,task,task_step,micro):
        pool=self.pools['train',task]
        batch=self.cfg['batch_sizes'][task]
        start=(task_step*self.cfg['gradient_accumulation']+micro)*batch
        out=[]
        for p in range(start,start+batch):
            epoch,offset=divmod(p,len(pool))
            key=(task,epoch)
            if key not in self.permutations:
                self.permutations={k:v for k,v in self.permutations.items() if k[0]!=task}
                self.permutations[key]=np.random.default_rng(np.random.SeedSequence([self.cfg['seed'],self.cfg['tasks'].index(task),epoch])).permutation(len(pool))
            out.append(pool[self.permutations[key][offset]])
        return out
    def batch(self,indices,task,device='cuda'):
        ids,mask=self.pad([self.tokens[i] for i in indices],left=True)
        targets,tm=self.pad([self.targets[i] for i in indices])
        b=dict(ids=ids,text_mask=mask,target_ids=targets,target_mask=tm,
               labels=torch.tensor([self.rows[i]['labels'] for i in indices]),
               hr_features=torch.from_numpy(np.array(self.hr_features[indices],copy=True)),
               lr_features=torch.from_numpy(np.array(self.lr_features[indices],copy=True)))
        b={k:v.to(device) for k,v in b.items()}
        if task in ['sr','segmentation']:
            hr=torch.from_numpy(np.array(self.images[indices],copy=True)).to(device).float()[:,None]/255
            valid=torch.zeros_like(hr)
            for j,i in enumerate(indices):
                y,x,h,w=self.rows[i]['box']
                valid[j,:,y:y+h,x:x+w]=1
            if task=='sr':
                b.update(hr=hr,lr=downsample(hr,self.cfg['scale']),valid=valid)
            else:
                b.update(seg_image=F.interpolate(hr,(256,256),mode='bilinear',align_corners=False),
                    seg_mask=F.interpolate(valid,(256,256),mode='area'),
                    seg_probs=torch.from_numpy(np.array(self.seg_probs[indices],copy=True)).to(device).float())
        return b
