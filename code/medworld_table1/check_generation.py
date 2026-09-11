"""Compare cached decoding with full-prefix recomputation on a real checkpoint."""
import argparse
import os
from pathlib import Path

from common import atomic_json, digest, seed_all
from data import Corpus
from model import MedWorld, hidden
import torch
import torch.nn.functional as F


def check(a):
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    cfg=checkpoint['config']
    seed_all(cfg['seed'])
    assert digest(Path(cfg['cache'])/'observations.jsonl') == checkpoint['signatures']['observations.jsonl']
    model=MedWorld(cfg)
    if checkpoint['stage']==2:model.begin_stage2()
    model.load_compact(checkpoint['model'])
    model.eval().cuda()
    corpus=Corpus(cfg,model.tokenizer)
    batch=corpus.batch(corpus.pairs['validate'][:2])
    checks=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        state=model.state(batch)
        if checkpoint['stage']==2:state=model.world(state,batch['horizon'])
        embed=model.decoder.backbone.get_input_embeddings()
        full=model.decoder.prefix(state)
        mask=torch.ones(full.shape[:2],dtype=torch.long,device='cuda')
        cached=hidden(model.decoder.backbone,full,mask,use_cache=True)
        for step in range(a.steps):
            recomputed=hidden(model.decoder.backbone,full,mask,use_cache=False)
            cached_logits=F.linear(cached.last_hidden_state[:,-1].to(embed.weight.dtype),embed.weight).float()
            full_logits=F.linear(recomputed.last_hidden_state[:,-1].to(embed.weight.dtype),embed.weight).float()
            ids=cached_logits.argmax(-1)
            top=full_logits.topk(2,dim=-1).values
            checks.append(dict(step=step,max_absolute_logit_difference=float((cached_logits-full_logits).abs().max()),
                mean_absolute_logit_difference=float((cached_logits-full_logits).abs().mean()),
                same_argmax=(ids==full_logits.argmax(-1)).tolist(),
                full_argmax_margin=(top[:,0]-top[:,1]).tolist()))
            mask=torch.cat([mask,torch.ones_like(mask[:,:1])],1)
            next_embed=embed(ids[:,None])
            full=torch.cat([full,next_embed],1)
            cached=hidden(model.decoder.backbone,next_embed,mask,use_cache=True,past=cached.past_key_values)
    result=dict(checkpoint=str(a.checkpoint),sha256=digest(a.checkpoint),patients=2,steps=a.steps,
        tied_output_embeddings=True,eos_token_id=model.tokenizer.eos_token_id,
        same_argmax_count=sum(sum(r['same_argmax']) for r in checks),comparisons=2*a.steps,
        maximum_logit_difference=max(r['max_absolute_logit_difference'] for r in checks),checks=checks,
        interpretation='BF16 numerical differences allowed; both paths use identical state, generated token prefix, mask and positions. This check does not establish clinical quality.')
    atomic_json(a.out,result)
    print({k:v for k,v in result.items() if k!='checks'},flush=True)


if __name__=='__main__':
    os.umask(0o077)
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=32)
    check(p.parse_args())
