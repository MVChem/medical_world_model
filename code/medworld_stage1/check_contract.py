"""Small meaningful checks: label masking, gradient routes, LR contract, checkpoint reload."""
import argparse
import json
import tempfile
from pathlib import Path
import bootstrap
from bootstrap import *
import torch
from networks import FourTaskModel,masked_classification,segmentation_loss,Segmentation,SuperResolution
from cache import downsample


def cpu_checks():
    seed_all(1)
    logits=torch.randn(2,13,requires_grad=True)
    labels=torch.full((2,13),-2)
    labels[0,0]=1;labels[1,0]=0;labels[0,1]=-1
    loss=masked_classification(logits,labels,torch.ones(13))
    loss.backward()
    assert logits.grad[:,1:].abs().sum()==0
    assert logits.grad[:,0].abs().sum()>0
    pred=torch.randn(2,3,32,32,requires_grad=True)
    mask=torch.zeros(2,1,32,32);mask[:,:,4:28,8:24]=1
    target=torch.rand_like(pred)
    segloss=segmentation_loss(pred,target,mask)
    segloss.backward()
    assert (pred.grad*(1-mask)).abs().sum()==0,'padding must have no segmentation gradient'
    perturbed=target*mask+(1-target)*(1-mask)
    assert torch.allclose(segloss,segmentation_loss(pred,perturbed,mask))
    slots=torch.randn(2,8,1024,requires_grad=True)
    head=Segmentation(1024)
    dense,coarse=head(torch.rand(2,1,64,64),slots)
    (dense.square().mean()+coarse.square().mean()).backward()
    assert (slots.grad.norm(dim=-1)>0).all()
    hr=torch.rand(2,1,64,96)
    for scale in (2,4):
        slots=torch.randn(2,8,1024,requires_grad=True)
        lr=downsample(hr,scale)
        assert lr.shape[-2:]==(64//scale,96//scale)
        sr=SuperResolution(1024,scale)
        result=sr(lr,slots)
        assert result.shape==hr.shape
        (result-hr).square().mean().backward()
        assert (slots.grad.norm(dim=-1)>0).all()
    print('CPU contracts passed',flush=True)


def gpu_checks(out):
    seed_all(20260910)
    cfg=json.loads((ROOT/'config.json').read_text())
    model=FourTaskModel(cfg).cuda().train()
    token=model.tokenizer('FINDINGS: Small left pleural effusion. IMPRESSION: Left pleural effusion.',return_tensors='pt',add_special_tokens=False)
    ids=token['input_ids'].cuda().expand(2,-1)
    hr=torch.rand(2,1,128,128,device='cuda')
    b=dict(ids=ids,text_mask=torch.ones_like(ids),target_ids=ids,target_mask=torch.ones_like(ids),
        labels=torch.randint(0,2,(2,13),device='cuda'),hr_features=torch.randn(2,64,768,device='cuda'),
        lr_features=torch.randn(2,64,768,device='cuda'),hr=hr,lr=downsample(hr,cfg['scale']),valid=torch.ones_like(hr),
        seg_image=hr,seg_probs=torch.rand(2,3,128,128,device='cuda'),seg_mask=torch.ones_like(hr))
    results={}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5)
    for task in cfg['tasks']:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.loss(b,task,torch.ones(13,device='cuda'))
        loss.backward()
        grad=float(model.encoder.slots.grad.float().norm())
        assert torch.isfinite(loss) and grad>0
        optimizer.step()
        results[task]=dict(loss=float(loss.detach()),slots_grad_norm=grad)
        print('GPU contract',task,results[task],flush=True)
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        a=model.encode(b,'sr')
        b2=dict(b,hr_features=b['hr_features']+100,hr=b['hr']+100)
        z=model.encode(b2,'sr')
        assert torch.equal(a,z),'SR encoder depends on HR inputs'
        logits=model.classification(model.encode(b,'classification')).float().cpu()
    state=model.compact_state()
    model.load_compact(state)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        restored=model.classification(model.encode(b,'classification')).float().cpu()
    assert torch.equal(logits,restored)
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'contracts.json',dict(cpu=True,gpu=results,sr_hr_input_invariance=True,compact_reload=True,
        peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true');p.add_argument('--out',type=Path,default=ROOT/'runs/overnight_20260910/checks')
    a=p.parse_args();cpu_checks()
    if a.gpu:gpu_checks(a.out)
