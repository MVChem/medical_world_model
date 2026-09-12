"""Identical per-model heads. Image branch has three explicitly separate variants."""
import torch
from torch import nn
import torch.nn.functional as F

def block(a,b,stride=1):
    return nn.Sequential(nn.Conv2d(a,b,3,stride=stride,padding=1),nn.GroupNorm(4,b),nn.GELU())

class DenseHead(nn.Module):
    def __init__(self,task,variant):
        super().__init__()
        self.task,self.variant=task,variant
        if variant=='image':
            self.stem=nn.Sequential(block(1,32,2),block(32,64,2))
        elif variant=='vjepa_adapter':
            # Newly initialized, task-trained adapter; no trained world-model checkpoint.
            self.adapter=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,128),nn.GELU(),nn.Linear(128,64))
        elif variant!='vjepa':raise ValueError(variant)
        self.hidden=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,64),nn.GELU())
        self.fuse=block(128,64)
        layers=[]
        a=64
        widths=[48,32,16]+([8] if task=='sr' else [])
        for b in widths:
            layers.extend([nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),block(a,b)])
            a=b
        layers.append(nn.Conv2d(a,1 if task=='sr' else 3,1))
        self.decode=nn.Sequential(*layers)

    def forward(self,visual,hidden):
        if self.variant=='image':
            z=self.stem(visual)
        else:
            if self.variant=='vjepa_adapter':z=self.adapter(visual.float())
            else:
                # Parameter-free branch: standardized native features, interleaved channels.
                z=F.layer_norm(visual.float(),(768,))[:,:,6::12]
            z=z.transpose(1,2).reshape(-1,64,24,24)
        z=F.interpolate(z,(32,32),mode='bilinear',align_corners=False)
        h=self.hidden(hidden.float()).transpose(1,2).reshape(-1,64,8,8)
        h=F.interpolate(h,(32,32),mode='bilinear',align_corners=False)
        logits=self.decode(self.fuse(torch.cat([z,h],1)))
        if self.task=='sr':
            if self.variant=='image':
                base=F.interpolate(visual,scale_factor=4,mode='bicubic',align_corners=False)
                return base+.1*logits
            return logits.sigmoid()
        return logits

class GroundingHead(nn.Module):
    def __init__(self,nqueries):
        super().__init__()
        self.image=nn.Sequential(block(1,32,2),block(32,64,2),nn.AdaptiveAvgPool2d((4,4)))
        self.hidden=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,128),nn.GELU())
        # Closed anatomical query vocabulary, same one-hot embedding table for all VLMs.
        self.query=nn.Embedding(nqueries,64)
        self.out=nn.Sequential(nn.Linear(64*16+128+64,256),nn.GELU(),nn.Linear(256,4))
    def forward(self,image,hidden,query):
        z=torch.cat([self.image(image).flatten(1),self.hidden(hidden.float()).mean(1),self.query(query)],1)
        raw=self.out(z).sigmoid()
        lo=raw[:,:2];hi=lo+(1-lo)*raw[:,2:]
        return torch.cat([lo,hi],1)

def box_iou(a,b):
    wh=(torch.minimum(a[:,2:],b[:,2:])-torch.maximum(a[:,:2],b[:,:2])).clamp_min(0)
    inter=wh.prod(1)
    area=(a[:,2:]-a[:,:2]).clamp_min(0).prod(1)+(b[:,2:]-b[:,:2]).clamp_min(0).prod(1)
    return inter/(area-inter).clamp_min(1e-8)

def segmentation_loss(logits,target,mask):
    bce=F.binary_cross_entropy_with_logits(logits.float(),target,reduction='none')
    bce=((bce*mask).sum((1,2,3))/(mask.sum((1,2,3))*target.shape[1]).clamp_min(1)).mean()
    p=logits.float().sigmoid()*mask;t=(target>.5).float()*mask
    dice=(2*(p*t).sum((2,3))+1)/(p.sum((2,3))+t.sum((2,3))+1)
    return bce+1-dice.mean()
