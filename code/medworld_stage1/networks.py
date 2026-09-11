"""Four readouts with shared multimodal state; SR has a separate LR state input."""
import copy
import bootstrap
from bootstrap import *
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import load_qwen,adapt,StateEncoder,ReportDecoder


class SlotQueries(nn.Module):
    def __init__(self,width,count,dim=128):
        super().__init__()
        self.norm=nn.LayerNorm(width)
        self.project=nn.Linear(width,dim)
        self.queries=nn.Parameter(torch.randn(count,dim)*.02)
        self.attn=nn.MultiheadAttention(dim,4,batch_first=True,dropout=0)
        self.normout=nn.LayerNorm(dim)
    def forward(self,s):
        z=self.project(self.norm(s.float()))
        q=self.queries[None].expand(len(s),-1,-1)
        out,_=self.attn(q,z,z,need_weights=False)
        return self.normout(q+out)


class Classification(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.read=SlotQueries(width,len(FINDINGS))
        self.out=nn.Linear(128,1)
    def forward(self,s):return self.out(self.read(s)).squeeze(-1)


class Segmentation(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.read=SlotQueries(width,3,64)
        self.image=nn.Sequential(nn.Conv2d(1,32,5,padding=2),nn.GroupNorm(4,32),nn.GELU(),
            nn.Conv2d(32,64,3,stride=2,padding=1),nn.GroupNorm(8,64),nn.GELU(),
            nn.Conv2d(64,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU())
        self.mask=nn.Linear(64,64)
        self.coarse=nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Linear(128,32*32))
    def forward(self,image,s):
        q=self.read(s)
        f=self.image(image)
        logits=torch.einsum('bcd,bdhw->bchw',self.mask(q),f)/8
        logits=F.interpolate(logits,image.shape[-2:],mode='bilinear',align_corners=False)
        coarse=self.coarse(q).reshape(len(s),3,32,32)
        return logits,coarse


class SRBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv2d(32,32,3,padding=1)
        self.c2=nn.Conv2d(32,32,3,padding=1)
        self.film=nn.Linear(128,64)
    def forward(self,x,z):
        scale,bias=self.film(z).chunk(2,-1)
        y=self.c1(x)*(1+.1*scale[:,:,None,None])+.1*bias[:,:,None,None]
        return x+.1*self.c2(F.gelu(y))


class SuperResolution(nn.Module):
    def __init__(self,width,scale):
        super().__init__()
        self.scale=scale
        self.read=SlotQueries(width,1)
        self.input=nn.Conv2d(1,32,3,padding=1)
        self.blocks=nn.ModuleList([SRBlock() for _ in range(6)])
        self.out=nn.Sequential(nn.Conv2d(32,scale*scale,3,padding=1),nn.PixelShuffle(scale))
    def forward(self,lr,s):
        z=self.read(s)[:,0]
        x=self.input(lr)
        for block in self.blocks:x=block(x,z)
        base=F.interpolate(lr,scale_factor=self.scale,mode='bicubic',align_corners=False)
        return base+.1*self.out(x)


def masked_classification(logits,labels,pos_weight):
    valid=(labels==0)|(labels==1)
    loss=F.binary_cross_entropy_with_logits(logits.float(),labels.clamp(0,1).float(),
                                            pos_weight=pos_weight,reduction='none')
    counts=valid.sum(0)
    perclass=(loss*valid).sum(0)/counts.clamp_min(1)
    return (perclass*(counts>0)).sum()/(counts>0).sum().clamp_min(1)


def segmentation_loss(logits,soft,valid):
    hard=(soft>.5).float()
    bce=F.binary_cross_entropy_with_logits(logits.float(),soft.float(),reduction='none')
    bce=(bce*valid).sum()/(valid.sum()*soft.shape[1]).clamp_min(1)
    p=logits.float().sigmoid()*valid
    target=hard*valid
    dice=1-((2*(p*target).sum((2,3))+1)/(p.sum((2,3))+target.sum((2,3))+1)).mean()
    return bce+dice


class FourTaskModel(nn.Module):
    def __init__(self,cfg,freeze_encoder=False):
        super().__init__()
        self.cfg=cfg
        self.tokenizer=AutoTokenizer.from_pretrained(cfg['qwen'],local_files_only=True)
        base=load_qwen(cfg)
        width=base.config.text_config.hidden_size
        assert width==1024 and cfg['slots']==8
        enc=base.model.language_model
        dec=copy.deepcopy(enc)
        del base
        self.encoder=StateEncoder(adapt(enc,cfg),cfg,width)
        self.diagnosis=ReportDecoder(adapt(dec,cfg),self.tokenizer,width)
        self.classification=Classification(width)
        self.segmentation=Segmentation(width)
        self.sr=SuperResolution(width,cfg['scale'])
        self.freeze_encoder=freeze_encoder
        if freeze_encoder:self.encoder.requires_grad_(False)
        for b in [self.encoder.backbone,self.diagnosis.backbone]:
            b.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        if freeze_encoder:self.encoder.backbone.gradient_checkpointing_disable()
    def train(self,mode=True):
        super().train(mode)
        if self.freeze_encoder:self.encoder.eval()
        return self
    def encode(self,b,task):
        features=b['lr_features'] if task=='sr' else b['hr_features']
        return self.encoder(features,b['ids'],b['text_mask'])
    def loss(self,b,task,pos_weight):
        s=self.encode(b,task)
        if task=='classification':
            loss=masked_classification(self.classification(s),b['labels'],pos_weight)
        elif task=='diagnosis':
            loss=self.diagnosis.loss(s,b['target_ids'],b['target_mask'])
        elif task=='segmentation':
            pred,coarse=self.segmentation(b['seg_image'],s)
            loss=segmentation_loss(pred,b['seg_probs'],b['seg_mask'])
            loss=loss+.2*segmentation_loss(coarse,F.interpolate(b['seg_probs'],(32,32),mode='area'),
                                          F.interpolate(b['seg_mask'],(32,32),mode='area'))
        else:
            pred=self.sr(b['lr'],s)
            loss=((pred.float()-b['hr']).square()*b['valid']).sum()/b['valid'].sum().clamp_min(1)
        return loss
    def compact_state(self):
        return {k:v.detach().cpu() for k,v in self.state_dict().items() if 'lora_' in k or '.backbone.' not in k}
    def load_compact(self,state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        assert not unexpected and not any('lora_' in k or '.backbone.' not in k for k in missing),(missing,unexpected)
