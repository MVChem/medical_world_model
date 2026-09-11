"""Frozen image features and CXAS probabilities; no trained student dependencies."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import bootstrap
from bootstrap import *
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def downsample(x, scale):
    return F.interpolate(x, scale_factor=1/scale, mode='bicubic', align_corners=False, antialias=True).clamp(0,1)


def normalized(x):
    mean = x.new_tensor([.485,.456,.406])[None,:,None,None]
    std = x.new_tensor([.229,.224,.225])[None,:,None,None]
    return (x.expand(-1,3,-1,-1)-mean)/std


class HRImages(Dataset):
    def __init__(self, cache):
        self.images=np.load(cache/'images.npy',mmap_mode='r')
        self.n=json.loads((cache/'manifest.json').read_text())['valid_count']
    def __len__(self):return self.n
    def __getitem__(self,i):return i,torch.from_numpy(np.array(self.images[i],copy=True))[None]


def feature_cache(cache, batch_size):
    if (cache/'features.json').exists():return
    seed_all(20260910)
    manifest=json.loads((cache/'manifest.json').read_text())
    sys.path.insert(0,str(PROJECT/'code/vjepa2'))
    from app.vjepa_2_1.models.vision_transformer import vit_base
    model=vit_base(img_size=(384,384),patch_size=16,num_frames=64,tubelet_size=2,use_sdpa=True,
                   use_rope=True,img_temporal_dim_size=1,interpolate_rope=True)
    checkpoint=PILOT/'weights/vjepa2_1_vitb.pt'
    weights=torch.load(checkpoint,map_location='cpu',weights_only=True)['ema_encoder']
    model.load_state_dict({k.replace('module.','').replace('backbone.',''):v for k,v in weights.items()},strict=True)
    del weights
    model.requires_grad_(False).eval().to('cuda',dtype=torch.bfloat16)
    shape=(manifest['valid_count'],64,768)
    arrays={k:np.lib.format.open_memmap(cache/f'{k}_features.tmp.npy',mode='w+',dtype=np.float16,shape=shape)
            for k in ['hr','lr']}
    loader=DataLoader(HRImages(cache),batch_size=batch_size,num_workers=4,pin_memory=True)
    started=time.time()
    with torch.inference_mode():
        for step,(idx,images) in enumerate(loader):
            hr=images.to('cuda',dtype=torch.float32)/255
            lr=downsample(hr,manifest['scale'])
            for name,x in [('hr',hr),('lr',lr)]:
                x=normalized(F.interpolate(x,(384,384),mode='bicubic',align_corners=False,antialias=True).clamp(0,1))
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    tokens=model(x.unsqueeze(2))
                assert tokens.shape[1:]==(576,768)
                tokens=F.adaptive_avg_pool2d(tokens.transpose(1,2).reshape(-1,768,24,24),(8,8)).flatten(2).transpose(1,2)
                assert torch.isfinite(tokens).all()
                arrays[name][idx.numpy()]=tokens.float().cpu().numpy()
            if step%25==0:
                done=int(idx[-1])+1
                atomic_json(cache/'features_progress.json',dict(done=done,total=shape[0],seconds=time.time()-started))
                print('features',done,shape[0],'seconds',round(time.time()-started),flush=True)
    for name,a in arrays.items():
        a.flush()
        os.replace(cache/f'{name}_features.tmp.npy',cache/f'{name}_features.npy')
    atomic_json(cache/'features.json',dict(shape=shape,checkpoint_sha256=digest(checkpoint),
        observations_sha256=manifest['observations_sha256'],scale=manifest['scale'],
        lr_source='HR antialiased bicubic reduction ONLY; never reuse HR image features',
        image_encoder='frozen V-JEPA2.1 ViT-B EMA 384, 24x24 pooled to 8x8'))


class TeacherImages(Dataset):
    def __init__(self,rows):self.rows=[r for r in rows if r['tasks']['segmentation']]
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(r['image']) as im:a=np.array(im.convert('L'),copy=True)
        x=torch.from_numpy(a)[None,None].float()/255
        # Same nearest interpolation convention as CXAS FileLoader.load_image.
        x=F.interpolate(x,(512,512),mode='nearest')
        return r['index'],normalized(x)[0],torch.tensor(r['box'])


def segmentation_cache(cache,batch_size):
    if (cache/'segmentation.json').exists():return
    seed_all(20260910)
    # Import the original architecture directly: CXAS's top-level package also
    # imports optional DICOM export/plotting dependencies, unused for JPEG inference.
    import importlib
    import types
    cxas_root=PROJECT/'code/ChestXRayAnatomySegmentation/cxas'
    package=types.ModuleType('stage1_cxas_unet')
    package.__path__=[str(cxas_root/'models/UNet')]
    sys.modules[package.__name__]=package
    BackboneUNet=importlib.import_module('stage1_cxas_unet.backbone_unet').BackboneUNet
    id2label_dict=json.loads((cxas_root/'data/paxray_labels.json').read_text())['label_dict']
    teacher=BackboneUNet('UNet_ResNet50_default',len(id2label_dict))
    checkpoint=torch.load(Path.home()/'.cxas/weights/UNet_ResNet50_default.pth',map_location='cpu',weights_only=False)
    teacher.load_state_dict({k.removeprefix('module.'):v for k,v in checkpoint['model'].items()},strict=True)
    del checkpoint
    teacher.requires_grad_(False).eval().cuda()
    channels=[next(int(k) for k,v in id2label_dict.items() if v==name) for name in ORGANS]
    rows=load_rows(cache/'observations.jsonl')
    probs=np.lib.format.open_memmap(cache/'seg_probs.tmp.npy',mode='w+',dtype=np.float16,shape=(len(rows),3,256,256))
    valid=np.zeros(len(rows),dtype=bool)
    rejected=[]
    loader=DataLoader(TeacherImages(rows),batch_size=batch_size,num_workers=4,pin_memory=True)
    started=time.time()
    with torch.inference_mode():
        for step,(indices,x,boxes) in enumerate(loader):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=teacher._forward(x.to('cuda'))['logits'][:,channels]
            p=logits.float().sigmoid()
            for i,box,pi in zip(indices.tolist(),boxes.tolist(),p):
                y,x0,h,w=box
                # Map square teacher coordinates back to aspect-preserved student coordinates.
                y,x0,h,w=[v//2 for v in [y,x0,h,w]]
                rect=F.interpolate(pi[None],(h,w),mode='bilinear',align_corners=False)[0]
                areas=(rect>.5).float().mean((1,2))
                ok=bool(torch.isfinite(rect).all() and ((areas>.002)&(areas<.85)).all())
                out=np.zeros((3,256,256),dtype=np.float16)
                out[:,y:y+h,x0:x0+w]=rect.cpu().numpy().astype(np.float16)
                probs[i]=out
                valid[i]=ok
                if not ok:rejected.append(dict(index=i,id=rows[i]['id'],areas=areas.cpu().tolist()))
            if step%25==0:
                done=min((step+1)*batch_size,len(loader.dataset))
                atomic_json(cache/'segmentation_progress.json',dict(done=done,total=len(loader.dataset),seconds=time.time()-started))
                print('teacher',done,len(loader.dataset),'seconds',round(time.time()-started),flush=True)
    probs.flush()
    del probs
    os.replace(cache/'seg_probs.tmp.npy',cache/'seg_probs.npy')
    np.save(cache/'seg_valid.npy',valid)
    write_rows(cache/'seg_rejected.jsonl',rejected)
    atomic_json(cache/'segmentation.json',dict(teacher='CXAS UNet_ResNet50_default',organs=ORGANS,
        channels=channels,soft_probabilities=True,dtype='float16',valid=int(valid.sum()),rejected=len(rejected),
        checkpoint_sha256=digest(Path.home()/'.cxas/weights/UNet_ResNet50_default.pth'),
        observations_sha256=digest(cache/'observations.jsonl'),teacher_input='ImageNet normalization, nearest square 512 (CXAS protocol)',
        strict_checkpoint_load=True,architecture_hashes={p.name:digest(p) for p in (cxas_root/'models/UNet').glob('*.py')},
        score_scope='teacher agreement, not human ground truth'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['features','segmentation'])
    p.add_argument('--cache',type=Path,default=ROOT/'data/overnight_20260910')
    p.add_argument('--batch-size',type=int)
    a=p.parse_args()
    (feature_cache if a.mode=='features' else segmentation_cache)(a.cache,a.batch_size or (24 if a.mode=='features' else 2))
