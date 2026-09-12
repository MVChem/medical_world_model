"""Cache native 24x24 V-JEPA features, including a distinct LR-only SR cache."""
import argparse
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from common import *
from features import downsample

def extract(run,batch_size=12):
    torch.set_num_threads(4)
    data=run/'data'
    if (data/'vjepa_complete.json').exists():return
    sys.path.insert(0,str(PROJECT/'code/vjepa2'))
    from app.vjepa_2_1.models.vision_transformer import vit_base
    model=vit_base(img_size=(384,384),patch_size=16,num_frames=64,tubelet_size=2,use_sdpa=True,
        use_rope=True,img_temporal_dim_size=1,interpolate_rope=True)
    checkpoint=PROJECT/'code/medworld_table1/weights/vjepa2_1_vitb.pt'
    w=torch.load(checkpoint,map_location='cpu',weights_only=True)['ema_encoder']
    model.load_state_dict({k.replace('module.','').replace('backbone.',''):v for k,v in w.items()},strict=True)
    del w
    model.requires_grad_(False).eval().cuda().to(torch.bfloat16)
    rows=read_rows(data/'observations.jsonl');n=len(rows)
    images=np.load(data/'images.npy',mmap_mode='r')
    lr_images=np.load(data/'lr_images.npy',mmap_mode='r')
    done=np.load(data/'vjepa_done.npy') if (data/'vjepa_done.npy').exists() else np.zeros((n,2),bool)
    arrays={name:np.lib.format.open_memmap(data/f'vjepa_{name}.npy',mode='r+' if (data/f'vjepa_{name}.npy').exists() else 'w+',
        dtype=np.float16,shape=(n,576,768)) for name in ['hr','lr']}
    started=time.time()
    for j,name in enumerate(['hr','lr']):
        ids=[i for i,r in enumerate(rows) if not done[i,j] and (name=='hr' or 'sr' in r['tasks'])]
        for start in range(0,len(ids),batch_size):
            ii=ids[start:start+batch_size]
            x=torch.from_numpy(np.array(lr_images[ii] if name=='lr' else images[ii],copy=True))[:,None].cuda().float()/255
            x=F.interpolate(x,(384,384),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
            mean=x.new_tensor([.485,.456,.406])[None,:,None,None]
            std=x.new_tensor([.229,.224,.225])[None,:,None,None]
            x=(x.expand(-1,3,-1,-1)-mean)/std
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                tokens=model(x.unsqueeze(2))
            assert tokens.shape[1:]==(576,768) and torch.isfinite(tokens).all()
            arrays[name][ii]=tokens.float().cpu().numpy().astype(np.float16)
            arrays[name].flush();done[ii,j]=True
            np.save(data/'vjepa_done.tmp.npy',done);os.replace(data/'vjepa_done.tmp.npy',data/'vjepa_done.npy')
            if start%(batch_size*20)==0:
                atomic(data/'vjepa_progress.json',dict(branch=name,done=int(done.sum()),seconds=time.time()-started))
                print('vjepa',name,start+len(ii),'/',len(ids),flush=True)
    atomic(data/'vjepa_complete.json',dict(shape=[n,576,768],checkpoint_sha256=digest(checkpoint),
        cohort_sha256=digest(data/'observations.jsonl'),frozen=True,hr_from='original image canvas',
        lr_image_sha256=digest(data/'lr_images.npy'),
        lr_from='uint8-rounded antialiased bicubic 4x LR ONLY',pooling='none; native spatial 24x24 grid'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT_RUN)
    p.add_argument('--batch-size',type=int,default=12)
    a=p.parse_args();extract(a.run,a.batch_size)
