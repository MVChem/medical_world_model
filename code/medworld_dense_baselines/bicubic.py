"""Parameter-free SR reference on exactly the same test images."""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity
from common import *

def evaluate(run):
    torch.set_num_threads(4)
    data=run/'data';rows=read_rows(data/'observations.jsonl')
    images=np.load(data/'images.npy',mmap_mode='r');lr=np.load(data/'lr_images.npy',mmap_mode='r')
    records=[]
    for r in rows:
        if r['split']!='test' or 'sr' not in r['tasks']:continue
        i=r['index'];y,x,h,w=r['box']
        low=torch.from_numpy(np.array(lr[i],copy=True))[None,None].float()/255
        pred=F.interpolate(low,scale_factor=4,mode='bicubic',align_corners=False).clamp(0,1)[0,0].numpy()
        a=pred[y:y+h,x:x+w];t=images[i,y:y+h,x:x+w].astype(np.float32)/255
        mse=float(np.mean((a-t)**2))
        records.append(dict(id=r['id'],subject_id=r['subject_id'],
            psnr=float(-10*np.log10(max(mse,1e-12))),ssim=float(structural_similarity(t,a,data_range=1.))))
    write_rows(run/'bicubic_per_sample.jsonl',records)
    atomic(run/'bicubic_metrics.json',dict(n=len(records),psnr=float(np.mean([r['psnr'] for r in records])),
        ssim=float(np.mean([r['ssim'] for r in records])),cohort_sha256=digest(data/'observations.jsonl'),
        scope='same fixed SR test cohort, no training, exact shared LR cache'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT_RUN)
    a=p.parse_args();evaluate(a.run)
