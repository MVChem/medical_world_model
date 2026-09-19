"""A fixed bicubic SR reference; not a native-Qwen dense prediction."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch.nn.functional as F
from ..config import load_config
from ..datasets import UnifiedData
from .super_resolution import super_resolution_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    data=UnifiedData(load_config(a.config));scores=[]
    for i in range(len(data.rows('sr','test'))):
        b=data.batch('sr','test',[i]);pred=F.interpolate(b['pixels'],(512,512),mode='bicubic',align_corners=False)
        scores.append(super_resolution_metrics(pred,b['targets'],b['mask']))
    Path(a.out).write_text(json.dumps({'method':'bicubic, no learning; separate from Qwen baseline','n':len(scores),
        'data_fingerprint':data.fingerprint,**{k:float(np.mean([s[k] for s in scores])) for k in ('psnr','ssim')}},indent=2)+'\n')


if __name__=='__main__':main()
