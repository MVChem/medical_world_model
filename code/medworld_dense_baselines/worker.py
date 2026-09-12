"""Recheck and claim idle devices immediately before a queued task starts."""
import os
import runpy
import subprocess
import sys
from pathlib import Path

if __name__=='__main__':
    selected={int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')}
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used',
                                 '--format=csv,noheader,nounits'],text=True,timeout=15)
    compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                     '--format=csv,noheader,nounits'],text=True,timeout=15)
    busy={line.split(',')[0].strip() for line in compute.splitlines() if ',' in line}
    for line in raw.splitlines():
        idx,uuid,mem=[x.strip() for x in line.split(',')]
        if int(idx) in selected and (int(mem)>=512 or uuid in busy):
            print('GPU_BUSY_AT_START: yielding to an existing process',flush=True)
            sys.exit(75)
    import torch
    # Make the claim visible in nvidia-smi during CPU-side checkpoint loading.
    reserved=[torch.empty(16*1024*1024,dtype=torch.uint8,device=f'cuda:{i}') for i in range(len(selected))]
    target=Path(sys.argv[1]).resolve();sys.argv=sys.argv[1:]
    sys.path.insert(0,str(target.parent))
    runpy.run_path(str(target),run_name='__main__')
