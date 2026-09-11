"""Resumable full local build. No GPU/model server is needed for record linkage."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from common import ROOT,DEFAULT_CXR,DEFAULT_IV


def main(args):
    os.umask(0o077)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    def command(script,*extra):
        return [sys.executable,str(ROOT/script),'--out',str(out),*map(str,extra)]
    subprocess.run(command('build.py','--cxr',args.cxr,'--iv',args.iv),check=True)
    processes=[]
    try:
        processes.append(subprocess.Popen(command('link_iv.py','--iv',args.iv,'--workers',args.table_workers)))
        processes.append(subprocess.Popen(command('qc_images.py','--workers',args.image_workers)))
        codes=[p.wait() for p in processes]
        if any(codes):raise RuntimeError(f'IV/image stages failed: {codes}; completed files can be resumed.')
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
    for script in ['audit_links.py','availability.py','finalize.py','review.py','preview.py','validate_outputs.py']:
        subprocess.run(command(script),check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'runs/full_20260909')
    p.add_argument('--cxr',type=Path,default=DEFAULT_CXR)
    p.add_argument('--iv',type=Path,default=DEFAULT_IV)
    p.add_argument('--table-workers',type=int,default=3)
    p.add_argument('--image-workers',type=int,default=16)
    main(p.parse_args())
