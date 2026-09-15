"""Score one completed model with official GREEN on the allocated physical GPU."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--model',required=True)
    args=p.parse_args();run=args.run.resolve()
    allocated=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(allocated)!=1 or not allocated[0].isdigit():raise ValueError('One physical GPU must be assigned by queue')
    single=run/'green_single_model'/args.model;single.mkdir(parents=True,exist_ok=True)
    for name in ['cohort',args.model]:
        dest=single/name
        if not dest.exists():dest.symlink_to(run/name,target_is_directory=True)
    for name in ['protocol.json','vocabulary.json']:
        shutil.copy2(run/name,single/name)
    model=next(m for m in json.loads((run/'models.json').read_text()) if m['id']==args.model)
    (single/'models.json').write_text(json.dumps([model],indent=2)+'\n')
    source=Path(__file__).resolve().parent.parent/'medworld_baselines/green_eval.py'
    subprocess.run([sys.executable,str(source),'--run',str(single),'--gpu',allocated[0]],check=True)

if __name__=='__main__':main()
