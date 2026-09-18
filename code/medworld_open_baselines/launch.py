"""Freeze source and launch a separate immutable comparator queue manifest."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--plan',type=Path,required=True)
    args=p.parse_args();project=args.project.resolve();run=args.run.resolve()
    root=run/('scheduler_'+args.name);root.mkdir(parents=True,exist_ok=True)
    if (root/'launch.json').exists():raise ValueError('Already launched; use frozen launch.json argv to resume')
    snapshot=root/'source';snapshot.mkdir(parents=True,exist_ok=True)
    base=project/'code/medworld_open_baselines'
    modules=['medworld_open_baselines','medworld_table1','medworld_common','medworld_stage1','medworld_baselines','medworld_dense_baselines']
    for name in modules:
        original=project/'code'/name;target=snapshot/'code'/name;target.mkdir(parents=True,exist_ok=True)
        for f in original.glob('*.py'):shutil.copy2(f,target/f.name)
        for asset in ['weights','vendor','metric_vendor','data','assets','configs']:
            if (original/asset).is_dir() and not (target/asset).exists():
                (target/asset).symlink_to((original/asset).resolve(),target_is_directory=True)
        if name=='medworld_open_baselines':
            for folder in original.glob('*_protocol'):
                shutil.copytree(folder,target/folder.name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    source=snapshot/'code/medworld_open_baselines'
    for folder in base.iterdir():
        if folder.is_dir() and (folder.name.endswith(('_vendor','_weights','_assets')) or folder.name=='third_party'):
            (source/folder.name).symlink_to(folder.resolve(),target_is_directory=True)
    shutil.copy2(project/'code/medworld_common/overnight_queue.py',source/'executor.py')
    plan=json.loads(args.plan.read_text())
    for job in plan['jobs']:
        # Freeze script argv only. Runtime cohorts, pretrained assets and output
        # paths stay exactly as declared in the reviewable plan.
        replaced=[]
        for value in job['argv']:
            candidate=snapshot/Path(value).relative_to(project) if value.endswith('.py') and value.startswith(str(project/'code')+'/') else None
            replaced.append(str(candidate) if candidate and candidate.is_file() else value)
        job['argv']=replaced
        job.setdefault('env',{}).update(MEDWORLD_PROJECT=str(project))
    plan['report_commands']=[[sys.executable,str(source/'report.py'),'--run',str(run)]]
    (root/'requested_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    manifest={str(f.relative_to(snapshot)):hashlib.sha256(f.read_bytes()).hexdigest() for f in snapshot.rglob('*.py')}
    (root/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    argv=[sys.executable,'-u',str(source/'baseline_queue.py'),'run','--run',str(root),
          '--plan',str(root/'requested_plan.json'),'--max-gpus','6','--gpu-order','1,2,3,6,7,0']
    env={**os.environ,'MEDWORLD_PROJECT':str(project),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
    with (root/'coordinator.log').open('a') as log:
        process=subprocess.Popen(argv,cwd=project,env=env,stdout=log,stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL,start_new_session=True)
    (root/'launch.json').write_text(json.dumps(dict(pid=process.pid,argv=argv,started=time.time(),
        policy='all idle GPUs, shared UUID locks, no foreign preemption, explicit per-job training budget'),indent=2)+'\n')
    print(json.dumps(dict(pid=process.pid,run=str(root),jobs=len(plan['jobs']))))

if __name__=='__main__':main()
