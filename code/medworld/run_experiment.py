"""Durable training -> full held-out evaluation -> native-Qwen comparison pipeline."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .config import PROJECT, load_config
from .launch_distributed import gpu_status, atomic


def registry(run, state, outcome=None):
    root=PROJECT/'experiments';root.mkdir(exist_ok=True)
    with (root/'.registry.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=root/'registry.json'
        entries=json.loads(path.read_text()) if path.exists() else []
        identity=run.name;relative=str(run.relative_to(PROJECT))
        entries=[e for e in entries if e['id']!=identity]
        if outcome is None:
            entries.append({'id':identity,'summary':'Qwen 0.8B: four-GPU eight-hour two-stage training, then matched downstream tests.','status':state,
                            'observed_at':datetime.now().astimezone().isoformat(),'run':relative,'live_status':relative+'/pipeline_status.json'})
        else:
            history=root/'README.md'
            row=f"| {datetime.now().astimezone().date()} | `{identity}` | {outcome} | [Run](../{relative}/) |\n"
            history.write_text(history.read_text().rstrip()+'\n'+row)
        atomic(path,entries)


def schedule(jobs, gpus, run, env):
    waiting=list(jobs);active={};finished=set();last_write=0
    try:
        while waiting or active:
            for gpu,(job,process,handle) in list(active.items()):
                code=process.poll()
                if code is not None:
                    handle.close();del active[gpu]
                    if code:raise RuntimeError(f"Evaluation {job['id']} exited {code}; see {job['log']}")
                    finished.add(job['id'])
            for gpu in gpus:
                if gpu in active:continue
                ready=next((j for j in waiting if set(j.get('after',[]))<=finished),None)
                if ready is None:continue
                status=gpu_status(gpu)
                if int(status['memory.used'])>1024 or int(status['utilization.gpu'])>5:continue
                log=run/ready['log'];log.parent.mkdir(parents=True,exist_ok=True);handle=log.open('a')
                command=[sys.executable,'-m',ready['module'],*ready['args'],'--gpu',gpu]
                process=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=handle,stderr=subprocess.STDOUT)
                active[gpu]=(ready,process,handle);waiting.remove(ready)
            if time.time()-last_write>15:
                atomic(run/'pipeline_status.json',{'phase':'evaluating','finished_jobs':sorted(finished),
                       'active_jobs':{gpu:{'id':j['id'],'pid':p.pid} for gpu,(j,p,h) in active.items()},
                       'pending_jobs':[j['id'] for j in waiting],'heartbeat_unix':time.time()})
                last_write=time.time()
            time.sleep(3)
    finally:
        for job,process,handle in active.values():
            if process.poll() is None:process.terminate()
        for job,process,handle in active.values():
            try:process.wait(timeout=60)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            handle.close()


def evaluation_jobs(run):
    jobs=[]
    # Start the two longer current-report jobs first, then distribute other tasks.
    task_order=[(stage,'report','test') for stage in ('stage1','stage2')]
    task_order += [(stage,task,split) for task,split in [('classification','test'),('segmentation','test'),('sr','test'),('segmentation','human_test')] for stage in ('stage1','stage2')]
    task_order += [('stage2','temporal','test')]
    for stage,task,split in task_order:
        name=task if split=='test' else 'segmentation_human';out=run/'evaluation'/stage/name
        jobs.append({'id':stage+'/'+name,'module':'medworld.evaluation.evaluate','log':f'evaluation/{stage}/{name}.log',
                     'args':['--checkpoint',str(run/(stage+'.pt')),'--out',str(out),'--task',task,'--split',split,'--max-new-tokens','384']})
    for stage in ('stage1','stage2'):
        root=run/'evaluation'/stage/'report'
        jobs.append({'id':stage+'/clinical_report','module':'medworld.evaluation.clinical_report',
                     'log':f'evaluation/{stage}/clinical_report.log','after':[stage+'/report'],
                     'args':['--predictions',str(root/'report.jsonl'),'--out',str(root/'clinical.json'),'--config',str(run/'config.json')]})
    return jobs


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--gpus',default='0,1,2,3');p.add_argument('--baseline-audit',required=True)
    a=p.parse_args();run=Path(a.out).resolve();cfg=load_config(a.config);gpus=a.gpus.split(',')
    if len(gpus)>4 or len(gpus)!=len(set(gpus)):raise ValueError('Use at most four distinct GPUs')
    audit=json.loads(Path(a.baseline_audit).read_text())
    if audit['status']!='passed' or 'native_report_rescore' not in audit:
        raise ValueError('A passed raw-Qwen audit and current-version native report re-score are required')
    if run.exists() and any(run.iterdir()):raise ValueError('Choose a fresh run directory')
    jobs=evaluation_jobs(run);child=None;outcome=None
    def interrupted(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    registry(run,'initializing')
    env=dict(os.environ,MEDWORLD_PROJECT_ROOT=str(PROJECT),PYTHONPATH=str(PROJECT/'code'),
             PYTORCH_ALLOC_CONF='expandable_segments:True',PYTHONUNBUFFERED='1')
    try:
        command=[sys.executable,'-m','medworld.launch_distributed','--config',str(Path(a.config).resolve()),'--gpus',a.gpus,'--out',str(run)]
        child=subprocess.Popen(command,cwd=PROJECT,env=env)
        initialized=False
        while child.poll() is None:
            if (run/'source_manifest.json').exists() and not initialized:
                atomic(run/'baseline_audit.json',audit)
                atomic(run/'evaluation_plan.json',{'jobs':jobs,'gpus':gpus,'baseline':'native Qwen3.5-0.8B, no project training',
                      'generation_tokens':384,'train_hours':cfg['total_hours'],'stage1_hours':cfg['stage1_hours'],
                      'evaluation_after_training':True,'test_selection':'Final stage1.pt and stage2.pt; no best-on-test selection'})
                initialized=True;registry(run,'training')
            if initialized:atomic(run/'pipeline_status.json',{'phase':'training','launcher_pid':child.pid,'heartbeat_unix':time.time()})
            time.sleep(15)
        if child.returncode:raise RuntimeError(f'Training launcher exited {child.returncode}')
        state=json.loads((run/'status.json').read_text())
        if state.get('stopped') or state.get('stage')!='stage2' or not state.get('stage_complete'):
            raise RuntimeError('Training stopped before completing both stages')
        frozen=dict(env,PYTHONPATH=str(run/'source'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        registry(run,'evaluating')
        schedule(jobs,gpus,run,frozen)
        subprocess.run([sys.executable,'-m','medworld.evaluation.dense_reference','--config',str(run/'config.json'),
                        '--out',str(run/'bicubic_reference.json')],cwd=PROJECT,env=frozen,check=True)
        subprocess.run([sys.executable,'-m','medworld.evaluation.compare_run','--run',str(run)],cwd=PROJECT,env=frozen,check=True)
        atomic(run/'pipeline_status.json',{'phase':'complete','heartbeat_unix':time.time(),'comparison':'COMPARISON.md'})
        outcome='Completed: eight-hour two-stage training and full downstream tests; native-Qwen comparison available.'
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=360)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        run.mkdir(parents=True,exist_ok=True)
        atomic(run/'pipeline_status.json',{'phase':'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
               'error':repr(error),'heartbeat_unix':time.time()})
        outcome='Interrupted; see pipeline_status.json.' if isinstance(error,KeyboardInterrupt) else 'Failed; see pipeline_status.json and original run logs.'
        raise
    finally:
        if outcome is not None:registry(run,'finished',outcome)


if __name__=='__main__':main()
