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


def registry(run, state, outcome=None, summary="Classification, segmentation and VQA training/evaluation."):
    root=PROJECT/'experiments';root.mkdir(exist_ok=True)
    with (root/'.registry.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=root/'registry.json'
        entries=json.loads(path.read_text()) if path.exists() else []
        identity=(run.parent.name + "_" + run.name) if run.name in ("baseline", "slots") else run.name;relative=str(run.relative_to(PROJECT))
        entries=[e for e in entries if e['id']!=identity]
        if outcome is None:
            entries.append({'id':identity,'summary':summary,'status':state,
                            'observed_at':datetime.now().astimezone().isoformat(),'run':relative,'live_status':relative+'/pipeline_status.json'})
        else:
            history=root/'README.md'
            row=f"| {datetime.now().astimezone().date()} | `{identity}` | {outcome} | [Run](../{relative}/) |\n"
            lines = [line for line in history.read_text().splitlines() if f'| `{identity}` |' not in line]
            history.write_text('\n'.join(lines).rstrip()+'\n'+row)
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


def evaluation_jobs(run, cfg=None):
    cfg = cfg or load_config(run / 'config.json' if (run / 'config.json').exists() else None)
    testing = cfg['testing']
    if not testing['enabled']:
        return []
    pairs = [(task, 'test') for task in testing['tasks']]
    if 'segmentation' in testing['tasks'] and testing['human_segmentation']:
        pairs.append(('segmentation', 'human_test'))
    jobs = []
    for task, split in pairs:
        name = task if split == 'test' else 'segmentation_human'
        jobs.append({'id': name, 'module': 'medworld.evaluation.evaluate', 'log': f'evaluation/{name}.log',
                     'args': ['--checkpoint', str(run / 'final.pt'), '--out', str(run / 'evaluation' / name),
                              '--task', task, '--split', split]})
    return jobs


def main():
    p = argparse.ArgumentParser(description='Train and test matched no-slots and eight-slot models.')
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--gpus', default='1,2,3,6')
    a = p.parse_args()
    cfg = load_config(a.config)
    if cfg['total_hours']:
        raise ValueError('Paired experiments require total_hours=0 and equal optimizer-update budgets')
    run = Path(a.out).resolve()
    if run.exists() and any(run.iterdir()):
        raise ValueError('Choose a new experiment directory')
    run.mkdir(parents=True, exist_ok=True)
    registry(run, 'training', summary='Three-task matched no-slots/slots training and test comparison.')
    child = None
    outcome = 'Failed; see pipeline_status.json and original run logs.'
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    env = dict(os.environ, PYTHONPATH=str(PROJECT / 'code'), MEDWORLD_PROJECT_ROOT=str(PROJECT))
    try:
        for name, use_slots in [('baseline', False), ('slots', True)]:
            config = run / f'{name}_config.json'
            atomic(config, {**cfg, 'slot_conditioning': use_slots})
            atomic(run / 'pipeline_status.json', {'phase': name, 'heartbeat_unix': time.time()})
            child = subprocess.Popen([sys.executable, '-m', 'medworld.launch_distributed', '--config', str(config),
                                      '--out', str(run / name), '--gpus', a.gpus], cwd=PROJECT, env=env)
            if child.wait():
                raise RuntimeError(f'{name} training/evaluation failed')
        if cfg['testing']['enabled']:
            from .evaluation.compare_run import compare
            compare(run / 'baseline', run / 'slots', run)
        atomic(run / 'pipeline_status.json', {'phase': 'complete', 'comparison': 'COMPARISON.md' if cfg['testing']['enabled'] else None, 'testing_enabled': cfg['testing']['enabled'], 'heartbeat_unix': time.time()})
        outcome = 'Completed: matched baseline/slots training; configured tests and comparison complete.' if cfg['testing']['enabled'] else 'Completed: matched baseline/slots training; testing disabled in config.'
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=360)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        atomic(run / 'pipeline_status.json', {'phase': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
               'error': repr(error), 'heartbeat_unix': time.time()})
        raise
    finally:
        registry(run, 'finished', outcome)


if __name__ == '__main__':
    main()
