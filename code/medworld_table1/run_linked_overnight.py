"""Persistent multi-GPU Table-1 run with owned processes and morning outputs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from common import ROOT, atomic_json, read_config
from summarize_linked import render, read

GPUS = {
    0:'GPU-0d675072-429f-224a-489f-9f5dfc454eb4',
    1:'GPU-0d7eec80-ebfc-ed01-d338-5172ed91d565',
    5:'GPU-ea10d9dc-c6d4-99e8-e1e5-20e9750381d3',
    6:'GPU-119eacc5-5d83-69af-02e7-8ce6ce532b78',
    7:'GPU-a1e1bb80-f4e0-7a4b-a80d-b9be8610bf40',
}


def environment(gpu):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=GPUS[gpu] if gpu is not None else '',
        CUDA_DEVICE_ORDER='PCI_BUS_ID', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')


def free(gpu):
    output = subprocess.check_output(['nvidia-smi','--id='+GPUS[gpu],
        '--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'], text=True)
    memory, util = map(int,output.strip().split(','))
    return memory < 512 and util < 10


def stop_server(root):
    state = read(root/'qwen9b_server/server_8130.json')
    if not state:
        return
    pid = state['pid']
    stat = Path(f'/proc/{pid}/stat')
    if stat.exists() and stat.read_text().split()[21] == state['proc_start_ticks']:
        cmd = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        if 'vllm.entrypoints.cli.main' not in cmd or '8130' not in cmd:
            raise RuntimeError('Owned server command does not match; refusing to stop')
        os.killpg(pid, signal.SIGTERM)
        for _ in range(30):
            if not stat.exists():
                break
            time.sleep(1)


def qwen_worker(root, config):
    try:
        for split in ('validate','test'):
            output = root/'qwen9b'/('evaluation_'+split)
            subprocess.run([sys.executable, str(ROOT/'evaluate_qwen9b.py'), '--config', str(config),
                '--split',split,'--out',str(output)], cwd=ROOT, env=environment(None), check=True)
    finally:
        stop_server(root)
    for split in ('validate','test'):
        subprocess.run([sys.executable,str(ROOT/'score.py'),'--config',str(config),
            '--mode','qwen9b','--split',split,'--out',str(root/'qwen9b'/('evaluation_'+split)),
            '--score-only'], cwd=ROOT, env=environment(7), check=True)


def launch(a):
    os.umask(0o077)
    root = a.run.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'runner_status.json').exists():
        raise FileExistsError('Coordinator already has state; inspect before relaunching')
    cfg = read_config(a.config)
    atomic_json(root/'config.json',cfg)
    source = root/'source'
    source.mkdir(exist_ok=True)
    for path in ROOT.glob('*.py'):
        shutil.copy2(path,source/path.name)
    shutil.copytree(ROOT/'tests',source/'tests',dirs_exist_ok=True)
    atomic_json(root/'gpu_plan.json',dict(physical_gpu_uuids=GPUS,
        ours=0,direct=1,matched=5,diagnostics_and_copy=6,qwen9b=7,
        excluded=[4],untouched_running_screening=[2,3]))
    jobs, tasks, finished = {}, {}, set()
    config_args = ['--config',str(root/'config.json')]

    def start(name,gpu,command):
        directory=root/name
        directory.mkdir(parents=True,exist_ok=True)
        with (directory/'console.log').open('a') as log:
            process=subprocess.Popen([sys.executable,*command],cwd=ROOT,env=environment(gpu),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        jobs[name]=process
        tasks[name]=dict(pid=process.pid,gpu=gpu,command=command,started_unix=time.time())
        print(f'{name}: pid={process.pid}, physical_gpu={gpu}',flush=True)

    def evaluation(name,gpu,mode,checkpoint,split='test',limit=0):
        cmd=[str(ROOT/'evaluate.py'),*config_args,'--mode',mode,'--split',split,'--out',str(root/name)]
        if checkpoint:cmd+=['--checkpoint',str(checkpoint)]
        if limit:cmd+=['--limit',str(limit)]
        start(name,gpu,cmd)

    def occupied(gpu):
        return any(tasks[k]['gpu']==gpu and p.poll() is None for k,p in jobs.items())

    # Qwen server was explicitly started on the otherwise unused physical GPU 7.
    start('qwen9b_worker',None,[str(ROOT/'run_linked_overnight.py'),'--qwen-worker',
        '--run',str(root),*config_args])
    last_plot=0
    while True:
        for name,proc in jobs.items():
            code=proc.poll()
            if code is not None and name not in finished:
                tasks[name].update(exit_code=code,finished_unix=time.time())
                finished.add(name)
                print(f'{name} exited {code}',flush=True)
        features=(Path(cfg['cache'])/'features.json').exists()
        for name,gpu,mode in [('ours',0,'ours'),('direct',1,'direct')]:
            smoke=name+'_smoke'
            if features and smoke not in jobs and free(gpu):
                start(smoke,gpu,[str(ROOT/'train.py'),*config_args,'--run',str(root/smoke),
                    '--mode',mode,'--smoke-steps','2'])
            if smoke in finished and jobs[smoke].returncode==0 and name not in jobs:
                start(name,gpu,[str(ROOT/'train.py'),*config_args,'--run',str(root/name),'--mode',mode])
        if (root/'ours/checkpoint_stage1.pt').exists() and 'matched' not in jobs and free(5):
            start('matched',5,[str(ROOT/'train.py'),*config_args,'--run',str(root/'matched'),
                '--follow',str(root/'ours')])
        for name,gpu in [('ours',0),('direct',1),('matched',5)]:
            if name in finished and jobs[name].returncode==0:
                val=name+'/evaluation_validate'
                test=name+'/evaluation_test'
                if val not in jobs:
                    evaluation(val,gpu,name,root/name/'checkpoint_final.pt',split='validate')
                if val in finished and test not in jobs:
                    evaluation(test,gpu,name,root/name/'checkpoint_final.pt')
        # One sequential auxiliary lane; an earlier read-only diagnostic may
        # already occupy GPU 6, so wait until it actually becomes free.
        if features and not occupied(6) and free(6):
            auxiliary=[]
            for split in ('validate','test'):
                name='copy/evaluation_'+split
                auxiliary.append((name,lambda name=name,split=split:evaluation(name,6,'copy',None,split)))
            old=ROOT/'runs/pilot_20260908_8h/ours/checkpoint_stage1.pt'
            for name,checkpoint,limit in [
                ('diagnostics_old_stage1',old,24),
                ('diagnostics_new_stage1',root/'ours/checkpoint_stage1.pt',24),
                ('diagnostics_new_final',root/'ours/checkpoint_final.pt',24)]:
                if checkpoint.exists():
                    auxiliary.append((name,lambda name=name,checkpoint=checkpoint,limit=limit:
                        start(name,6,[str(ROOT/'diagnose_states.py'),'--checkpoint',str(checkpoint),
                            '--out',str(root/name),'--limit',str(limit)])))
            for model in ('ours','direct','matched'):
                checkpoint=root/model/'checkpoint_4h.pt'
                name=model+'/midpoint_validate'
                if checkpoint.exists():
                    auxiliary.append((name,lambda name=name,checkpoint=checkpoint,model=model:
                        evaluation(name,6,model,checkpoint,split='validate',limit=32)))
            for name,fn in auxiliary:
                if name not in jobs:
                    fn()
                    break
        failures={k:v['exit_code'] for k,v in tasks.items() if v.get('exit_code') not in (None,0)}
        essential=['ours/evaluation_test','direct/evaluation_test','matched/evaluation_test',
            'copy/evaluation_test','qwen9b_worker','diagnostics_new_final']
        done=all(k in finished for k in essential) and all(p.poll() is not None for p in jobs.values())
        # If a training/smoke job fails, surface it and finish all independent work;
        # never invent missing checkpoints or quietly omit a failed method.
        train_failed=any(k in failures for k in ('ours_smoke','direct_smoke','ours','direct','matched'))
        if train_failed and all(p.poll() is not None for p in jobs.values()):
            done=True
        atomic_json(root/'runner_status.json',dict(pid=os.getpid(),
            state=('failed' if failures else 'complete') if done else ('running_with_failures' if failures else 'running'),
            updated_unix=time.time(),features_ready=features,tasks=tasks,failures=failures,
            train_deadline=cfg['train_deadline'],requested_results_by='2026-09-10T07:30:00+08:00'))
        make_plot=time.time()-last_plot>300 or done
        render(root,plot=make_plot)
        if make_plot:last_plot=time.time()
        if done:
            return 1 if failures else 0
        time.sleep(15)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--qwen-worker',action='store_true')
    a=p.parse_args()
    if a.qwen_worker:
        qwen_worker(a.run.resolve(),Path(a.config).resolve())
    else:
        sys.exit(launch(a))
