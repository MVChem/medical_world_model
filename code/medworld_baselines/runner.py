"""Persistent, resumable sweep coordinator. Only stops servers it started/adopted."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
import signal
import subprocess
import sys
import time
import traceback
from base import *
from infer import request
from report import render

SOURCE=Path(__file__).resolve().parent

def alive(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text()
        return stat.split(') ',1)[1][0]!='Z'
    except (FileNotFoundError,ProcessLookupError):return False

def env_for(gpus):
    return dict(os.environ,MEDWORLD_PROJECT=str(PROJECT),CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),
        NCCL_P2P_DISABLE='1',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        VLLM_USE_FLASHINFER_SAMPLER='0')

def execute(run,name,cmd,gpus=(),timeout=18000):
    with (run/(name+'.log')).open('a') as log:
        p=subprocess.Popen(cmd,env=env_for(gpus),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic(run/(name+'_process.json'),dict(pid=p.pid,command=cmd,gpus=list(gpus),started=time.time()))
        try:return p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            return 124

def start_server(run,model,gpus,port):
    cmd=[sys.executable,'-m','vllm.entrypoints.cli.main','serve',model['path'],
        '--served-model-name','baseline-'+model['id'],'--host','127.0.0.1','--port',str(port),
        '--tensor-parallel-size',str(len(gpus)),'--dtype','bfloat16','--max-model-len','8192',
        '--max-num-seqs','16','--max-num-batched-tokens','4096','--gpu-memory-utilization','0.90',
        '--limit-mm-per-prompt','{"image":1,"video":0}','--enforce-eager','--disable-custom-all-reduce',
        '--no-enable-log-requests','--disable-uvicorn-access-log']
    if model['family']=='qwen':cmd+=['--reasoning-parser','qwen3']
    with (run/(model['id']+'_server.log')).open('a') as log:
        p=subprocess.Popen(cmd,env=env_for(gpus),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    meta=dict(pid=p.pid,command=cmd,gpus=gpus,started=time.time(),owned=True)
    atomic(run/(model['id']+'_server.json'),meta)
    return meta

def stop_server(meta):
    pid=meta['pid']
    if not alive(pid):return
    command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    if 'vllm.entrypoints.cli.main' not in command or os.getpgid(pid)!=pid:
        raise RuntimeError('Owned server identity changed; refusing to stop process')
    os.killpg(pid,signal.SIGTERM)
    for _ in range(30):
        if not alive(pid):break
        time.sleep(1)
    else:os.killpg(pid,signal.SIGKILL)
    time.sleep(8)

def infer_model(run,model,endpoint):
    cmd=[sys.executable,str(SOURCE/'infer.py'),'--run',str(run),'--model',model['id'],
        '--endpoint',endpoint,'--concurrency','8','--ready-timeout','1200']
    return execute(run,model['id']+'_infer',cmd,timeout=18000)

def complete(run,name):
    p=run/name/'test/status.json'
    return p.exists() and read(p).get('status')=='complete'

def model_job(run,model,gpus,port,adopt=False):
    name=model['id'];meta=None
    try:
        if complete(run,name):return
        if adopt:
            meta=read(run/(name+'_server.json'))
            worker=read(run/(name+'_worker.json'))['pid']
            while alive(worker):time.sleep(10)
            if complete(run,name):return
        if meta is None or not alive(meta['pid']):meta=start_server(run,model,gpus,port)
        for attempt in range(2):
            code=infer_model(run,model,f'http://127.0.0.1:{port}')
            if code==0:break
            if not alive(meta['pid']):meta=start_server(run,model,gpus,port)
        atomic(run/name/'lifecycle.json',dict(status='complete' if complete(run,name) else 'failed',returncode=code,updated=time.time()))
    except Exception:
        atomic(run/name/'lifecycle.json',dict(status='failed',error=traceback.format_exc(),updated=time.time()))
        print(traceback.format_exc(),flush=True)
    finally:
        if meta is not None:stop_server(meta)
        atomic(run/name/'inference_finished.json',dict(complete=complete(run,name),updated=time.time()))

def main(args):
    os.umask(0o077);run=args.run.resolve()
    lock=(run/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    inventory={m['id']:m for m in read(run/'models.json')}
    state=dict(pid=os.getpid(),started=time.time(),status='running',stage='medgemma27b')
    atomic(run/'coordinator_status.json',state)
    def refresh():
        while state['status']=='running':
            try:
                render(run)
                state['updated']=time.time();state['phase']=state['stage'];atomic(run/'coordinator_status.json',state)
            except Exception:print(traceback.format_exc(),flush=True)
            time.sleep(30)
    import threading
    threading.Thread(target=refresh,daemon=True).start()
    def q9():
        try:
            pid=read(run/'qwen9b_worker.json')['pid']
            while alive(pid):time.sleep(10)
            if not complete(run,'qwen9b'):infer_model(run,inventory['qwen9b'],inventory['qwen9b']['endpoint'])
        finally:atomic(run/'qwen9b/inference_finished.json',dict(complete=complete(run,'qwen9b'),updated=time.time()))
    def metrics_lane():
        pending=set(inventory)
        while pending:
            for name in list(pending):
                if not (run/name/'inference_finished.json').exists():continue
                if (run/name/'test/responses.jsonl').exists():
                    cmd=[sys.executable,str(SOURCE/'score.py'),'--run',str(run),'--model',name]
                    rc=execute(run,name+'_score',cmd,[7],timeout=7200)
                    atomic(run/name/'score_finished.json',dict(returncode=rc,updated=time.time()))
                pending.remove(name)
            render(run)
            if pending:time.sleep(20)
    with ThreadPoolExecutor(max_workers=4) as pool:
        q9_future=pool.submit(q9)
        model_job(run,inventory['medgemma27b'],[0,5,6,7],8140,adopt=True)
        state['stage']='remaining_models';atomic(run/'coordinator_status.json',state)
        big=pool.submit(model_job,run,inventory['qwen27b_fp8'],[0,5],8141)
        def small_lane():
            for name in ['qwen08b','qwen4b']:
                model_job(run,inventory[name],[6],8142)
            if (SOURCE/'green_eval.py').exists():
                execute(run,'green_eval',[sys.executable,str(SOURCE/'green_eval.py'),'--run',str(run),'--gpu','6'],[6],timeout=18000)
        small=pool.submit(small_lane)
        model_job(run,inventory['medgemma4b'],[7],8143)
        metrics_lane()
        for future in [q9_future,big,small]:future.result()
    state.update(status='finished',phase='finished',finished=time.time());atomic(run/'coordinator_status.json',state);render(run)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args())
