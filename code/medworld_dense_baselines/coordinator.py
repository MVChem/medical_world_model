"""Persistent free-GPU queue. Never preempt or terminate another user's process."""
import argparse
import datetime as dt
import fcntl
import json
import signal
import subprocess
import sys
import time
from common import *

GPU_PREFERENCE=[7,2,3,6,5,1,0,4]

def gpu_inventory():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
                                 '--format=csv,noheader,nounits'],text=True,timeout=15)
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'],text=True,timeout=15)
    busy={line.split(',')[0].strip() for line in active.splitlines() if ',' in line}
    info=[]
    for line in raw.splitlines():
        i,uuid,mem,util=[v.strip() for v in line.split(',')]
        info.append(dict(index=int(i),memory_mib=int(mem),utilization=int(util),has_compute_process=uuid in busy))
    return info

def make_jobs(run):
    jobs=[]
    def add(key,script,args,ngpu,marker,deps=()):
        jobs.append(dict(id=key,script=script,args=args,gpus=ngpu,marker=str(marker),deps=list(deps),
                         status='queued',attempts=0))
    add('vjepa','vjepa.py',[],1,run/'data/vjepa_complete.json')
    order=['qwen08b','qwen4b','medgemma4b','qwen9b','qwen27b_fp8','medgemma27b']
    ngpus={'qwen27b_fp8':2,'medgemma27b':3}
    for mid in order:
        add('features_'+mid,'features.py',['--model',mid],ngpus.get(mid,1),run/mid/'features_complete.json')
    for mid in order:
        add('direction_'+mid,'direction.py',['predict','--model',mid],ngpus.get(mid,1),run/mid/'direction_metrics.json',
            ['features_'+mid])
    for mid in order:
        for task in ['segmentation','sr','grounding']:
            for variant in (['image'] if task=='grounding' else ['image','vjepa','vjepa_adapter']):
                add(f'{mid}_{task}_{variant}','train.py',['--model',mid,'--task',task,'--variant',variant,'--epochs','20'],
                    1,run/mid/f'{task}_{variant}'/'metrics.json',
                    ['features_'+mid]+(['vjepa'] if variant!='image' else []))
    return jobs

def alive(pid):
    try:
        os.kill(pid,0)
        return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except (ProcessLookupError,FileNotFoundError):return False

def run_queue(run,source):
    os.umask(0o077)
    lock=(run/'coordinator.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert (run/'data/manifest.json').exists() and (run/'direction/manifest.json').exists()
    path=run/'queue.json'
    jobs=json.loads(path.read_text()) if path.exists() else make_jobs(run)
    children={};stop=False
    def terminate(_signum,_frame):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    started=time.time();last_report=0
    while not stop:
        now=time.time()
        for j in jobs:
            if j['status']=='running':
                child=children.get(j['id'])
                rc=child.poll() if child else None
                living=rc is None and alive(j['pid'])
                if living:continue
                j['finished']=now;j['returncode']=rc
                if j.pop('preflight',False):
                    j.update(status='queued',attempts=0,retry_after=now)
                    print('preflight finished',j['id'],'cache will resume',flush=True)
                    continue
                if Path(j['marker']).exists() and rc in (0,None):j['status']='complete'
                else:
                    j['status']='failed' if j['attempts']>=3 else 'queued'
                    j['retry_after']=now+90
                    # Retry OOM with microbatch 1, preserving the effective batch and epochs.
                    log=Path(j['log']).read_text(errors='replace')[-20000:]
                    if rc==75:
                        j['status']='queued';j['attempts']-=1
                    if j['script']=='train.py' and ('OutOfMemory' in log or 'out of memory' in log) and '--microbatch' not in j['args']:
                        j['args']+=['--microbatch','1']
                print('finished',j['id'],j['status'],rc,flush=True)
            elif j['status']!='complete' and Path(j['marker']).exists():j['status']='complete'
        try:inv=gpu_inventory()
        except (subprocess.SubprocessError,OSError) as e:
            print('GPU inventory temporarily unavailable',type(e).__name__,flush=True)
            time.sleep(15);continue
        held={i for j in jobs if j['status']=='running' for i in j['assigned_gpus']}
        free=[i for i in GPU_PREFERENCE if any(r['index']==i and r['memory_mib']<512 and
              r['utilization']<5 and not r['has_compute_process'] for r in inv) and i not in held]
        complete={j['id'] for j in jobs if j['status']=='complete'}
        # Feature jobs take priority; training and direction can fill remaining cards.
        for j in jobs:
            if j['status']!='queued' or j.get('retry_after',0)>now or not set(j['deps'])<=complete:continue
            if len(free)<j['gpus']:continue
            assigned=free[:j['gpus']]
            # MedGemma 27B can fit 3x21 GiB; use four cards when idle cards are available.
            if j['gpus']==3 and len(free)>=4:assigned=free[:4]
            free=[i for i in free if i not in assigned]
            j['attempts']+=1
            logfile=run/'logs'/f'{j["id"]}_attempt{j["attempts"]}_{int(now)}.log'
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,assigned)),MEDWORLD_PROJECT=str(PROJECT),
                     MEDWORLD_MODEL_FILE=str(run/'models.json'),
                     OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
                     HF_HUB_DISABLE_PROGRESS_BARS='1',TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR='1',
                     VLLM_USE_FLASHINFER_SAMPLER='0')
            cmd=[sys.executable,'-u',str(source/'worker.py'),str(source/j['script']),'--run',str(run)]+j['args']
            with logfile.open('a') as log:
                child=subprocess.Popen(cmd,stdout=log,stderr=log,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
            children[j['id']]=child
            j.update(status='running',pid=child.pid,assigned_gpus=assigned,started=now,log=str(logfile),command=cmd)
            print('started',j['id'],assigned,'pid',child.pid,flush=True)
            atomic(path,jobs)
        counts={s:sum(j['status']==s for j in jobs) for s in ['queued','running','complete','failed']}
        atomic(path,jobs)
        atomic(run/'status.json',dict(pid=os.getpid(),started=started,updated=now,counts=counts,gpus=inv,
            delivery_target='2026-09-14T08:00:00+08:00',policy='all idle GPUs; never preempt foreign jobs',
            phase='running' if counts['queued'] or counts['running'] else 'finished',
            status='running' if counts['queued'] or counts['running'] else 'finished_with_errors' if counts['failed'] else 'finished'))
        if now-last_report>120:
            subprocess.run([sys.executable,str(source/'report.py'),'--run',str(run)],env=dict(os.environ,MEDWORLD_PROJECT=str(PROJECT)))
            last_report=now
        if not counts['queued'] and not counts['running']:break
        # Blocked dependants become explicit failures once all possible work is exhausted.
        failed={j['id'] for j in jobs if j['status']=='failed'}
        for j in jobs:
            if j['status']=='queued' and set(j['deps'])&failed:
                j.update(status='failed',reason='dependency failed: '+','.join(sorted(set(j['deps'])&failed)))
        time.sleep(15)
    atomic(path,jobs)
    subprocess.run([sys.executable,str(source/'report.py'),'--run',str(run)],env=dict(os.environ,MEDWORLD_PROJECT=str(PROJECT)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT_RUN)
    p.add_argument('--source',type=Path,default=ROOT)
    a=p.parse_args();run_queue(a.run.resolve(),a.source.resolve())
