"""Matched held-out VLM evaluation with the established prompts and scorers."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

def atomic(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def prepare(args):
    project=Path(os.environ['MEDWORLD_PROJECT'])
    old=project/'code/medworld_baselines/runs/raw_models_20260911'
    args.run.mkdir(parents=True,exist_ok=True)
    for name in ['protocol.json','vocabulary.json']:
        dst=args.run/name
        if dst.exists() and dst.read_bytes()!=(old/name).read_bytes():
            raise ValueError('Frozen original protocol differs')
        shutil.copy2(old/name,dst)
    shutil.copytree(old/'cohort',args.run/'cohort',dirs_exist_ok=True)
    # Original reference cache is valid because inputs and protocol are byte-identical.
    if (old/'reference_labels').exists():
        shutil.copytree(old/'reference_labels',args.run/'reference_labels',dirs_exist_ok=True)
    source=args.run/'source'
    source.mkdir(exist_ok=True)
    baseline_source=Path(__file__).resolve().parent.parent/'medworld_baselines'
    for p in baseline_source.glob('*.py'):
        shutil.copy2(p,source/p.name)
    inventory=[]
    for name in args.models.split(','):
        asset=json.loads((args.assets/name/'download_status.json').read_text())
        if asset['status']!='complete':raise ValueError(f'{name} assets not complete')
        inventory.append(dict(id=name,label=asset['repo'],family='native',path=asset['path'],tp=1,
                              config_sha256=asset['config_sha256'],revision=asset['revision']))
    atomic(args.run/'models.json',inventory)
    atomic(args.run/'provenance.json',dict(status='complete',source_run=str(old),
        protocol_sha256=hashlib.sha256((args.run/'protocol.json').read_bytes()).hexdigest(),
        training='none; the paper labels these rows ZS',created=time.time(),models=inventory,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')}))

def infer(args):
    model=next(m for m in json.loads((args.run/'models.json').read_text()) if m['id']==args.model)
    source=args.run/'source'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    cmd=[sys.executable,'-m','vllm.entrypoints.cli.main','serve',model['path'],
         '--served-model-name',args.model,'--host','127.0.0.1','--port',str(port),
         '--dtype','bfloat16','--max-model-len','8192','--max-num-seqs','8',
         '--max-num-batched-tokens','4096','--gpu-memory-utilization','0.90',
         '--limit-mm-per-prompt','{"image":1,"video":0}', '--enforce-eager',
         '--logprobs-mode','raw_logprobs','--no-enable-log-requests','--disable-uvicorn-access-log']
    env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'VLLM_USE_FLASHINFER_SAMPLER':'0','OMP_NUM_THREADS':'4'}
    out=args.run/args.model;out.mkdir(parents=True,exist_ok=True)
    started=time.time()
    server=None
    def stop(signum, frame):raise KeyboardInterrupt('Worker stopping')
    signal.signal(signal.SIGTERM,stop)
    try:
        with (out/'server.log').open('a') as log:
            server=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        atomic(out/'server_launch.json',dict(pid=server.pid,argv=cmd,started=started,
                                           cuda_visible_devices=env.get('CUDA_VISIBLE_DEVICES')))
        sys.path.insert(0,str(source))
        from infer import request
        deadline=time.monotonic()+1200
        while True:
            if server.poll() is not None:raise RuntimeError('vLLM server exited; see server.log')
            try:
                request(f'http://127.0.0.1:{port}',route='/v1/models',timeout=2);break
            except Exception:
                if time.monotonic()>deadline:raise TimeoutError('vLLM start timeout')
                time.sleep(2)
        argv=[sys.executable,str(source/'infer.py'),'--run',str(args.run),'--model',args.model,
              '--endpoint',f'http://127.0.0.1:{port}','--concurrency','8']
        if args.limit:argv+=['--limit',str(args.limit),'--tasks','table1_report,table1_prob,table2_report,table2_prob']
        subprocess.run(argv,env=env,check=True)
        atomic(out/'inference_finished.json',dict(complete=True,status='complete',updated=time.time(),
               seconds=time.time()-started,smoke=bool(args.limit)))
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:server.wait(timeout=40)
            except subprocess.TimeoutExpired:server.kill();server.wait(timeout=15)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','infer'])
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--assets',type=Path)
    p.add_argument('--models',default='qwen25vl7b,llava16mistral7b')
    p.add_argument('--model')
    p.add_argument('--limit',type=int,default=0)
    args=p.parse_args();args.run=args.run.resolve()
    prepare(args) if args.mode=='prepare' else infer(args)

if __name__=='__main__':main()
