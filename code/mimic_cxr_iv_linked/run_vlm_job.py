"""Durable full-sweep coordinator: resume, refresh statistics, verify completion."""
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
import urllib.request
from common import dump_json,file_sha256


def run(a):
    os.umask(0o077)
    root=Path(__file__).parent.resolve()
    a.out.mkdir(parents=True,exist_ok=True)
    lock=(a.out/'coordinator.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dump_json(a.out/'job.json',dict(pid=os.getpid(),args=vars(a)|{'out':str(a.out),'manifest':str(a.manifest),'server_state':str(a.server_state)},started_at=time.strftime('%Y-%m-%d %H:%M:%S %z')))
    snapshot=a.out/'source'
    snapshot.mkdir(exist_ok=True)
    hashes={}
    for name in ['common.py','screen_vlm.py','vlm_protocol.py','vlm_triage.py','summarize_vlm.py','prepare_vlm.py','serve_vlm.py','run_vlm_job.py','vlm_runtime/sitecustomize.py']:
        source=root/name
        target=snapshot/name
        target.parent.mkdir(exist_ok=True,parents=True)
        if target.exists() and file_sha256(source)!=file_sha256(target):
            raise ValueError('Source snapshot changed; use a new job folder.')
        shutil.copy2(source,target)
        hashes[name]=file_sha256(source)
    dump_json(a.out/'source_hashes.json',hashes)
    ready=set()
    for _ in range(120):
        for url in a.endpoints:
            if url in ready: continue
            try:
                with urllib.request.urlopen(url+'/health',timeout=2) as r:
                    if r.status==200: ready.add(url)
            except Exception: pass
        if len(ready)==len(a.endpoints): break
        print(f'Servers ready: {len(ready)}/{len(a.endpoints)}',flush=True)
        time.sleep(10)
    else: raise RuntimeError('Requested model servers did not become healthy in 20 minutes.')
    args=[sys.executable,str(root/'screen_vlm.py'),'--manifest',str(a.manifest),'--out',str(a.out),
        '--scope',a.scope,'--concurrency',str(a.concurrency),'--endpoints',*a.endpoints]
    summary=[sys.executable,str(root/'summarize_vlm.py'),'--manifest',str(a.manifest),'--results',str(a.out),'--out',str(a.out/'summary')]
    completed=False
    for attempt in range(4):
        with (a.out/'screen.log').open('a') as log:
            child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
        dump_json(a.out/'client_process.json',{'pid':child.pid,'attempt':attempt+1,'args':args})
        updated=0
        while child.poll() is None:
            if time.monotonic()-updated>300 and (a.out/'progress.json').exists():
                with (a.out/'summary.log').open('a') as log:
                    subprocess.run(summary,stdout=log,stderr=subprocess.STDOUT,check=True)
                updated=time.monotonic()
            time.sleep(10)
        with (a.out/'summary.log').open('a') as log:
            subprocess.run(summary,stdout=log,stderr=subprocess.STDOUT,check=True)
        stats=json.loads((a.out/'summary/statistics.json').read_text())
        if stats['complete']:
            dump_json(a.out/'completion.json',dict(complete=True,pairs=stats['completed_pairs'],
                expected_pairs=stats['expected_pairs'],statistics_sha256=file_sha256(a.out/'summary/statistics.json'),
                labels_sha256=file_sha256(a.out/'summary/pair_labels.jsonl'),completed_at=time.strftime('%Y-%m-%d %H:%M:%S %z')))
            completed=True
            print(f'ALL {stats["completed_pairs"]:,} PAIRS COMPLETE',flush=True)
            break
        print(f'Client exit {child.returncode}; retrying only missing pairs after 30 seconds.',flush=True)
        time.sleep(30)
    if a.stop_owned_servers and completed:
        for url in a.endpoints:
            port=__import__('urllib.parse',fromlist=['urlparse']).urlparse(url).port
            record=a.server_state/f'server_{port}.json'
            if not record.exists(): continue
            r=json.loads(record.read_text())
            proc=Path(f'/proc/{r["pid"]}')
            if not proc.exists(): continue
            ticks=(proc/'stat').read_text().split()[21]
            cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            if (ticks==r.get('proc_start_ticks') and 'vllm.entrypoints.cli.main' in cmd
                and '--served-model-name mimic-qwen35-9b' in cmd and f'--port {port}' in cmd):
                os.killpg(r['pid'],signal.SIGTERM)
    if not completed: raise RuntimeError('Full sweep still has unresolved pairs; see attempts table and unresolved_errors.json.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--endpoints',nargs='+',required=True)
    p.add_argument('--scope',choices=['strict','expanded'],default='expanded')
    p.add_argument('--concurrency',type=int,default=16)
    p.add_argument('--server-state',type=Path,required=True)
    p.add_argument('--stop-owned-servers',action='store_true')
    run(p.parse_args())
