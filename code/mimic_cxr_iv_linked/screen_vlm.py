"""Resumable localhost-only paired-image/report screening with SQLite commits."""
import argparse
import asyncio
from collections import Counter
import copy
from datetime import datetime,timezone
import hashlib
import fcntl
import json
import os
from pathlib import Path
import random
import sqlite3
import time
from urllib.parse import urlparse
import httpx
from common import read_jsonl,dump_json,file_sha256
from vlm_protocol import VERSION,SYSTEM,SCHEMA,request,parse_response,review_flags,protocol_hash


def now(): return datetime.now(timezone.utc).isoformat()


async def run(a):
    os.umask(0o077)
    a.out.mkdir(parents=True,exist_ok=True)
    # Exactly one coordinator may write this resume database at a time.
    job_lock=(a.out/'writer.lock').open('a')
    fcntl.flock(job_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for url in a.endpoints:
        if urlparse(url).hostname not in ('127.0.0.1','localhost','::1'):
            raise ValueError('Restricted research inputs must use a localhost endpoint.')
    manifest=list(read_jsonl(a.manifest))
    if a.scope=='strict': manifest=[r for r in manifest if r['original_strict_subset']]
    if a.train_only: manifest=[r for r in manifest if r['split']=='train']
    # Random pair order prevents the live prefix being merely the first few patients.
    manifest.sort(key=lambda r:hashlib.sha256(('20260909:'+r['pair_id']).encode()).hexdigest())
    if a.limit: manifest=manifest[:a.limit]
    assert len({r['pair_id'] for r in manifest})==len(manifest)
    config={'version':VERSION,'protocol_sha256':protocol_hash(),'manifest_sha256':file_sha256(a.manifest),
        'selected_pair_ids_sha256':hashlib.sha256('\n'.join(r['pair_id'] for r in manifest).encode()).hexdigest(),
        'scope':a.scope,'limit':a.limit,'train_only':a.train_only,'model':a.model,
        'model_snapshot':a.model_snapshot,'temperature':0,'seed':20260909,'max_tokens':512,
        'incomplete_output_retry_cap':2048,
        'incomplete_output_retry_evidence_max_chars':240,
        'thinking':False,'image_count_per_request':2,'image_max_pixels':1048576,
        'image_min_pixels':262144,'no_report_truncation':True,
        'request_implementation_sha256':file_sha256(Path(__file__).with_name('vlm_protocol.py')),
        'interpretation':'Retrospective model-generated weak labels, not adjudicated truth; both images and both reports are supplied. Image assessment is report-conditioned. These labels must not be passed as forecast inputs.'}
    config_path=a.out/'config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:
        previous=json.loads(config_path.read_text())
        if previous == {k:v for k,v in config.items() if k not in ('incomplete_output_retry_cap','incomplete_output_retry_evidence_max_chars')}:
            # Explicit, narrow migration: retain all successful base-512 outputs;
            # increase budget only after an incomplete, previously uncounted call.
            dump_json(a.out/'config_before_length_retry.json',previous)
        else:
            raise ValueError('Different protocol, model or input selection; use a new output directory.')
    dump_json(config_path,config)
    dump_json(a.out/'protocol.json',{'system':SYSTEM,'schema':SCHEMA})
    db=sqlite3.connect(a.out/'results.sqlite')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS results(pair_id TEXT PRIMARY KEY, subject_id TEXT, split TEXT, label TEXT, flags TEXT, raw_response TEXT, usage TEXT, endpoint TEXT, latency REAL, completed_at TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY, pair_id TEXT, endpoint TEXT, error TEXT, at TEXT)')
    if 'generation_max_tokens' not in {r[1] for r in db.execute('PRAGMA table_info(results)')}:
        db.execute('ALTER TABLE results ADD COLUMN generation_max_tokens INTEGER DEFAULT 512')
    columns={r[1] for r in db.execute('PRAGMA table_info(attempts)')}
    if 'raw_response' not in columns: db.execute('ALTER TABLE attempts ADD COLUMN raw_response TEXT')
    if 'generation_max_tokens' not in columns: db.execute('ALTER TABLE attempts ADD COLUMN generation_max_tokens INTEGER')
    db.commit()
    done={r[0] for r in db.execute('SELECT pair_id FROM results')}
    expected={r['pair_id'] for r in manifest}
    if not done<=expected: raise ValueError('Database contains foreign pair IDs')
    q=asyncio.Queue()
    for r in manifest:
        if r['pair_id'] not in done: q.put_nowait(r)
    started=time.monotonic()
    initial=len(done)
    completed=initial
    failed=0
    counts=Counter(json.loads(r[0])['primary_class'] for r in db.execute('SELECT label FROM results'))
    outstanding_errors=[]
    finished=False

    def status():
        elapsed=time.monotonic()-started
        speed=(completed-initial)/elapsed if elapsed else 0
        s={'updated_at':now(),'total':len(manifest),'completed':completed,'remaining':len(manifest)-completed,
            'unresolved_failures_this_launch':failed,'class_counts_completed_only':dict(counts),
            'elapsed_seconds_this_launch':elapsed,'pairs_per_second_this_launch':speed,
            'estimated_remaining_hours':(len(manifest)-completed)/speed/3600 if speed else None,
            'status':'complete' if completed==len(manifest) else 'finished_with_errors' if finished else 'running',
            'process_id':os.getpid(),'denominator_note':'Partial counts are not final whole-cohort statistics.'}
        dump_json(a.out/'progress.json',s)
        return s

    async def monitor():
        while not finished:
            s=status()
            print(json.dumps(s),flush=True)
            await asyncio.sleep(30)

    async with httpx.AsyncClient(timeout=httpx.Timeout(900,connect=10),limits=httpx.Limits(max_connections=a.concurrency*len(a.endpoints)+8),trust_env=False) as client:
        for endpoint in a.endpoints:
            health=await client.get(endpoint+'/health')
            health.raise_for_status()
            models=await client.get(endpoint+'/v1/models')
            models.raise_for_status()
            if a.model not in {r['id'] for r in models.json()['data']}:
                raise ValueError('Served model mismatch')

        async def worker(endpoint):
            nonlocal completed,failed
            while True:
                try: row=q.get_nowait()
                except asyncio.QueueEmpty: return
                try:
                    request_body=copy.deepcopy(request(row,a.model))
                    bounded_retry=False
                    for attempt in range(a.retries):
                        t=time.monotonic()
                        body=None
                        try:
                            response=await client.post(endpoint+'/v1/chat/completions',json=request_body)
                            response.raise_for_status()
                            body=response.json()
                            label,raw=parse_response(body)
                            flags=review_flags(label)
                            if bounded_retry: flags=sorted(set(flags)|{'bounded_evidence_retry'})
                            db.execute('INSERT INTO results VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                                (row['pair_id'],row['subject_id'],row['split'],json.dumps(label),json.dumps(flags),
                                 json.dumps(body),json.dumps(body.get('usage',{})),endpoint,time.monotonic()-t,now(),request_body['max_tokens']))
                            db.commit()
                            completed+=1
                            counts[label['primary_class']]+=1
                            break
                        except Exception as e:
                            error=f'{type(e).__name__}: {e}'
                            if isinstance(e,httpx.HTTPStatusError): error+=' '+e.response.text[:2000]
                            db.execute('INSERT INTO attempts(pair_id,endpoint,error,at,raw_response,generation_max_tokens) VALUES(?,?,?,?,?,?)',
                                (row['pair_id'],endpoint,error,now(),json.dumps(body) if body else None,request_body['max_tokens']))
                            db.commit()
                            if body and body.get('choices',[{}])[0].get('finish_reason')=='length':
                                remaining_context=8192-body.get('usage',{}).get('prompt_tokens',6144)
                                request_body['max_tokens']=max(request_body['max_tokens'],min(2048,remaining_context,2*request_body['max_tokens']))
                                for field in ('image_evidence','report_evidence'):
                                    request_body['response_format']['json_schema']['schema']['properties'][field].update(minLength=1,maxLength=240)
                                bounded_retry=True
                            if attempt+1==a.retries:
                                failed+=1
                                outstanding_errors.append({'pair_id':row['pair_id'],'error':error})
                            else: await asyncio.sleep(min(2**attempt,15))
                finally: q.task_done()
                if failed>=50 and completed==initial:
                    raise RuntimeError('No successful requests after 50 failures; refusing a failed full sweep.')

        mon=asyncio.create_task(monitor())
        try:
            await asyncio.gather(*(worker(e) for e in a.endpoints for _ in range(a.concurrency)))
        finally:
            finished=True
            mon.cancel()
            status()
            db.close()
            dump_json(a.out/'unresolved_errors.json',outstanding_errors)
    if failed: raise RuntimeError(f'{failed} pairs unresolved; rerun to resume only missing pairs.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--endpoints',nargs='+',default=['http://127.0.0.1:8120'])
    p.add_argument('--model',default='mimic-qwen35-9b')
    p.add_argument('--model-snapshot',default='Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a')
    p.add_argument('--scope',choices=['strict','expanded'],default='expanded')
    p.add_argument('--concurrency',type=int,default=32)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--train-only',action='store_true')
    p.add_argument('--retries',type=int,default=3)
    asyncio.run(run(p.parse_args()))
