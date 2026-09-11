"""Resumable native VLM inference. This module never reads reference artifacts."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import io
import math
import os
import time
import urllib.error
import urllib.request
from PIL import Image, ImageOps
from base import *

HORIZONS = ['6 to 24 hours', '1 to 3 days', '3 to 7 days', '7 to 30 days']

def request(endpoint, payload=None, route='/v1/chat/completions', timeout=300):
    req=urllib.request.Request(endpoint+route, data=json.dumps(payload).encode() if payload is not None else None,
                              headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=timeout) as f:
            return json.load(f)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'HTTP {e.code}: {e.read().decode()[:1200]}') from e

@lru_cache(maxsize=1024)
def image_url(path):
    with Image.open(path) as im:
        im=ImageOps.pad(im.convert('RGB'),(512,512),method=Image.Resampling.BICUBIC,color=(0,0,0))
        out=io.BytesIO();im.save(out,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(out.getvalue()).decode()

def future_context(row):
    return ('Current radiograph report:\n'+row['report']+'\n\n'+row['ehr']+
        '\n\nRequested follow-up horizon: '+HORIZONS[row['horizon']]+'. '
        'Only the supplied current study is the comparison baseline. No future study has been observed.\n')

def prompt_for(task, row, finding=None, vocabulary=None):
    if task=='table1_report':
        return future_context(row)+'Predict the follow-up chest radiograph report. Write only FINDINGS and IMPRESSION, without reasoning or introductory text.'
    if task=='table1_prob':
        return future_context(row)+f'Will {finding.lower()} be present on the follow-up chest radiograph at the requested horizon? Answer exactly Yes or No.'
    if task=='table2_report':
        return 'Describe this chest radiograph. Write only FINDINGS and IMPRESSION, without reasoning or introductory text.'
    if task=='table2_prob':
        return f'Is {finding.lower()} present on this chest radiograph? Answer exactly Yes or No.'
    if task=='qa':
        return (row['question']+'\nReturn only a JSON array of standardized names chosen from this fixed ontology: '+
                json.dumps(vocabulary)+'. Use [] if none are present. Do not include explanations.')
    raise ValueError(task)

def probability_from_logprobs(yes, no):
    if not all(math.isfinite(x) and x<=1e-5 for x in [yes,no]):
        raise ValueError('Invalid raw candidate log probabilities')
    delta=max(-700.,min(700.,no-yes))
    return 1./(1.+math.exp(delta))

def main(args):
    os.umask(0o077)
    run=args.run.resolve()
    model=next(m for m in read(run/'models.json') if m['id']==args.model)
    endpoint=args.endpoint or model.get('endpoint')
    deadline=time.monotonic()+args.ready_timeout
    while True:
        try:
            server=request(endpoint,route='/v1/models',timeout=5)['data'][0]
            if Path(server['root']).resolve()!=Path(model['path']).resolve():
                raise ValueError('Endpoint checkpoint differs from requested model')
            break
        except Exception:
            if time.monotonic()>deadline:raise
            time.sleep(5)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model['path'],local_files_only=True)
    candidate_ids={s:tokenizer.encode(s,add_special_tokens=False) for s in ['Yes','No']}
    if any(len(ids)!=1 for ids in candidate_ids.values()):
        raise ValueError('Protocol requires complete single-token Yes and No candidates')
    candidate_ids={s:ids[0] for s,ids in candidate_ids.items()}
    out=run/model['id']/args.split;out.mkdir(parents=True,exist_ok=True)
    plan=dict(protocol_sha256=digest(run/'protocol.json'),model_config=model['config_sha256'],
        checkpoint=model['path'],candidates=candidate_ids,split=args.split,
        prompt_source_sha256=digest(__file__),max_report_tokens=384,max_qa_tokens=192)
    if (out/'plan.json').exists() and read(out/'plan.json')!=plan:
        raise ValueError('Resume protocol/model/prompts changed')
    atomic(out/'plan.json',plan)
    vocabulary=read(run/'vocabulary.json')
    inputs1=rows(run/'cohort'/f'table1_inputs_{args.split}.jsonl')[:args.limit or None]
    inputs2=rows(run/'cohort'/f'table2_inputs_{args.split}.jsonl')[:args.limit or None]
    inputsq=rows(run/'cohort'/f'qa_inputs_{args.split}.jsonl')[:args.limit or None]
    groups={
        'table1_report':[(r,None) for r in inputs1],
        'table1_prob':[(r,f) for r in inputs1 for f in FUTURE_FINDINGS],
        'table2_report':[(r,None) for r in inputs2 if r['report_generation']],
        'table2_prob':[(r,f) for r in inputs2 if r['classification'] for f in CURRENT_FINDINGS],
        'qa':[(r,None) for r in inputsq],
    }
    chosen=list(groups) if args.tasks=='all' else args.tasks.split(',')
    done={}
    journal=out/'responses.jsonl'
    if journal.exists():
        # Ignore only an interrupted final write, preserving every committed line.
        raw=journal.read_bytes(); lines=raw.splitlines(keepends=True)
        if lines and not lines[-1].endswith(b'\n'):
            journal.write_bytes(b''.join(lines[:-1]))
        for r in rows(journal):
            if r.get('ok'):done[r['key']]=r
    def key(task,row,finding):
        return task+'|'+row['id']+'|'+(finding or '')
    def payload(row,text,tokens):
        d=dict(model=server['id'],messages=[dict(role='user',content=[
            dict(type='image_url',image_url=dict(url=image_url(row['image']))),dict(type='text',text=text)])],
            temperature=0,seed=20260911,max_tokens=tokens)
        if model['family']=='qwen':d['chat_template_kwargs']={'enable_thinking':False}
        return d
    def predict(task,row,finding):
        started=time.monotonic()
        result=dict(key=key(task,row,finding),task=task,id=row['id'],finding=finding,ok=False)
        text=prompt_for(task,row,finding,vocabulary)
        for attempt in range(3):
            try:
                d=payload(row,text,1 if task.endswith('_prob') else (192 if task=='qa' else 384))
                if task.endswith('_prob'):
                    d.update(allowed_token_ids=list(candidate_ids.values()),logprobs=True,top_logprobs=20)
                    response=request(endpoint,d)
                    item=response['choices'][0]['logprobs']['content'][0]
                    lp={x['token']:x['logprob'] for x in item['top_logprobs']}
                    lp[item['token']]=item['logprob']
                    requests=1
                    for word,token_id in candidate_ids.items():
                        if word not in lp:
                            d['allowed_token_ids']=[token_id]
                            extra=request(endpoint,d)['choices'][0]['logprobs']['content'][0]
                            if extra['token']!=word:raise ValueError('Forced candidate token mismatch')
                            lp[word]=extra['logprob'];requests+=1
                    # All endpoints use vLLM raw_logprobs, verified by forced-token smoke checks.
                    result.update(probability=probability_from_logprobs(lp['Yes'],lp['No']),
                        candidate_logprobs={w:lp[w] for w in candidate_ids},requests=requests,usage=response['usage'])
                else:
                    response=request(endpoint,d);choice=response['choices'][0]
                    result.update(text=choice['message'].get('content') or '',finish_reason=choice['finish_reason'],usage=response['usage'])
                result.update(ok=True,attempts=attempt+1,seconds=time.monotonic()-started)
                return result
            except Exception as e:
                result['error']=f'{type(e).__name__}: {e}'
                if attempt<2:time.sleep(2*(attempt+1))
        result['seconds']=time.monotonic()-started
        return result
    state=dict(model=args.model,split=args.split,started=time.time(),pid=os.getpid(),endpoint=endpoint,tasks={},status='running')
    def flush():
        state['updated']=time.time();atomic(out/'status.json',state)
    with journal.open('a',buffering=1) as log:
        for task in chosen:
            pool_rows=groups[task]
            pending=[(r,f) for r,f in pool_rows if key(task,r,f) not in done]
            state['current_task']=task
            state['tasks'][task]=dict(expected=len(pool_rows),completed=len(pool_rows)-len(pending),errors=0)
            flush()
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures=[pool.submit(predict,task,r,f) for r,f in pending]
                for future in as_completed(futures):
                    r=future.result();log.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
                    if r['ok']:
                        done[r['key']]=r;state['tasks'][task]['completed']+=1
                    else:state['tasks'][task]['errors']+=1
                    if (state['tasks'][task]['completed']+state['tasks'][task]['errors'])%20==0:
                        flush();print(args.model,args.split,task,state['tasks'][task],flush=True)
            flush()
    state['status']='complete' if all(v['completed']==v['expected'] for v in state['tasks'].values()) else 'partial_errors'
    state['finished']=time.time();flush()
    print(json.dumps(state),flush=True)
    if state['status']!='complete':raise SystemExit(2)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--model',required=True)
    p.add_argument('--endpoint');p.add_argument('--split',choices=['test','validate'],default='test')
    p.add_argument('--tasks',default='all');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--concurrency',type=int,default=8);p.add_argument('--ready-timeout',type=int,default=900)
    main(p.parse_args())
