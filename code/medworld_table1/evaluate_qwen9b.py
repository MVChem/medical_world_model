"""Local Qwen3.5-9B zero-shot forecasting with source-only inputs."""
import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from PIL import Image, ImageOps
from common import ROOT, atomic_json, digest, load_rows, read_config, write_rows


def request_json(url, data=None, timeout=300):
    request = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def run(a):
    from transformers import AutoTokenizer
    cfg = read_config(a.config)
    tokenizer = AutoTokenizer.from_pretrained(cfg['qwen'], local_files_only=True)
    observations = {r['id']:r for r in load_rows(Path(cfg['cache'])/'observations.jsonl')}
    rows = load_rows(Path(cfg['cache'])/(a.split+'.jsonl'))[:a.limit or None]
    a.out.mkdir(parents=True, exist_ok=True)
    plan = dict(config_sha256=digest(a.config), observations_sha256=digest(Path(cfg['cache'])/'observations.jsonl'),
        split=a.split, ids=[r['id'] for r in rows], endpoint=a.endpoint)
    plan_path = a.out/'generation_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Generation resume configuration or input differs')
    atomic_json(plan_path,plan)
    for attempt in range(120):
        try:
            models = request_json(a.endpoint+'/v1/models', timeout=5)
            break
        except Exception:
            time.sleep(5)
    else:
        raise RuntimeError('Dedicated local Qwen server did not become ready')
    model_name = models['data'][0]['id']
    def predict(row):
        obs = observations[row['source']]
        report = tokenizer.decode(tokenizer.encode(obs['report'], add_special_tokens=False)[:cfg['report_tokens']], skip_special_tokens=True)
        evidence = 'Current radiograph report:\n'+report+'\n\n'+obs['ehr_text']
        with Image.open(obs['image']) as image:
            image = ImageOps.pad(image.convert('RGB'), (512,512), method=Image.Resampling.BICUBIC)
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
        url = 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()
        horizon = ['6 to 24 hours','1 to 3 days','3 to 7 days','7 to 30 days'][row['horizon']]
        payload = dict(model=model_name, messages=[dict(role='user', content=[
            dict(type='image_url', image_url=dict(url=url)),
            dict(type='text', text=f'Current chest radiograph and available current evidence:\n{evidence}\nPredict the follow-up chest radiograph report at {horizon}. Write FINDINGS and IMPRESSION.')])],
            temperature=0, seed=cfg['seed'], max_tokens=cfg['generation_tokens'],
            chat_template_kwargs={'enable_thinking':False})
        for attempt in range(3):
            try:
                response = request_json(a.endpoint+'/v1/chat/completions', payload)
                text = response['choices'][0]['message']['content']
                if not text or not text.strip():
                    raise ValueError('Empty forecast response')
                return dict(id=row['id'], report=text, scores=None, response=response,
                    source_id=row['source'], source_inputs_only=True)
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(3)
    completed = {}
    if (a.out/'predictions.partial.jsonl').exists():
        completed = {r['id']:r for r in load_rows(a.out/'predictions.partial.jsonl')}
        if not set(completed).issubset({r['id'] for r in rows}):
            raise ValueError('Resume IDs differ')
    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        futures = {pool.submit(predict, row):row for row in rows if row['id'] not in completed}
        for future in as_completed(futures):
            row = future.result()
            completed[row['id']] = row
            write_rows(a.out/'predictions.partial.jsonl', [completed[r['id']] for r in rows if r['id'] in completed])
            if len(completed)%10 == 0:
                print(f'Qwen9B generated {len(completed)}/{len(rows)}', flush=True)
    predictions = [completed[r['id']] for r in rows]
    write_rows(a.out/'predictions.jsonl', predictions)
    counts = Counter(r['report'] for r in predictions)
    atomic_json(a.out/'diversity.json', dict(n=len(rows), unique_reports=len(counts),
        most_common_count=max(counts.values()), top5_fraction=sum(v for _,v in counts.most_common(5))/len(rows), empty_reports=0))
    atomic_json(a.out/'generation.json', dict(mode='qwen9b', split=a.split, count=len(rows),
        training='Zero-shot local pretrained Qwen3.5-9B; no training on this cohort',
        source_only=True, model=model_name, endpoint=a.endpoint, image_preprocessing='512px aspect-preserving black pad',
        config_sha256=digest(a.config), generation_tokens=cfg['generation_tokens'],
        length_limited=sum(r['response']['choices'][0]['finish_reason']=='length' for r in predictions),
        finding_scores='Unavailable; no trained continuous disease head'))


if __name__ == '__main__':
    os.umask(0o077)
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--endpoint', default='http://127.0.0.1:8130')
    p.add_argument('--split', choices=['validate','test'], default='test')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--concurrency', type=int, default=8)
    run(p.parse_args())
