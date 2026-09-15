"""Download public table baselines at immutable revisions, with resumable status."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20),b''):h.update(block)
    return h.hexdigest()

def ranged_blob(repo,revision,filename,path,size,expected):
    """Small resumable HTTP ranges avoid multi-GB connections being dropped."""
    import requests
    if path.exists() and path.stat().st_size==size and digest(path)==expected:return
    parts=path.with_name(path.name+'.parts');parts.mkdir(exist_ok=True)
    chunk=4<<20
    def part(index):
        begin=index*chunk;end=min(size,begin+chunk)-1
        target=parts/f'{index:06d}'
        if target.exists() and target.stat().st_size==end-begin+1:return
        for attempt in range(16):
            try:
                url=f'https://huggingface.co/{repo}/resolve/{revision}/{filename}?download=true&part={index}&attempt={attempt}'
                response=requests.get(url,headers={'Range':f'bytes={begin}-{end}'},timeout=45)
                response.raise_for_status()
                if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {begin}-{end}/{size}':
                    raise ValueError('Server did not honor exact range')
                if len(response.content)!=end-begin+1:raise ValueError('Truncated range')
                temp=target.with_suffix('.tmp');temp.write_bytes(response.content);temp.replace(target)
                return
            except Exception:
                if attempt==15:raise
                time.sleep(min(8,attempt+1))
    total=(size+chunk-1)//chunk
    with ThreadPoolExecutor(max_workers=12) as pool:
        for n,_ in enumerate(pool.map(part,range(total)),1):
            if n%100==0 or n==total:print(repo,filename,n,'/',total,'ranges',flush=True)
    temp=path.with_name(path.name+'.assembled')
    with temp.open('wb') as output:
        for i in range(total):output.write((parts/f'{i:06d}').read_bytes())
    if digest(temp)!=expected:raise ValueError('Official LFS SHA256 mismatch')
    temp.replace(path)
    for p in parts.iterdir():p.unlink()
    parts.rmdir()

REPOS = {
    'qwen25vl7b': 'Qwen/Qwen2.5-VL-7B-Instruct',
    'llava16mistral7b': 'llava-hf/llava-v1.6-mistral-7b-hf',
    'llava_med7b': 'microsoft/llava-med-v1.5-mistral-7b',
    'chexagent8b': 'StanfordAIMI/CheXagent-8b',
    'maira2': 'microsoft/maira-2',
    'clip336': 'openai/clip-vit-large-patch14-336',
}

def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)

def download(model, root):
    from huggingface_hub import HfApi, snapshot_download
    folder = root / model
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / 'download_status.json'
    old = json.loads(state_path.read_text()) if state_path.exists() else {}
    state = dict(model=model, repo=REPOS[model], started=time.time(), pid=os.getpid(), status='resolving')
    atomic(state_path, state)
    try:
        info = HfApi().model_info(REPOS[model], revision=old.get('revision') or 'main',files_metadata=True)
        state.update(revision=info.sha, gated=info.gated, status='downloading')
        atomic(state_path, state)
        path = snapshot_download(REPOS[model], revision=info.sha, max_workers=4,
            ignore_patterns=['*.msgpack', '*.h5', '*.ot', '*.onnx', '*.gguf', '*.bin', '*.safetensors',
                             'optimizer.pt', 'training_args.bin', 'trainer_state.json'])
        files=[s for s in info.siblings if s.rfilename.endswith('.safetensors') or (model=='clip336' and s.rfilename=='pytorch_model.bin')]
        for record in files:
            if not record.lfs:raise ValueError('Missing official LFS hash')
            target=Path(path)/record.rfilename;target.parent.mkdir(parents=True,exist_ok=True)
            ranged_blob(REPOS[model],info.sha,record.rfilename,target,record.lfs.size,record.lfs.sha256)
        # Avoid silently declaring a tokenizer-only snapshot successful.
        shards = [Path(path)/r.rfilename for r in files]
        if not shards:
            raise RuntimeError('No safetensors weight files in snapshot')
        state.update(status='complete', path=path, finished=time.time(),
                     weight_files=[dict(name=r.rfilename,bytes=r.lfs.size,sha256=r.lfs.sha256) for r in files],
                     config_sha256=hashlib.sha256((Path(path)/'config.json').read_bytes()).hexdigest())
        atomic(state_path, state)
        print(model, 'complete', path, flush=True)
    except Exception as exc:
        state.update(status='blocked' if 'GatedRepo' in type(exc).__name__ else 'failed',
                     error_type=type(exc).__name__, error=str(exc)[:1200], finished=time.time())
        atomic(state_path, state)
        print(model, state['status'], type(exc).__name__, flush=True)
    return state['status'] == 'complete'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--models',default=','.join(REPOS))
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    os.umask(0o077)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results=list(pool.map(lambda m:download(m,args.root.resolve()),args.models.split(',')))
    raise SystemExit(0 if all(results) else 1)

if __name__=='__main__':main()
