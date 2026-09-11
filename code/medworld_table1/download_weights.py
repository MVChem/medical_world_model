"""Download public model weights with verified ranged requests. No dataset input."""
import argparse
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

import httpx


def download(repo, filename, out, workers=12, endpoint='https://hf-mirror.com', use_env=False):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # The mirror transports the same immutable Hub blob; require LFS SHA-256.
    with httpx.Client(trust_env=use_env, timeout=60, follow_redirects=True) as c:
        response = c.get(f'{endpoint}/api/models/{repo}?blobs=true')
        response.raise_for_status()
        info = response.json()
        record = next(s for s in info['siblings'] if s['rfilename'] == filename)
        size, expected = record['lfs']['size'], record['lfs']['sha256']
    revision = info['sha']
    if out.exists() and out.stat().st_size == size:
        h = hashlib.sha256()
        with out.open('rb') as f:
            for block in iter(lambda: f.read(1 << 20), b''):
                h.update(block)
        if h.hexdigest() == expected:
            print(f'already verified {out}', flush=True)
            return
    part_dir = out.with_name(out.name + '.parts')
    part_dir.mkdir(exist_ok=True)
    chunk_size = 4 * 1024 * 1024
    def part(i):
        start, end = i * chunk_size, min(size, (i+1)*chunk_size)-1
        dest = part_dir / f'{i:05d}'
        if dest.exists() and dest.stat().st_size == end-start+1:
            return i
        temporary = dest.with_suffix('.tmp')
        for attempt in range(16):
            try:
                offset = temporary.stat().st_size if temporary.exists() else 0
                if offset > end-start+1:
                    raise ValueError('Oversized partial range')
                with httpx.Client(trust_env=use_env, timeout=25, follow_redirects=True) as client:
                    while offset < end-start+1:
                        a, b = start+offset, min(end, start+offset+1048576-1)
                        url = f'{endpoint}/{repo}/resolve/{revision}/{filename}?download=true&part={i}&offset={offset}'
                        with client.stream('GET', url, headers={'Range': f'bytes={a}-{b}'}) as response:
                            response.raise_for_status()
                            if response.status_code != 206 or response.headers.get('content-range') != f'bytes {a}-{b}/{size}':
                                raise ValueError('Server did not honor the requested range')
                            with temporary.open('ab') as f:
                                for block in response.iter_bytes():
                                    f.write(block)
                        offset = temporary.stat().st_size
                if offset != end-start+1:
                    raise ValueError('Incomplete range')
                temporary.replace(dest)
                return i
            except Exception as exc:
                print(f'retry chunk={i} attempt={attempt+1} reason={type(exc).__name__}', flush=True)
                if attempt == 15:
                    raise
                time.sleep(min(2**attempt, 8))
    total = (size + chunk_size-1)//chunk_size
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for n, _ in enumerate(pool.map(part, range(total)), 1):
            if n % 20 == 0 or n == total:
                print(f'{filename}: {n}/{total} chunks', flush=True)
    h = hashlib.sha256()
    tmp = out.with_name(out.name + '.assembled')
    with tmp.open('wb') as f:
        for i in range(total):
            block = (part_dir / f'{i:05d}').read_bytes()
            h.update(block)
            f.write(block)
    if h.hexdigest() != expected:
        raise ValueError('Downloaded blob SHA-256 mismatch')
    tmp.replace(out)
    out.with_name(out.name + '.provenance.json').write_text(json.dumps(dict(repo=repo, revision=revision,
        filename=filename, sha256=expected, size=size), indent=2))
    for p in part_dir.iterdir():
        p.unlink()
    part_dir.rmdir()
    print(f'verified {out}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('repo')
    p.add_argument('filename')
    p.add_argument('out')
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--endpoint', default='https://hf-mirror.com')
    p.add_argument('--use-env', action='store_true')
    a = p.parse_args()
    download(a.repo, a.filename, a.out, a.workers, a.endpoint, a.use_env)
