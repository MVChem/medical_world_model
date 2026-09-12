"""Download the public NIH Montgomery images and left/right human lung masks."""
import argparse
import concurrent.futures
import re
import time
import requests
from PIL import Image
from common import DEFAULT_RUN, Path, atomic, digest

BASE = 'https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/'

def download(out):
    jobs = []
    for folder in ['CXR_png', 'ManualMask/leftMask', 'ManualMask/rightMask']:
        r = requests.get(BASE + folder + '/index.html', timeout=60)
        r.raise_for_status()
        names = sorted(set(re.findall(r"href=['\"]([^'\"]+\.png)['\"]", r.text)))
        assert len(names) == 138, (folder, len(names))
        jobs += [(folder + '/' + n) for n in names]
    def one(rel):
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            for attempt in range(5):
                try:
                    r = requests.get(BASE + rel, timeout=120)
                    r.raise_for_status()
                    tmp = path.with_suffix('.tmp')
                    tmp.write_bytes(r.content)
                    with Image.open(tmp) as im:
                        im.verify()
                    tmp.replace(path)
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)
        with Image.open(path) as im:
            im.verify()
        return {'path': rel, 'sha256': digest(path), 'bytes': path.stat().st_size}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        files = []
        for item in pool.map(one, jobs):
            files.append(item)
            if len(files) % 30 == 0:
                print('downloaded', len(files), '/', len(jobs), flush=True)
    atomic(out / 'manifest.json', dict(source=BASE, files=files, images=138,
        human_organs=['right lung', 'left lung'], use='external test only; no training or selection'))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, default=DEFAULT_RUN)
    a = p.parse_args()
    download(a.run / 'data/montgomery')
