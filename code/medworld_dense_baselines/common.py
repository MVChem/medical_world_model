"""Independent frozen-VLM dense probes; no world-model checkpoints."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get('MEDWORLD_PROJECT', ROOT.parent.parent)).resolve()
DEFAULT_RUN = ROOT / 'runs/dense_20260912'
OLD = PROJECT / 'code/medworld_stage1/data/overnight_20260910'
LINKED = PROJECT / 'code/mimic_cxr_iv_linked/runs/full_20260909'
GOLD = Path('/home/data1/data/MIMIC/MIMIC_CXR/chest-imagenome-dataset-1.0.0/gold_dataset')
MODEL_FILE = Path(os.environ.get('MEDWORLD_MODEL_FILE',
    PROJECT / 'code/medworld_baselines/runs/raw_models_20260911/models.json'))
SEED = 20260912

def read_rows(path):
    with Path(path).open() as f:
        return [json.loads(x) for x in f if x.strip()]

def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    os.replace(tmp, path)

def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def rank(value):
    return hashlib.sha256(f'{SEED}:{value}'.encode()).hexdigest()

def load_model_spec(mid):
    return next(m for m in json.loads(MODEL_FILE.read_text()) if m['id'] == mid)
