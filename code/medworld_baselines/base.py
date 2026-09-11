"""Paths and durable local IO for the pretrained-model evaluation sweep."""
import hashlib
import json
import os
from pathlib import Path

PROJECT = Path(os.environ.get('MEDWORLD_PROJECT', Path(__file__).resolve().parents[2]))
T1 = PROJECT / 'code/medworld_table1'
T2 = PROJECT / 'code/medworld_stage1'
CACHE = Path('/home/data2/chk/.cache/huggingface/hub')
FUTURE_FINDINGS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion', 'Pneumothorax']
CURRENT_FINDINGS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']

def read(path):
    return json.loads(Path(path).read_text())

def rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)

def models():
    def p(repo, revision):
        return str(CACHE / ('models--' + repo.replace('/', '--')) / 'snapshots' / revision)
    return [
        dict(id='qwen08b', label='Qwen3.5-0.8B', family='qwen', path=str(T1/'weights/Qwen3.5-0.8B'), tp=1),
        dict(id='qwen4b', label='Qwen3.5-4B', family='qwen', path=p('Qwen/Qwen3.5-4B','851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'), tp=1),
        dict(id='qwen9b', label='Qwen3.5-9B', family='qwen', path=p('Qwen/Qwen3.5-9B','c202236235762e1c871ad0ccb60c8ee5ba337b9a'), tp=1, endpoint='http://127.0.0.1:8120'),
        dict(id='qwen27b_fp8', label='Qwen3.5-27B-FP8', family='qwen', path=p('Qwen/Qwen3.5-27B-FP8','97f5941bf617e31c5e237364a8602ce3f03a551a'), tp=2),
        dict(id='medgemma4b', label='MedGemma-1.5-4B', family='gemma', path=p('google/medgemma-1.5-4b-it','e9792da5fb8ee651083d345ec4bce07c3c9f1641'), tp=1),
        dict(id='medgemma27b', label='MedGemma-27B', family='gemma', path=p('google/medgemma-27b-it','2d3e00ea38b50018bf5dd3aa1009457cd2d5a48f'), tp=4),
    ]
