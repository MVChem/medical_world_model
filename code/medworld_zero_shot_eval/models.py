"""Local pretrained checkpoints for native evaluations."""
import os
from pathlib import Path

PROJECT = Path(os.environ.get('MEDWORLD_PROJECT', Path(__file__).resolve().parents[2]))
CACHE = Path('/home/data2/chk/.cache/huggingface/hub')

def models():
    def p(repo, revision):
        return str(CACHE / ('models--' + repo.replace('/', '--')) / 'snapshots' / revision)
    return [
        dict(id='qwen08b', label='Qwen3.5-0.8B', family='qwen', path=str(PROJECT/'code/data/medworld/weights/Qwen3.5-0.8B'), tp=1),
        dict(id='qwen4b', label='Qwen3.5-4B', family='qwen', path=p('Qwen/Qwen3.5-4B','851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'), tp=1),
        dict(id='qwen9b', label='Qwen3.5-9B', family='qwen', path=p('Qwen/Qwen3.5-9B','c202236235762e1c871ad0ccb60c8ee5ba337b9a'), tp=1, endpoint='http://127.0.0.1:8120'),
        dict(id='qwen27b_fp8', label='Qwen3.5-27B-FP8', family='qwen', path=p('Qwen/Qwen3.5-27B-FP8','97f5941bf617e31c5e237364a8602ce3f03a551a'), tp=2),
        dict(id='medgemma4b', label='MedGemma-1.5-4B', family='gemma', path=p('google/medgemma-1.5-4b-it','e9792da5fb8ee651083d345ec4bce07c3c9f1641'), tp=1),
        dict(id='medgemma27b', label='MedGemma-27B', family='gemma', path=p('google/medgemma-27b-it','2d3e00ea38b50018bf5dd3aa1009457cd2d5a48f'), tp=4),
    ]
