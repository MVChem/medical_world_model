"""Compare native Qwen and trained MedWorld only on identical test records."""
import json
from pathlib import Path

from ..datasets.protocol import _rows, _sha256
from ..runtime import atomic_json


def compare_native(conditioned, native, out, tasks):
    conditioned, native, out = map(Path, (conditioned, native, out))
    raw = json.loads((native / 'summary.json').read_text())
    status = json.loads((native / 'status.json').read_text())
    if (status.get('status') != 'complete' or raw.get('partial') or raw.get('limit') is not None
            or raw.get('split') != 'test' or raw.get('model_id') != 'qwen08b'):
        raise ValueError('Expected complete native Qwen 0.8B test results')
    checkpoint_hash = _sha256(conditioned / 'final.pt')
    protocol = json.loads((conditioned / 'data_protocol.json').read_text())
    import hashlib
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if raw.get('data_fingerprint') != fingerprint:
        raise ValueError('Native Qwen data fingerprint differs')
    rows = []
    for task in tasks:
        directory = conditioned / 'evaluation' / task
        trained = json.loads((directory / 'summary.json').read_text())
        if (trained.get('limit') is not None or trained.get('split') != 'test'
                or trained.get('checkpoint_sha256') != checkpoint_hash
                or trained.get('data_fingerprint') != fingerprint):
            raise ValueError('Expected matched test results on final.pt')
        records = []
        for folder, summary in ((directory, trained), (native, raw)):
            path = folder / f'{task}.jsonl'
            if summary.get('predictions_sha256', {}).get(task) != _sha256(path):
                raise ValueError('Prediction hash differs')
            data = _rows(path)
            if not data or len(data) != summary['tasks'][task]['n']:
                raise ValueError('Incomplete test predictions')
            records.append(data)
        fields = ['id', 'patient'] + (['labels'] if task == 'classification'
                                     else ['question', 'answer', 'semantic_type'])
        if (len(records[0]) != len(records[1])
                or any(any(a[k] != b[k] for k in fields) for a, b in zip(*records))):
            raise ValueError('Test IDs or references differ')
        if task == 'vqa' and trained.get('vqa_selection') != raw.get('vqa_selection'):
            raise ValueError('VQA sampling protocol differs')
        keys = ('macro_auroc', 'macro_ap') if task == 'classification' else ('exact_match', 'micro_f1')
        for key in keys:
            x, y = raw['tasks'][task][key], trained['tasks'][task][key]
            rows.append({'task': task, 'metric': key, 'n': len(records[0]), 'qwen': x, 'slots': y,
                         'delta': y - x if x is not None and y is not None else None})
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / 'qwen_comparison.json', {'native': str(native), 'conditioned': str(conditioned), 'rows': rows})
    return rows
