"""Matched, separately trained image-only versus image+slots task comparison."""
import argparse
import json
from pathlib import Path


def compare(baseline, conditioned, out):
    from ..runtime import read_checkpoint, atomic_json
    from ..datasets.protocol import _rows, _sha256
    baseline, conditioned, out = map(Path, (baseline, conditioned, out))
    checkpoints = [read_checkpoint(r / 'final.pt') for r in (baseline, conditioned)]
    a, b = checkpoints
    if a['config']['slot_conditioning'] or not b['config']['slot_conditioning']:
        raise ValueError('Expected a trained no-slots baseline and a trained slots model')
    ca, cb = [dict(s['config']) for s in checkpoints]
    for cfg in (ca, cb):
        cfg.pop('slot_conditioning')
        cfg.pop('testing')
        cfg.pop('baselines', None)
    if ca != cb or a['progress']['step'] != b['progress']['step']:
        raise ValueError('Comparison requires equal configs and equal optimizer update counts')
    if a['progress'].get('world_size') != b['progress'].get('world_size'):
        raise ValueError('Comparison requires the same world size and effective batches')
    if any(s['progress'].get('stopped') or s['progress'].get('complete') is False for s in checkpoints):
        raise ValueError('Comparison requires successfully completed training')
    if a['data_fingerprint'] != b['data_fingerprint'] or a['weights_fingerprint'] != b['weights_fingerprint']:
        raise ValueError('Data or initial pretrained weights differ')
    from ..config import load_config
    testing = load_config(conditioned / 'config.json')['testing']
    other_testing = load_config(baseline / 'config.json')['testing']
    if not testing['enabled'] or any(testing[k] != other_testing[k] for k in ('enabled', 'tasks', 'human_segmentation', 'vqa_per_type', 'vqa_seed')):
        raise ValueError('Both runs must select the same enabled tests')
    rows = []
    for task, folder, keys in [('classification', 'classification', ('macro_auroc', 'macro_ap')),
                               ('segmentation', 'segmentation', ('mean_dice', 'mean_iou')),
                               ('segmentation', 'segmentation_human', ('mean_dice', 'mean_iou')),
                               ('vqa', 'vqa', ('exact_match', 'micro_f1'))]:
        if task not in testing['tasks'] or (folder == 'segmentation_human' and not testing['human_segmentation']):
            continue
        summaries, records = [], []
        for run, saved in zip((baseline, conditioned), checkpoints):
            directory = run / 'evaluation' / folder
            summary = json.loads((directory / 'summary.json').read_text())
            expected_split = 'human_test' if folder == 'segmentation_human' else 'test'
            if (summary['limit'] is not None or summary['split'] != expected_split
                    or summary['data_fingerprint'] != saved['data_fingerprint']
                    or summary['checkpoint_sha256'] != _sha256(run / 'final.pt')):
                raise ValueError('Expected full matched test results on final.pt')
            if task == 'vqa':
                selection = summary.get('vqa_selection', {})
                if (selection.get('per_type', 0) != testing['vqa_per_type'] or
                        selection.get('seed', 42) != testing['vqa_seed']):
                    raise ValueError('VQA sampling protocol differs from selected tests')
            summaries.append(summary['tasks'][task])
            records.append(_rows(directory / f'{task}.jsonl'))
        fields = ['id', 'patient'] + (['labels'] if task == 'classification' else ['question', 'answer'] if task == 'vqa' else [])
        if not records[0] or len(records[0]) != len(records[1]) or any(x['n'] != len(records[0]) for x in summaries):
            raise ValueError('Test denominators differ')
        if any(any(x[k] != y[k] for k in fields) for x, y in zip(*records)):
            raise ValueError('Test IDs or references differ')
        for key in keys:
            x, y = [s[key] for s in summaries]
            rows.append({'task': folder, 'metric': key, 'n': len(records[0]), 'baseline': x, 'slots': y,
                         'delta': y - x if x is not None and y is not None else None})
    out.mkdir(parents=True, exist_ok=True)
    updates = {'baseline': a['progress']['step'], 'slots': b['progress']['step']}
    atomic_json(out / 'comparison.json', {'baseline': str(baseline), 'conditioned': str(conditioned),
                'budget_mode': 'time' if ca['total_hours'] else 'steps', 'optimizer_steps': updates, 'rows': rows})
    def fmt(x):
        return 'N/A' if x is None else f'{x:.5f}'
    lines = ['# Image-only vs image + slots', '',
             'Same training settings, effective batches and exact optimizer update counts.',
             f"Optimizer updates: no slots {updates['baseline']}, slots {updates['slots']}.", '',
             '| Task | Metric | N | No slots | With 8 slots | Delta |', '|---|---|---:|---:|---:|---:|']
    lines += ['| ' + ' | '.join([r['task'], r['metric'], str(r['n']), fmt(r['baseline']), fmt(r['slots']), fmt(r['delta'])]) + ' |' for r in rows]
    (out / 'COMPARISON.md').write_text('\n'.join(lines) + '\n')
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', required=True)
    p.add_argument('--conditioned', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    compare(a.baseline, a.conditioned, a.out)
