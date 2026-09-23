"""Matched, separately trained image-only versus image+slots task comparison."""
import argparse
import json
import math
from pathlib import Path
from ..architecture import architecture, baseline_label, uses_slot_branch, uses_temporal


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
        cfg['architecture'] = architecture(cfg)
    for key in ('latent_weight', 'visual_consistency_weight'):
        if ca.get(key) != 0:
            raise ValueError('Raw-input baseline must disable all auxiliary objectives')
        value = cb.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError('Slots auxiliary objective weights must be finite and nonnegative')
        ca.pop(key)
        cb.pop(key)
    initializations = [s.get('metadata', {}).get('task_initialization_sha256') for s in checkpoints]
    if any(not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value)
           for value in initializations) or initializations[0] != initializations[1]:
        raise ValueError('Comparison requires identical recorded task initialization SHA256')
    for cfg in (ca, cb):
        cfg.pop('slot_conditioning')
        cfg.pop('testing')
        cfg.pop('baselines', None)
    if ca != cb or a['progress']['step'] != b['progress']['step']:
        raise ValueError('Comparison requires equal configs and equal optimizer update counts')
    if a['progress'].get('world_size') != b['progress'].get('world_size'):
        raise ValueError('Comparison requires the same world size and effective batches')
    if any(s['progress'].get('stopped') or s['progress'].get('complete') is not True for s in checkpoints):
        raise ValueError('Comparison requires successfully completed training')
    if a['data_fingerprint'] != b['data_fingerprint']:
        raise ValueError('Training data differ')
    sample_counts = [s['progress'].get('task_samples') for s in checkpoints]
    if any(not isinstance(value, dict) for value in sample_counts) or any(
            type(sample_counts[0].get(task)) is not int or sample_counts[0][task] != sample_counts[1].get(task)
            for task in ('classification', 'segmentation', 'vqa')):
        raise ValueError('Current-task training sample counts differ or are missing')
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
            if summary.get('predictions_sha256', {}).get(task) != _sha256(directory / f'{task}.jsonl'):
                raise ValueError('Prediction file SHA256 differs from evaluation summary')
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
        if task == 'segmentation':
            groups = [s.get('by_dataset', {}) for s in summaries]
            if set(groups[0]) != set(groups[1]):
                raise ValueError('Segmentation source datasets differ')
            for dataset in sorted(groups[0]):
                left, right = groups[0][dataset], groups[1][dataset]
                if any(left[k] != right[k] for k in ('n_images', 'n_volumes', 'active_channels', 'target_names')):
                    raise ValueError('Segmentation dataset denominators or channel semantics differ')
                for key in keys:
                    x, y = left[key], right[key]
                    rows.append({'task': f'{folder}/{dataset}', 'metric': key, 'n': left['n_volumes'],
                                 'unit': 'MRI volumes or CXR images', 'baseline': x, 'slots': y, 'delta': y - x})
    out.mkdir(parents=True, exist_ok=True)
    updates = {'baseline': a['progress']['step'], 'slots': b['progress']['step']}
    arms = {name: {'label': baseline_label(saved['config']) if name == 'baseline' else 'Raw inputs + slots',
                   'slot_branch': uses_slot_branch(saved['config']), 'temporal_training': uses_temporal(saved['config']),
                   'latent_weight': saved['config']['latent_weight'],
                   'visual_consistency_weight': saved['config'].get('visual_consistency_weight', 0),
                   'weights_fingerprint': saved['weights_fingerprint'],
                   'task_initialization_sha256': saved.get('metadata', {}).get('task_initialization_sha256')}
            for name, saved in zip(('baseline', 'slots'), checkpoints)}
    atomic_json(out / 'comparison.json', {'baseline': str(baseline), 'conditioned': str(conditioned),
                'architecture': architecture(a['config']), 'arms': arms,
                'budget_mode': 'time' if ca['total_hours'] else 'steps', 'optimizer_steps': updates, 'rows': rows})
    def fmt(x):
        return 'N/A' if x is None else f'{x:.5f}'
    lines = ['# ' + baseline_label(a['config']) + ' vs inputs + slots', '',
             'Same task initialization, current-task data, effective batches and optimizer updates. Auxiliary objectives apply only to the slots arm when enabled.',
             f"Optimizer updates: no slots {updates['baseline']}, slots {updates['slots']}.", '',
             '| Arm | Slot branch | Temporal weight | VSSC weight |', '|---|---|---:|---:|']
    lines += [f"| {arm['label']} | {arm['slot_branch']} | {arm['latent_weight']} | {arm['visual_consistency_weight']} |" for arm in arms.values()]
    lines += ['', '| Task | Metric | N | ' + baseline_label(a['config']) + ' | With 8 slots | Delta |', '|---|---|---:|---:|---:|---:|']
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
