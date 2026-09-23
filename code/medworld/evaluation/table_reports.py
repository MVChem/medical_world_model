"""Export complete, matched paper Tables 1 and 2 for the three current arms."""
import hashlib
import json
import math
from pathlib import Path

from ..datasets.protocol import _rows, _sha256
from ..runtime import atomic_json, read_checkpoint
from .future_metrics import SCHEMA as FUTURE_SCHEMA, reference_fingerprint, validate_radgraph_provenance
from .future_common import scoring_protocol
from .protocol import FUTURE_TASKS, TABLE1_METRICS, TABLE2_METRICS, metric_protocol, validate_metric_protocol
from .selection import validate_references


ARMS = ('baseline', 'qwen', 'slots')
LABELS = {'baseline': 'Raw-input baseline', 'qwen': 'Qwen3.5-9B zero-shot', 'slots': 'MedWorld + slots'}


def _json(path):
    return json.loads(Path(path).read_text())


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _metric(summary, key, *, days=False):
    value = summary.get(key)
    if (isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
            or value < 0 or (not days and value > 1)):
        raise ValueError(f'Tables require a finite complete {key} score; no pending numeric cells')
    return float(value)


def _trained_checkpoints(run, cfg):
    saved = {arm: read_checkpoint(run / arm / 'final.pt') for arm in ('baseline', 'slots')}
    baseline, slots = saved['baseline'], saved['slots']
    tasks = ('classification', 'segmentation', 'vqa', *FUTURE_TASKS)
    for arm, checkpoint in saved.items():
        settings, progress = checkpoint['config'], checkpoint['progress']
        if (settings['slot_conditioning'] != (arm == 'slots') or not settings.get('future_enabled')
                or Path(settings['qwen']).resolve() != Path(cfg['qwen']).resolve()):
            raise ValueError('Both trained arms must use the current future-enabled Qwen9B configuration')
        if progress.get('complete') is not True or progress.get('stopped'):
            raise ValueError('Tables require successfully completed training for both arms')
        if type(progress.get('step')) is not int or progress['step'] <= 0:
            raise ValueError('Tables require positive recorded optimizer update counts')
        if type(progress.get('world_size')) is not int or progress['world_size'] <= 0:
            raise ValueError('Tables require a recorded training world size')
        counts = progress.get('task_samples', {})
        if any(type(counts.get(task)) is not int or counts[task] <= 0 for task in tasks):
            raise ValueError('Tables require recorded training sample counts for all eight tasks')
    for key in ('step', 'world_size'):
        if baseline['progress'][key] != slots['progress'][key]:
            raise ValueError('Trained arms must have equal optimizer updates and world size')
    if any(baseline['progress']['task_samples'][task] != slots['progress']['task_samples'][task] for task in tasks):
        raise ValueError('Trained arms must have equal sample counts for all eight tasks')
    initialization = baseline.get('metadata', {}).get('task_initialization_sha256')
    if (not isinstance(initialization, str) or len(initialization) != 64
            or any(char not in '0123456789abcdef' for char in initialization)
            or initialization != slots.get('metadata', {}).get('task_initialization_sha256')):
        raise ValueError('Trained arms require identical recorded task initialization')
    if baseline['data_fingerprint'] != slots['data_fingerprint']:
        raise ValueError('Trained arms use different data fingerprints')
    if any(baseline['config'][key] != 0 for key in ('latent_weight', 'visual_consistency_weight')):
        raise ValueError('The raw baseline must disable slot auxiliary objectives')
    excluded = {'slot_conditioning', 'latent_weight', 'visual_consistency_weight', 'testing', 'baselines'}
    settings = [{key: value for key, value in checkpoint['config'].items() if key not in excluded}
                for checkpoint in (baseline, slots)]
    if settings[0] != settings[1]:
        raise ValueError('Trained arm configurations differ')
    return saved


def _summary(folder, table, task, fingerprint, *, checkpoint_sha=None, split='test'):
    summary = _json(folder / 'summary.json')
    validate_metric_protocol(summary, table)
    if (summary.get('limit') is not None or summary.get('partial') or summary.get('split') != split
            or summary.get('data_fingerprint') != fingerprint):
        raise ValueError('Tables require full matched test cohorts and data fingerprints')
    if checkpoint_sha is not None and summary.get('checkpoint_sha256') != checkpoint_sha:
        raise ValueError('Trained evaluation does not belong to final.pt')
    if summary.get('predictions_sha256', {}).get(task) != _sha256(folder / f'{task}.jsonl'):
        raise ValueError('Prediction SHA256 differs from the evaluation summary')
    records = _rows(folder / f'{task}.jsonl')
    if not records or len(records) != summary.get('tasks', {}).get(task, {}).get('n'):
        raise ValueError('Test records are empty or their denominators differ')
    return summary, records


def _native_provenance(run, cfg, weights):
    from medworld_zero_shot_eval.future_evaluate import native_readout_protocol
    for name, table in (('qwen', 'table2'), ('qwen_future', 'table1')):
        folder = run / name
        summary, status = _json(folder / 'summary.json'), _json(folder / 'status.json')
        validate_metric_protocol(summary, table)
        if (status.get('status') != 'complete' or status.get('partial') or summary.get('partial')
                or summary.get('model_id') != 'qwen9b'
                or Path(summary.get('model', '')).resolve() != Path(cfg['qwen']).resolve()
                or summary.get('weights_sha256') != weights):
            raise ValueError('Native results require completed Qwen9B tests with the exact pretrained weight hashes')
        if table == 'table1' and summary.get('readout_protocol') != native_readout_protocol(cfg):
            raise ValueError('Native future readout/codebook/output budgets differ from the current protocol')


def _segmentation(summary, records, expected):
    metrics = summary['tasks']['segmentation']
    groups = metrics.get('by_dataset', {})
    if set(groups) != set(expected):
        raise ValueError('Segmentation must contain exactly the required reviewed datasets')
    for dataset, group in groups.items():
        rows = [record for record in records if record['dataset'] == dataset]
        if (not rows or group.get('n_images') != len(rows)
                or group.get('n_volumes') != len({row['volume_id'] for row in rows})
                or group.get('n_patients') != len({row['patient'] for row in rows})):
            raise ValueError('Segmentation dataset image/volume/patient denominators differ')
        if any(group.get('active_channels') != row['active_channels'] or group.get('target_names') != row['target_names'] for row in rows):
            raise ValueError('Segmentation channel semantics differ')
        for key in TABLE2_METRICS['segmentation']:
            _metric(group, key)
    for key in TABLE2_METRICS['segmentation']:
        expected_mean = sum(group[key] for group in groups.values()) / len(groups)
        if not math.isclose(_metric(metrics, key), expected_mean, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError('Main segmentation score must be the equal dataset macro mean')
    return groups


def write_tables(run, cfg):
    run = Path(run)
    if (not cfg.get('future_enabled') or cfg.get('baselines') != {'no_slots': True, 'qwen': True}
            or set(cfg['testing']['tasks']) != set(TABLE2_METRICS)
            or set(cfg['testing']['future_tasks']) != set(FUTURE_TASKS)
            or not cfg['testing']['human_segmentation']):
        raise ValueError('Both complete tables require raw baseline, Qwen9B, slots and every configured test')
    from ..run_experiment import native_spec
    if native_spec(cfg)['id'] != 'qwen9b':
        raise ValueError('Current paper tables require Qwen3.5-9B')
    checkpoints = _trained_checkpoints(run, cfg)
    fingerprint = checkpoints['slots']['data_fingerprint']
    hashes = {arm: _sha256(run / arm / 'final.pt') for arm in checkpoints}
    weights = {path.name: _sha256(path) for path in sorted(Path(cfg['qwen']).glob('*.safetensors'))}
    if not weights:
        raise ValueError('Qwen9B local pretrained weight inventory is empty')
    source_hashes = {}
    for path in sorted(Path(cfg['qwen']).iterdir()):
        if path.is_file() and (path.suffix in ('.json', '.safetensors', '.model') or path.name == 'merges.txt'):
            source_hashes['qwen/' + path.name] = weights[path.name] if path.name in weights else _sha256(path)
    source_hashes['jepa'] = _sha256(Path(cfg['jepa']))
    source_fingerprint = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    if (checkpoints['slots']['weights_fingerprint'] != source_fingerprint
            or checkpoints['baseline']['weights_fingerprint'] != hashlib.sha256(b'medworld/raw_input_v1/no-pretrained-assets').hexdigest()):
        raise ValueError('Training pretrained assets differ from the native comparison inventory')
    _native_provenance(run, cfg, weights)
    data_protocols = [_json(run / folder / 'data_protocol.json') for folder in ('baseline', 'slots', 'qwen', 'qwen_future')]
    if (any(value != data_protocols[0] for value in data_protocols)
            or hashlib.sha256(json.dumps(data_protocols[0], sort_keys=True).encode()).hexdigest() != fingerprint):
        raise ValueError('Recorded training and native data protocols differ')
    expected_future_fingerprint = hashlib.sha256(json.dumps(data_protocols[0]['future'], sort_keys=True).encode()).hexdigest()
    table1 = {arm: {} for arm in ARMS}
    table2 = {arm: {} for arm in ARMS}
    denominators = {'table1': {}, 'table2': {}}
    provenance = {'table1': {}, 'table2': {}}
    future_fingerprints = set()
    for task, metric in TABLE1_METRICS.items():
        matched, protocols = [], []
        for arm in ARMS:
            folder = run / ('qwen_future' if arm == 'qwen' else f'{arm}/evaluation/{task}')
            summary, records = _summary(folder, 'table1', task, fingerprint, checkpoint_sha=hashes.get(arm))
            references = [{key: value for key, value in row.items() if key != 'prediction'} for row in records]
            expected = {'id', 'patient', 'target'} | ({'finding'} if task == 'progression' else set())
            if (any(set(row) != expected | {'prediction'} for row in records)
                    or len({reference['id'] for reference in references}) != len(references)):
                raise ValueError('Future records must contain the exact canonical reference fields and unique IDs')
            reference_hash = reference_fingerprint(references)
            recorded = summary.get('references', {}).get(task)
            if recorded != {'n': len(references), 'references_sha256': reference_hash}:
                raise ValueError('Future reference SHA256 differs from the frozen cohort')
            protocol = _json(folder / f'{task}_protocol.json')
            if protocol != scoring_protocol(task, references, cfg):
                raise ValueError('Future per-task scorer protocol differs from the current fixed definition')
            if task == 'future_report':
                validate_radgraph_provenance(protocol['radgraph'])
            scores = summary['tasks'][task]
            if (scores.get('schema') != FUTURE_SCHEMA or scores.get('task') != task
                    or scores.get('unit') != protocol['unit'] or scores.get('complete') is not True
                    or scores.get('reference_sha256') != reference_hash
                    or protocol.get('reference_sha256') != reference_hash
                    or scores.get('protocol_sha256') != _digest(protocol)):
                raise ValueError('Future scores require complete matching reference and scorer protocols')
            if not isinstance(summary.get('future_data_fingerprint'), str) or not summary['future_data_fingerprint']:
                raise ValueError('Future dataset fingerprint is required')
            future_fingerprints.add(summary['future_data_fingerprint'])
            table1[arm][metric if task != 'mortality_30d' else 'mortality_auroc'] = _metric(scores, metric, days=task == 'remaining_los')
            matched.append(references)
            protocols.append(protocol)
        if matched[0] != matched[1] or matched[0] != matched[2] or protocols[0] != protocols[1] or protocols[0] != protocols[2]:
            raise ValueError('Table 1 arms must use identical ordered references and scorer protocols')
        denominators['table1'][task] = len(matched[0])
        provenance['table1'][task] = {'references_sha256': reference_fingerprint(matched[0]), 'protocol_sha256': _digest(protocols[0])}
    if future_fingerprints != {expected_future_fingerprint}:
        raise ValueError('Table 1 future dataset fingerprints differ')
    internal = metric_protocol('table2')['segmentation_main_datasets']
    external = metric_protocol('table2')['segmentation_external_dataset']
    segmentation = {arm: {} for arm in ARMS}
    for task in ('classification', 'vqa', 'segmentation', 'segmentation_human'):
        actual_task = 'segmentation' if task == 'segmentation_human' else task
        matched, selections = [], []
        for arm in ARMS:
            if arm == 'qwen' and actual_task == 'segmentation':
                continue
            folder = run / ('qwen' if arm == 'qwen' else f'{arm}/evaluation/{task}')
            summary, records = _summary(folder, 'table2', actual_task, fingerprint,
                                        checkpoint_sha=hashes.get(arm), split='human_test' if task == 'segmentation_human' else 'test')
            manifest = validate_references(summary, actual_task, records)
            matched.append(manifest)
            if actual_task == 'vqa':
                selection = summary['vqa_selection']
                if selection.get('per_type') != cfg['testing']['vqa_per_type'] or selection.get('seed') != cfg['testing']['vqa_seed']:
                    raise ValueError('Table 2 VQA selection differs from the configured cohort')
                selections.append(selection)
            metrics = summary['tasks'][actual_task]
            if actual_task == 'segmentation':
                groups = _segmentation(summary, records, [external] if task == 'segmentation_human' else internal)
                segmentation[arm].update(groups)
            if task != 'segmentation_human':
                for key in TABLE2_METRICS[actual_task]:
                    table2[arm][key] = _metric(metrics, key)
        if any(value != matched[0] for value in matched) or any(value != selections[0] for value in selections):
            raise ValueError('Table 2 arms must use identical references and VQA selections')
        denominators['table2'][task] = matched[0]['n']
        provenance['table2'][task] = matched[0]
    for key in TABLE2_METRICS['segmentation']:
        table2['qwen'][key] = None
    for dataset in (*internal, external):
        left, right = segmentation['baseline'][dataset], segmentation['slots'][dataset]
        if any(left[key] != right[key] for key in ('n_images', 'n_volumes', 'n_patients', 'active_channels', 'target_names')):
            raise ValueError('Trained segmentation dataset denominators differ')
        segmentation['qwen'][dataset] = {'mean_dice': None, 'mean_iou': None,
                                          'reason': 'Native Qwen has no segmentation head'}
    result = {'schema': metric_protocol('table1')['schema'], 'methods': LABELS,
              'metric_protocols': {table: metric_protocol(table) for table in ('table1', 'table2')},
              'table1': table1, 'table2': table2, 'segmentation_by_dataset': segmentation,
              'denominators': denominators, 'provenance': provenance,
              'data_fingerprint': fingerprint, 'future_data_fingerprint': next(iter(future_fingerprints)),
              'checkpoint_sha256': hashes, 'native_weights_sha256': weights,
              'optimizer_updates': checkpoints['slots']['progress']['step'],
              'task_samples': checkpoints['slots']['progress']['task_samples']}
    def cells(arm, values, keys, days=False):
        return '| ' + LABELS[arm] + ' | ' + ' | '.join(
            'N/A' if values.get(key) is None else f'{values[key] if days and key == "mae_days" else values[key] * 100:.2f}'
            for key in keys) + ' |'
    first = ['# Table 1: Future prediction', '',
             'Scores are percentages except remaining length of stay, which is MAE in days. All arms use identical held-out references.', '',
             '| Method | Future VQA Acc. ↑ | Progression BAcc. ↑ | RadGraph F1 ↑ | Mortality AUROC ↑ | LOS MAE (days) ↓ |',
             '|---|---:|---:|---:|---:|---:|']
    keys1 = ('accuracy', 'balanced_accuracy', 'radgraph_f1', 'mortality_auroc', 'mae_days')
    first.extend(cells(arm, table1[arm], keys1, days=True) for arm in ARMS)
    first += ['', 'Test counts: ' + ', '.join(f'{task}={count:,}' for task, count in denominators['table1'].items()) + '.',
              'Native LOS uses the fixed day-grid probability expectation documented in [its readout protocol](qwen_future/summary.json).', '']
    second = ['# Table 2: Current downstream tasks', '',
              'All scores are percentages. Main segmentation scores equally average the three internal reviewed datasets; MRI slices are aggregated by volume.', '',
              '| Method | Classification AUC ↑ | AP ↑ | VQA EM ↑ | VQA micro-F1 ↑ | Dice ↑ | IoU ↑ |',
              '|---|---:|---:|---:|---:|---:|---:|']
    keys2 = ('macro_auroc', 'macro_ap', 'exact_match', 'micro_f1', 'mean_dice', 'mean_iou')
    second.extend(cells(arm, table2[arm], keys2) for arm in ARMS)
    second += ['', 'Test counts: ' + ', '.join(f'{task}={count:,}' for task, count in denominators['table2'].items()) + '.',
               '', '## Segmentation by dataset', '',
               '| Dataset | Evaluation units | Raw baseline Dice | IoU | Qwen9B Dice | IoU | Slots Dice | IoU |',
               '|---|---:|---:|---:|---:|---:|---:|---:|']
    for dataset in (*internal, external):
        left, right = segmentation['baseline'][dataset], segmentation['slots'][dataset]
        second.append(f'| {dataset} | {left["n_volumes"]} | {left["mean_dice"] * 100:.2f} | {left["mean_iou"] * 100:.2f} | N/A | N/A | {right["mean_dice"] * 100:.2f} | {right["mean_iou"] * 100:.2f} |')
    second += ['', 'Evaluation units are MRI volumes or CXR images. Montgomery is external and excluded from the three-dataset main mean. Native Qwen has no segmentation head.', '']
    atomic_json(run / 'tables.json', result)
    (run / 'TABLE1.md').write_text('\n'.join(first))
    (run / 'TABLE2.md').write_text('\n'.join(second))
    (run / 'COMPARISON.md').write_text('\n'.join(first + second) + '\n[Machine-readable tables](tables.json) · [Table 1](TABLE1.md) · [Table 2](TABLE2.md)\n')
    return result
