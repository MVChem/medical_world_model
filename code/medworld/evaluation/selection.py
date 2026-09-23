"""Deterministic test cohorts and reference integrity for Table 2."""
import hashlib
import json
import random


REFERENCE_FIELDS = {
    'classification': ('id', 'patient', 'labels'),
    'vqa': ('id', 'patient', 'question', 'answer', 'semantic_type'),
    'segmentation': ('id', 'patient', 'dataset', 'volume_id', 'active_channels',
                     'target_names', 'target_sha256'),
}


def segmentation_reference_sha256(target, mask):
    """Bind segmentation comparisons to pixel targets and supervision masks."""
    digest = hashlib.sha256()
    for tensor in (target, mask):
        values = tensor.detach().cpu().contiguous().numpy()
        digest.update(str((values.shape, str(values.dtype))).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def reference_manifest(task, records):
    """Fingerprint the ordered cohort and targets without model predictions."""
    if task not in REFERENCE_FIELDS:
        raise ValueError(f'Unsupported Table 2 task: {task}')
    fields = REFERENCE_FIELDS[task]
    try:
        references = [{key: row[key] for key in fields} for row in records]
    except KeyError as error:
        raise ValueError(f'{task}: missing test reference field {error.args[0]}') from error
    ids = [row['id'] for row in references]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f'{task}: test IDs must be nonempty and unique')
    return {'n': len(ids), 'ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            'references_sha256': hashlib.sha256(json.dumps(
                references, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()}


def validate_references(summary, task, records):
    manifest = reference_manifest(task, records)
    if summary.get('references', {}).get(task) != manifest:
        raise ValueError(f'{task}: test IDs or references differ from recorded cohort')
    if summary.get('tasks', {}).get(task, {}).get('n') != manifest['n']:
        raise ValueError(f'{task}: test denominator differs from recorded cohort')
    if task == 'vqa' and summary.get('limit') is None:
        selection = summary.get('vqa_selection', {})
        ids = [row['id'] for row in records]
        if (selection.get('ids') != ids or selection.get('n') != len(ids)
                or selection.get('ids_sha256') != manifest['ids_sha256']):
            raise ValueError('VQA sampling protocol differs from evaluated test IDs')
    return manifest


def select_vqa(rows, per_type=0, seed=42):
    if type(per_type) is not int or per_type < 0:
        raise ValueError('VQA sample size must be a nonnegative integer')
    if type(seed) is not int:
        raise ValueError('VQA sampling seed must be an integer')
    ids = [row['id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('VQA test IDs must be unique')
    if not per_type:
        indices = list(range(len(rows)))
    else:
        groups = {kind: [] for kind in ('verify', 'choose', 'query')}
        for i, row in enumerate(rows):
            groups[row['semantic_type'].lower()].append(i)
        rng = random.Random(seed)
        indices = []
        for kind, group in groups.items():
            if len(group) < per_type:
                raise ValueError(f'Not enough {kind} questions')
            group.sort(key=lambda i: rows[i]['id'])
            indices.extend(rng.sample(group, per_type))
        indices.sort(key=lambda i: rows[i]['id'])
    ids = [rows[i]['id'] for i in indices]
    return indices, {'per_type': per_type, 'seed': seed, 'n': len(ids),
                     'ids': ids, 'ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
