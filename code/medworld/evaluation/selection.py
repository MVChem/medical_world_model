"""Answer-independent deterministic sampling of VQA test questions."""
import hashlib
import json
import random


def select_vqa(rows, per_type=0, seed=42):
    if type(per_type) is not int or per_type < 0:
        raise ValueError('VQA sample size must be a nonnegative integer')
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
