"""Strict VQA label-set scoring; malformed answers are errors, never dropped."""
import json
from ...datasets.vqa import VOCABULARY


def vqa_metrics(records):
    vocabulary = {x.lower() for x in VOCABULARY}
    def score(rows):
        tp = fp = fn = exact = invalid = 0
        for row in rows:
            target = {x.lower().strip() for x in row['answer']}
            try:
                value = json.loads(row['prediction'])
                if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
                    raise ValueError('Expected JSON string array')
                predicted = {x.lower().strip() for x in value}
                if not predicted <= vocabulary:
                    raise ValueError('Unknown answer')
            except (ValueError, TypeError):
                predicted = {'__invalid__'}
                invalid += 1
            tp += len(target & predicted)
            fp += len(predicted - target)
            fn += len(target - predicted)
            exact += target == predicted
        return {'n': len(rows), 'exact_match': exact / len(rows) if rows else None,
                'micro_f1': 2 * tp / max(1, 2 * tp + fp + fn), 'invalid': invalid}
    return {**score(records), 'by_type': {kind: score([r for r in records if r['semantic_type'] == kind])
            for kind in sorted({r['semantic_type'] for r in records})}}
