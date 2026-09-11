"""Plan Table 1 metrics. Unknown labels never shrink a prediction-dependent mask."""
import hashlib
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score


def f1_binary(truth, prediction):
    truth, prediction = np.asarray(truth, bool), np.asarray(prediction, bool)
    tp, fp, fn = (truth & prediction).sum(), (~truth & prediction).sum(), (truth & ~prediction).sum()
    denominator = 2*tp + fp + fn
    return float(2*tp/denominator) if denominator else 0.


def mean_supported(values):
    valid = [v for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def clinical_metrics(current, target, predicted, scores, findings):
    current, target, predicted = map(np.asarray, (current, target, predicted))
    valid = (target == 0) | (target == 1)
    joint = valid & ((current == 0) | (current == 1))
    aps, fs, events, details = [], [], [], {}
    for k, name in enumerate(findings):
        mask = valid[:, k]
        gt, pred = target[mask, k] == 1, predicted[mask, k] == 1
        # Support is defined exclusively from references, shared by all methods.
        f1 = f1_binary(gt, pred) if gt.any() else None
        ap = None
        if scores is not None and gt.any() and (~gt).any():
            ap = float(average_precision_score(gt, np.asarray(scores)[mask, k]))
        fs.append(f1)
        aps.append(ap)
        mask2 = joint[:, k]
        c, t, p = current[mask2, k], target[mask2, k], predicted[mask2, k]
        result = {}
        for a, b, event in [(0, 1, 'onset'), (1, 0, 'resolution')]:
            true_event, pred_event = (c == a) & (t == b), (c == a) & (p == b)
            score = f1_binary(true_event, pred_event) if true_event.any() else None
            events.append(score)
            result[event] = dict(f1=score, support=int(true_event.sum()))
        details[name] = dict(ap=ap, f1=f1, evaluable=int(mask.sum()), positives=int(gt.sum()),
                             transition_evaluable=int(mask2.sum()), **result)
    return dict(finding_auprc=mean_supported(aps), chexbert_f1=mean_supported(fs), transition_f1=mean_supported(events),
                supported_findings=sum(v is not None for v in fs), supported_events=sum(v is not None for v in events),
                per_finding=details)


def candidate_pools(rows, current, target, seed=42, negatives=31):
    current, target = np.asarray(current), np.asarray(target)
    valid = (target == 0) | (target == 1)
    groups = defaultdict(list)
    keys = []
    for i, row in enumerate(rows):
        # A fixed coarse current state, matching view, time bin and truth coverage.
        key = (row['horizon'], row['view'], min(int((current[i] == 1).sum()), 2), tuple(valid[i]))
        keys.append(key)
        if valid[i].any():
            groups[key].append(i)
    pools = []
    for i, row in enumerate(rows):
        candidates = [j for j in groups[keys[i]] if rows[j]['patient'] != row['patient']]
        candidates.sort(key=lambda j: hashlib.sha256(f'{seed}:{row["id"]}:{rows[j]["id"]}'.encode()).hexdigest())
        chosen, patients = [], set()
        for j in candidates:
            if rows[j]['patient'] not in patients:
                patients.add(rows[j]['patient'])
                chosen.append(j)
                if len(chosen) == negatives:
                    break
        if len(chosen) == negatives:
            pools.append(dict(query=i, candidates=[i]+chosen, mask=valid[i].tolist()))
    return pools


def retrieval(predicted, target, pools):
    predicted, target = np.asarray(predicted), np.asarray(target)
    scores = []
    for pool in pools:
        i, js, mask = pool['query'], pool['candidates'], np.asarray(pool['mask'], bool)
        assert mask.any()
        distance = (predicted[i][None, mask] != target[js][:, mask]).mean(1)
        tied = np.isclose(distance, distance.min(), rtol=0, atol=1e-12)
        scores.append(float(1/tied.sum()) if tied[0] else 0.)
    return dict(future_r1=float(np.mean(scores)) if scores else None, retrieval_queries=len(scores),
                retrieval_coverage=len(scores)/len(predicted) if len(predicted) else 0.)
