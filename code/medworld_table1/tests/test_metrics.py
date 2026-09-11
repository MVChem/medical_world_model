import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import candidate_pools, clinical_metrics, retrieval


def test_fractional_ties_and_unknown_predictions():
    target = np.array([[0, 1], [0, 1], [1, 0]])
    pool = [dict(query=0, candidates=[0, 1, 2], mask=[True, True])]
    assert retrieval(target, target, pool)['future_r1'] == .5
    missing = np.full_like(target, -2)
    # Missing predictions do not create an empty/optimistically perfect mask.
    assert retrieval(missing, target, pool)['future_r1'] == 1/3


def test_stable_cases_count_towards_transition_false_positives():
    current = np.array([[0], [0], [1], [1]])
    target = np.array([[1], [0], [0], [1]])
    good = target.copy()
    bad = np.array([[1], [1], [0], [0]])
    a = clinical_metrics(current, target, good, None, ['edema'])
    b = clinical_metrics(current, target, bad, None, ['edema'])
    assert a['transition_f1'] == 1.
    assert np.isclose(b['transition_f1'], 2/3)


def test_reference_mask_and_probability_ap():
    c = np.zeros((5, 1), dtype=int)
    t = np.array([[1], [0], [1], [-2], [-1]])
    p = np.array([[-2], [0], [1], [1], [1]])
    scores = [[.9], [.1], [.8], [.99], [.99]]
    result = clinical_metrics(c, t, p, scores, ['edema'])
    assert result['finding_auprc'] == 1.
    assert result['per_finding']['edema']['evaluable'] == 3
    assert np.isclose(result['chexbert_f1'], 2/3)


def test_pool_patient_exclusion_and_matching():
    rows = [dict(id=str(i), patient=str(i//2), horizon=0, view='AP') for i in range(70)]
    labels = np.tile([[0, 1]], (70, 1))
    pools = candidate_pools(rows, labels, labels)
    assert len(pools) == 70
    for pool in pools:
        ids = pool['candidates']
        assert len(ids) == 32
        assert len({rows[i]['patient'] for i in ids}) == 32
    assert pools == candidate_pools(rows, labels, labels)
