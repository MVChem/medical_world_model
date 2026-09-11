import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semantic_schema import FINDINGS, validate
from semantic_metrics import degree_interval, state, pair_events, score


def annotation(degree='', assertion='present', side='right', change='not_stated', evidence=None):
    a = dict(findings={name: [] for name in FINDINGS}, other_findings=[], devices=[], technical_factors=[])
    if assertion is not None:
        a['findings']['Pleural Effusion'] = [dict(assertion=assertion, laterality=side, site='', degree=degree,
            change=change, comparison_reference='', evidence=evidence or f'{degree} {side} pleural effusion.'.strip())]
    return a


def event(a, b, scope='overall', verified=None):
    return next(r for r in pair_events(a, b, verified) if r['finding'] == 'Pleural Effusion' and r['scope'] == scope)['event']


def test_large_to_small_penalizes_persistence_while_presence_still_matches():
    current, reference = annotation('large'), annotation('small')
    good = score([dict(id='x', current=current, reference=reference, prediction=reference)])
    bad = score([dict(id='x', current=current, reference=reference, prediction=current)])
    assert good['presence_macro_f1'] == bad['presence_macro_f1'] == 1
    assert good['endpoint_events']['overall']['change_macro_f1'] == 1
    assert bad['endpoint_events']['overall']['change_macro_f1'] == 0
    assert good['degree']['exact_accuracy'] == 1
    assert bad['degree']['exact_accuracy'] == 0
    assert bad['degree']['normalized_error'] > 0


@pytest.mark.parametrize('prediction', [None, annotation(assertion=None), annotation('small', assertion='uncertain')])
def test_prediction_unknown_or_failure_never_removes_reference_fields(prediction):
    row = dict(id='x', current=annotation('large'), reference=annotation('small'), prediction=prediction)
    missing = score([row]); good = score([dict(row, prediction=row['reference'])])
    assert missing['degree']['evaluable_fields'] == good['degree']['evaluable_fields']
    assert missing['event_evaluable_fields'] == good['event_evaluable_fields']
    assert missing['degree']['exact_accuracy'] == 0
    assert missing['endpoint_events']['overall']['change_macro_f1'] == 0


def test_ranges_overlap_and_equal_bins_are_not_proof_of_stability():
    assert degree_interval('Pleural Effusion', 'moderate-to-large') == [3, 4]
    assert degree_interval('Pleural Effusion', 'extensive') is None
    assert event(annotation('moderate-to-large'), annotation('small')) == 'improved'
    assert event(annotation('moderate-to-large'), annotation('moderate')) is None
    assert event(annotation('small'), annotation('small')) is None


def test_laterality_and_mixed_directions_are_preserved():
    c, t = annotation('large'), annotation('small')
    c['findings']['Pleural Effusion'] += annotation('small', side='left')['findings']['Pleural Effusion']
    t['findings']['Pleural Effusion'] += annotation('large', side='left')['findings']['Pleural Effusion']
    assert event(c, t) is None
    assert event(c, t, 'right') == 'improved'
    assert event(c, t, 'left') == 'worsened'
    assert event(annotation('large', side='right'), annotation('small', side='left')) is None


def test_partial_negation_and_conflicts_are_not_global_absence():
    assert state(annotation(assertion='absent', side='left'), 'Pleural Effusion')['presence'] is None
    assert state(annotation(assertion='absent', side='unspecified'), 'Pleural Effusion')['presence'] == 0
    a = annotation('large')
    a['findings']['Pleural Effusion'] += annotation(assertion='absent', side='unspecified')['findings']['Pleural Effusion']
    assert state(a, 'Pleural Effusion')['reason'] == 'conflicting'


def test_comparative_phrase_requires_verified_anchor():
    c, t = annotation('small'), annotation('small', change='decreased')
    assert event(c, t) is None
    verified = {'Pleural Effusion|overall': [t['findings']['Pleural Effusion'][0]['evidence']]}
    assert event(c, t, verified=verified) == 'improved'
    with pytest.raises(ValueError):
        event(c, t, verified={'Pleural Effusion|overall': ['invented evidence']})


def test_stable_reference_penalizes_false_improvement():
    c = annotation('large')
    stable = annotation('large', change='unchanged')
    small = annotation('small')
    verified = {'Pleural Effusion|overall': [stable['findings']['Pleural Effusion'][0]['evidence']]}
    result = score([
        dict(id='change', current=c, reference=small, prediction=small),
        dict(id='stable', current=c, reference=stable, prediction=small, reference_comparisons=verified)])
    improved = result['endpoint_events']['overall']['per_finding']['Pleural Effusion']['per_class']['improved']
    assert improved['tp'] == 1 and improved['fp'] == 1
    assert improved['f1'] == pytest.approx(2/3)


def test_evidence_is_checked_against_the_report():
    a = annotation('large')
    assert validate(a, 'large right pleural effusion.') == []
    assert validate(a, 'No pleural effusion.')
    a['findings']['Pleural Effusion'][0]['degree'] = 'small'
    assert validate(a, 'large right pleural effusion.')


def test_hallucinated_newness_without_change_language_fails_validation():
    a = annotation('large', change='new')
    assert any('lexical cue' in e for e in validate(a, 'large right pleural effusion.'))
