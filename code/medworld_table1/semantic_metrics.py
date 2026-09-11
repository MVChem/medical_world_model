"""Deterministic, reference-masked scoring of frozen semantic annotations.

Grades encode explicit report words, not validated clinical severity thresholds.
No automatic use of unanchored comparative phrases or LLM self-confidence.
"""
from collections import Counter, defaultdict
import re
from semantic_schema import FINDINGS

RULE_VERSION = 'endpoint-semantic-rules-v0.1'
EVENTS = ['onset', 'worsened', 'improved', 'resolution', 'stable_present', 'stable_absent']
CHANGE_EVENTS = EVENTS[:4]
SCOPES = ['overall', 'left', 'right']
DEGREE_WORDS = {
    'Pleural Effusion': {'trace': 1, 'tiny': 1, 'minimal': 1, 'small': 2, 'moderate': 3, 'large': 4},
    'Pneumothorax': {'trace': 1, 'tiny': 1, 'minimal': 1, 'small': 2, 'moderate': 3, 'large': 4},
}
for _name in FINDINGS:
    if _name not in DEGREE_WORDS:
        DEGREE_WORDS[_name] = {'minimal': 1, 'mild': 2, 'moderate': 3, 'severe': 4, 'marked': 4}


def degree_interval(name, text):
    if not text: return None
    tokens = re.findall(r'[a-z]+', text.lower())
    words = DEGREE_WORDS[name]
    if any(t not in words and t not in {'to', 'and'} for t in tokens): return None
    grades = [words[t] for t in tokens if t in words]
    return [min(grades), max(grades)] if grades else None


def _scope_mentions(mentions, scope):
    if scope == 'overall': return mentions
    return [m for m in mentions if m['laterality'] in (scope, 'bilateral') or
            (m['laterality'] == 'unspecified' and m['assertion'] == 'absent')]


def state(annotation, name, scope='overall'):
    """Unknown/conflicting -> no reference eligibility; no inference from silence."""
    mentions = _scope_mentions(annotation['findings'][name], scope)
    if not mentions: return dict(presence=None, reason='not_mentioned', degree=None, site=None)
    present = [m for m in mentions if m['assertion'] == 'present']
    absent = [m for m in mentions if m['assertion'] == 'absent']
    uncertain = [m for m in mentions if m['assertion'] == 'uncertain']
    # Opposite sides can consistently be present and absent. A global denial conflicts.
    conflict = any(a['laterality'] in ('unspecified', 'bilateral') or
                   p['laterality'] in ('unspecified', 'bilateral') or a['laterality'] == p['laterality']
                   for a in absent for p in present)
    if conflict: return dict(presence=None, reason='conflicting', degree=None, site=None)
    if present:
        grade, site = None, None
        # Never collapse different sides/regions into a maximum degree.
        regions = {(m['laterality'], m['site'].lower().strip()) for m in present}
        grades = [degree_interval(name, m['degree']) for m in present]
        if len(regions) == 1 and not uncertain and all(g is not None for g in grades) and all(g == grades[0] for g in grades):
            grade = grades[0]; site = next(iter(regions))[1]
        return dict(presence=1, reason='present', degree=grade, site=site)
    if uncertain: return dict(presence=None, reason='uncertain', degree=None, site=None)
    # A unilateral negative alone cannot establish whole-patient absence.
    covers_all = any(m['laterality'] in ('unspecified', 'bilateral') for m in absent) or {'left', 'right'} <= {m['laterality'] for m in absent}
    if absent and (scope != 'overall' or covers_all):
        return dict(presence=0, reason='absent', degree=None, site=None)
    return dict(presence=None, reason='partial_negative', degree=None, site=None)


def transition(current, future):
    c, t = current['presence'], future['presence']
    if c is None or t is None: return None
    if c == 0 and t == 1: return 'onset'
    if c == 1 and t == 0: return 'resolution'
    if c == t == 0: return 'stable_absent'
    a, b = current['degree'], future['degree']
    if a is None or b is None or current['site'] != future['site']: return None
    if b[1] < a[0]: return 'improved'
    if b[0] > a[1]: return 'worsened'
    # Identical bins/ranges still permit within-bin improvement/worsening.
    return None


def reported_changes(annotation, name, scope):
    return sorted({m['change'] for m in _scope_mentions(annotation['findings'][name], scope)
                   if m['change'] != 'not_stated'})


def pair_events(current, future, verified_comparisons=None):
    """Auditable event rows; comparative phrases retained, never silently anchored."""
    rows = []
    verified_comparisons = verified_comparisons or {}
    for name in FINDINGS:
        for scope in SCOPES:
            c, t = state(current, name, scope), state(future, name, scope)
            event = transition(c, t)
            # Overall degree may be unilateral; laterality compatibility is essential.
            if scope == 'overall' and event in ('improved', 'worsened'):
                cp = {m['laterality'] for m in current['findings'][name] if m['assertion'] == 'present'}
                tp = {m['laterality'] for m in future['findings'][name] if m['assertion'] == 'present'}
                if cp != tp: event = None
            # External adjudication must identify exact source evidence, per finding/scope.
            # LLM confidence or "previous study" alone never supplies this mapping.
            verified = verified_comparisons.get(name+'|'+scope, [])
            relevant = _scope_mentions(future['findings'][name], scope)
            if any(e not in [m['evidence'] for m in relevant] for e in verified):
                raise ValueError('Verified comparison evidence is not in the finding/scope')
            anchored = {m['change'] for m in relevant if m['evidence'] in verified and m['change'] != 'not_stated'}
            mapped = None
            if len(anchored) == 1 and c['presence'] is not None and t['presence'] is not None:
                change = next(iter(anchored))
                if c['presence'] == t['presence'] == 1:
                    mapped = {'decreased': 'improved', 'increased': 'worsened', 'unchanged': 'stable_present'}.get(change)
                elif c['presence'] == 0 and t['presence'] == 1 and change == 'new': mapped = 'onset'
                elif c['presence'] == 1 and t['presence'] == 0 and change == 'resolved': mapped = 'resolution'
            conflict = bool(mapped and event and mapped != event)
            if conflict: event = None
            elif mapped: event = mapped
            rows.append(dict(finding=name, scope=scope, current=c, future=t, event=event,
                reported_change=reported_changes(future, name, scope),
                comparison_anchor='externally_verified' if verified else 'unverified',
                event_basis='conflicting_evidence' if conflict else 'verified_comparison' if mapped else
                    'explicit_endpoint_states' if event else 'insufficient_endpoint_evidence'))
    return rows


def macro_f1(truth, predicted, classes):
    detail = {}
    for label in classes:
        tp = sum(t == label and p == label for t, p in zip(truth, predicted))
        fp = sum(t != label and p == label for t, p in zip(truth, predicted))
        fn = sum(t == label and p != label for t, p in zip(truth, predicted))
        support = sum(t == label for t in truth)
        detail[str(label)] = dict(tp=tp, fp=fp, fn=fn, support=support,
            f1=2*tp/(2*tp+fp+fn) if support else None)
    supported = [d['f1'] for d in detail.values() if d['f1'] is not None]
    return dict(macro_f1=sum(supported)/len(supported) if supported else None, per_class=detail)


def score(cohort):
    """Rows contain id,current,reference,prediction; prediction=None is a failed extraction.

    All metric denominators use current/reference only. Reference annotations must
    be validated upstream. Prediction failures remain misses on those fixed fields.
    """
    ids = [r['id'] for r in cohort]
    if len(ids) != len(set(ids)): raise ValueError('Duplicate evaluation IDs')
    presence = {name: [[], []] for name in FINDINGS}
    event_data = {scope: {name: [[], []] for name in FINDINGS} for scope in SCOPES}
    degree_n = degree_matches = 0
    degree_errors, missing_degrees, fields, categorical, failures = [], 0, 0, [], 0
    for row in cohort:
        c, r, p = row['current'], row['reference'], row['prediction']
        revents = pair_events(c, r, row.get('reference_comparisons'))
        pevents = {(v['finding'], v['scope']): v for v in pair_events(c, p, row.get('prediction_comparisons'))} if p else {}
        failures += p is None
        for name in FINDINGS:
            truth = state(r, name)['presence']
            pred = state(p, name)['presence'] if p else None
            if truth is not None:
                presence[name][0].append(truth); presence[name][1].append(pred)
        for ref in revents:
            name, scope = ref['finding'], ref['scope']
            pred = pevents.get((name, scope))
            if ref['event'] is not None:
                ts, ps = event_data[scope][name]
                ts.append(ref['event']); ps.append(pred['event'] if pred else None)
                fields += 1
            # Degree correctness is reference-defined, including predictions of absence.
            rd = ref['future']['degree']
            if rd is not None:
                degree_n += 1
                pd = pred['future']['degree'] if pred else None
                compatible = bool(pred and pred['future']['site'] == ref['future']['site'])
                if scope == 'overall' and p:
                    rside = {m['laterality'] for m in r['findings'][name] if m['assertion'] == 'present'}
                    pside = {m['laterality'] for m in p['findings'][name] if m['assertion'] == 'present'}
                    compatible = compatible and rside == pside
                if pd is None or not compatible:
                    missing_degrees += 1; error = 1.
                else:
                    degree_matches += pd == rd
                    # Endpoint discrepancy, normalized to the 1..4 ordinal span.
                    error = (abs(rd[0]-pd[0])+abs(rd[1]-pd[1]))/6
                degree_errors.append(error)
            if ref['event'] or ref['reported_change']:
                categorical.append(dict(id=row['id'], finding=name, scope=scope,
                    reference_event=ref['event'], predicted_event=pred['event'] if pred else None,
                    reference_reported_change=ref['reported_change']))
    presence_detail = {name: macro_f1(*values, [1]) for name, values in presence.items()}
    pfs = [v['macro_f1'] for v in presence_detail.values() if v['macro_f1'] is not None]
    events = {}
    for scope, per_finding in event_data.items():
        detail = {name: macro_f1(*values, EVENTS) for name, values in per_finding.items()}
        supported = [x['f1'] for v in detail.values() for x in v['per_class'].values() if x['f1'] is not None]
        changed = [v['per_class'][k]['f1'] for v in detail.values() for k in CHANGE_EVENTS if v['per_class'][k]['f1'] is not None]
        events[scope] = dict(macro_f1=sum(supported)/len(supported) if supported else None,
            change_macro_f1=sum(changed)/len(changed) if changed else None,
            evaluable_fields=sum(len(v[0]) for v in per_finding.values()), per_finding=detail)
    return dict(version=RULE_VERSION, n=len(cohort), prediction_extraction_failures=failures,
        presence_macro_f1=sum(pfs)/len(pfs) if pfs else None, presence=presence_detail,
        endpoint_events=events, event_evaluable_fields=fields,
        degree=dict(evaluable_fields=degree_n, exact_accuracy=degree_matches/degree_n if degree_n else None,
            normalized_error=sum(degree_errors)/degree_n if degree_n else None,
            missing_or_incompatible_predictions=missing_degrees,
            aggregation='Exploratory field mean; overall/side fields overlap. Not a Table 1 metric.'),
        diagnostics=categorical)
