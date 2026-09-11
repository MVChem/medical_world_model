"""CPU-only audit of saved pilot outputs. Does not change training or official scores."""
import ast
import collections
import datetime
import json
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

from metrics import clinical_metrics

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/pilot_20260908_8h'
OUT = RUN / 'analysis_20260909'
OUT.mkdir(exist_ok=True, mode=0o700)
os.umask(0o077)


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


cohort = rows(ROOT / 'data/pilot/test.jsonl')
findings = read(RUN / 'config.json')['findings']
names = ['copy', 'direct', 'matched', 'ours']
display = {'copy': 'Copy current', 'direct': 'Direct Qwen', 'matched': 'Frozen encoder', 'ours': 'MedWorld'}
colors = {'copy': '#999999', 'direct': '#3974b5', 'matched': '#d48a36', 'ours': '#188979'}
patients, patient_idx = np.unique([r['patient'] for r in cohort], return_inverse=True)
data = {}
summary = {'n': len(cohort), 'patients': len(patients), 'models': {}}
for name in names:
    directory = RUN / name / ('evaluation' if name != 'copy' else '')
    metric = read(directory / 'metrics.json')
    labels = read(directory / 'chexbert_labels.json')
    current, target, predicted = [np.asarray(labels[k]) for k in ['current', 'target', 'predicted']]
    generated = rows(directory / 'predictions.jsonl')
    assert [r['id'] for r in generated] == [r['id'] for r in cohort]
    scores = np.asarray([r['scores'] for r in generated]) if name != 'copy' else None
    verified = clinical_metrics(current, target, predicted, scores, findings)
    for key in ['finding_auprc', 'chexbert_f1', 'transition_f1']:
        assert verified[key] == metric[key], (name, key)
    if data:
        assert np.array_equal(current, data['copy']['current'])
        assert np.array_equal(target, data['copy']['target'])
    counts = collections.Counter(r['report'] for r in generated)
    top_text, top_count = counts.most_common(1)[0]
    top_idx = next(i for i, r in enumerate(generated) if r['report'] == top_text)
    unique_labels, label_counts = np.unique(predicted, axis=0, return_counts=True)
    valid = np.isin(current, [0, 1]) & np.isin(target, [0, 1])
    change = valid & (current != target)
    stable = valid & (current == target)
    predicted_change = valid & np.isin(predicted, [0, 1]) & (current != predicted)
    tp, fp, fn = int((change & predicted_change).sum()), int((stable & predicted_change).sum()), int((change & ~predicted_change).sum())
    event_fs = [v[event]['f1'] for v in metric['per_finding'].values() for event in ['onset', 'resolution'] if v[event]['support'] >= 5]
    without_single_atelectasis = [v[event]['f1'] for disease, v in metric['per_finding'].items() for event in ['onset', 'resolution'] if v[event]['support'] and (disease, event) != ('Atelectasis', 'onset')]
    result = {k: metric[k] for k in ['finding_auprc', 'chexbert_f1', 'transition_f1', 'radgraph_f1']}
    result.update(unique_reports=len(counts), top_report_count=top_count, top5_report_count=sum(v for _, v in counts.most_common(5)),
                  unique_label_patterns=len(unique_labels), top_label_pattern_count=int(label_counts.max()),
                  median_report_words=float(np.median([len(r['report'].split()) for r in generated])),
                  prediction_label_counts={d: {str(v): int((predicted[:, j] == v).sum()) for v in [-2, -1, 0, 1]} for j, d in enumerate(findings)},
                  events=dict(tp=tp, fp=fp, fn=fn, micro_f1=2*tp/(2*tp+fp+fn), stable_fields=int(stable.sum()), stable_false_change_rate=fp/int(stable.sum())),
                  transition_without_atelectasis_onset=float(np.mean(without_single_atelectasis)),
                  transition_support_at_least_5=float(np.mean(event_fs)))
    if name == 'ours':
        fixed = clinical_metrics(current, target, np.repeat(predicted[top_idx][None], len(cohort), axis=0), None, findings)
        result['posthoc_constant_most_frequent_report'] = {k: fixed[k] for k in ['chexbert_f1', 'transition_f1']}
        result['posthoc_constant_most_frequent_report']['note'] = 'Test-output-selected diagnostic, not a predeclared or trained baseline.'
    summary['models'][name] = result
    data[name] = dict(current=current, target=target, predicted=predicted, scores=scores, metrics=metric, counts=counts)

valid_target = np.isin(target, [0, 1])
known = np.isin(current, [0, 1]) & valid_target
changed = (known & (current != target)).any(1)
summary['cohort'] = dict(changed=int(changed.sum()), stable_observed=int((~changed & known.any(1)).sum()),
                         no_joint_known=int((~known.any(1)).sum()), known_future_fields=int(valid_target.sum()),
                         known_transition_fields=int(known.sum()), change_events=int((known & (current != target)).sum()),
                         prevalence={d: float((target[valid_target[:, j], j] == 1).mean()) for j, d in enumerate(findings)})
summary['constant_score_macro_ap'] = float(np.mean(list(summary['cohort']['prevalence'].values())))

# Paired patient-cluster bootstrap: preserve all pairs of each sampled patient.
# AP/F1 use the same reference-supported denominator for all methods in each replicate.
rng = np.random.default_rng(20260909)
B = 2000
bootstrap = {name: {k: [] for k in ['finding_auprc', 'chexbert_f1', 'transition_f1']} for name in names}
for b in range(B):
    weights = np.bincount(rng.integers(len(patients), size=len(patients)), minlength=len(patients))[patient_idx]
    for name in names:
        pred, scores = data[name]['predicted'], data[name]['scores']
        aps, fs, events = [], [], []
        for j in range(len(findings)):
            mask = valid_target[:, j]
            y, p, w = target[mask, j] == 1, pred[mask, j] == 1, weights[mask]
            if w[y].sum():
                tp, fp, fn = w[y & p].sum(), w[~y & p].sum(), w[y & ~p].sum()
                fs.append(2*tp/(2*tp+fp+fn))
                if scores is not None and w[~y].sum():
                    aps.append(average_precision_score(y, scores[mask, j], sample_weight=w))
            for a, z in [(0, 1), (1, 0)]:
                truth_event = known[:, j] & (current[:, j] == a) & (target[:, j] == z)
                pred_event = known[:, j] & (current[:, j] == a) & (pred[:, j] == z)
                if weights[truth_event].sum():
                    tp = weights[truth_event & pred_event].sum()
                    fp = weights[~truth_event & pred_event].sum()
                    fn = weights[truth_event & ~pred_event].sum()
                    events.append(2*tp/(2*tp+fp+fn))
        for key, values in [('finding_auprc', aps), ('chexbert_f1', fs), ('transition_f1', events)]:
            bootstrap[name][key].append(float(np.mean(values)) if values else np.nan)
summary['bootstrap'] = dict(replicates=B, seed=20260909, unit='patient', interval='paired percentile 95%',
    note='Reference support is recomputed within each replicate, shared across models. Rare-event Transition F1 intervals are unstable; these intervals exclude training-seed variability.', differences={})
for other in ['direct', 'matched', 'copy']:
    intervals = {}
    for key in ['finding_auprc', 'chexbert_f1', 'transition_f1']:
        if other == 'copy' and key == 'finding_auprc':
            continue
        diff = np.asarray(bootstrap['ours'][key]) - np.asarray(bootstrap[other][key])
        intervals[key] = dict(difference=summary['models']['ours'][key]-summary['models'][other][key],
                              low=float(np.nanpercentile(diff, 2.5)), high=float(np.nanpercentile(diff, 97.5)))
    summary['bootstrap']['differences']['ours_minus_'+other] = intervals

# Inspect matched validation trajectories without comparing unlike total losses.
validation = {}
for name in ['ours', 'matched', 'direct']:
    history = rows(RUN / name / 'metrics.jsonl')
    by_step = {r['step']: r for r in history}
    vals = []
    for line in (RUN / name / 'console.log').read_text().splitlines():
        if line.startswith('validation step='):
            step, value = line[len('validation step='):].split(': ', 1)
            record = by_step[int(step)]
            if record['stage'] == 2:
                vals.append(dict(step=int(step), stage_step=record['stage_step'], hours=record['train_hours'], **ast.literal_eval(value)))
    final = read(RUN / name / 'validation_final.json')
    status = read(RUN / name / 'status.json')
    vals.append(dict(step=status['step'], stage_step=status['stage_step'], hours=status['train_hours'], **final['metrics']))
    validation[name] = vals
    summary['models'][name]['training'] = dict(status=status, first_stage2_validation=vals[0], final_validation=vals[-1],
        first100_stage2_training_mean={k: float(np.mean([r[k] for r in history if r['stage'] == 2][:100])) for k in ['loss', 'text', 'finding']},
        last100_training_mean={k: float(np.mean([r[k] for r in history[-100:]])) for k in ['loss', 'text', 'finding']},
        finite_logged_values=all(np.isfinite(r[k]) for r in history for k in ['loss', 'grad_norm']))

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
for ax, key, title in zip(axs, ['text', 'finding', 'latent'], ['Validation report CE', 'Validation finding BCE', 'Validation latent MSE']):
    for name, vals in validation.items():
        if key not in vals[0]:
            continue
        ax.plot([v['stage_step'] for v in vals], [v[key] for v in vals], '-o', ms=3, color=colors[name], label=display[name])
    ax.set(title=title, xlabel='Stage-2 optimizer updates', ylabel='Loss')
    ax.grid(alpha=.2)
axs[0].legend(fontsize=9)
fig.suptitle('Pilot validation: fixed 64 pairs; Stage-2 curves only', fontsize=13)
fig.savefig(OUT / 'training.png', dpi=180)
fig.savefig(OUT / 'training.pdf')
plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
ax = axs[0]
for i, name in enumerate(names):
    values = sorted(data[name]['counts'].values(), reverse=True)
    ax.bar(i, values[0]/10, color=colors[name])
    ax.text(i, values[0]/10+1.5, f'{values[0]/10:.1f}%\n{len(values)} unique', ha='center', fontsize=9)
ax.set(xticks=range(4), xticklabels=[display[n].replace(' ', '\n') for n in names], ylabel='Share of test reports (%)', title='Most frequent exact report', ylim=(0, 80))
ax = axs[1]
x = np.arange(6)
for i, name in enumerate(names):
    ax.bar(x+(i-1.5)*.2, [data[name]['metrics']['per_finding'][d]['f1'] for d in findings], .2, color=colors[name], label=display[name])
ax.set(xticks=x, xticklabels=['Atelect.', 'Cardiom.', 'Consol.', 'Edema', 'Effusion', 'Pneumo.'], title='Report-derived finding F1', ylim=(0, 1), ylabel='F1')
ax.tick_params(axis='x', rotation=35)
ax.legend(fontsize=8)
ax = axs[2]
for i, name in enumerate(names):
    ax.bar(i-.17, summary['models'][name]['transition_f1'], .32, color=colors[name])
    ax.bar(i+.17, summary['models'][name]['transition_without_atelectasis_onset'], .32, color=colors[name], alpha=.45, hatch='//')
ax.set(xticks=range(4), xticklabels=[display[n].replace(' ', '\n') for n in names], ylabel='Macro event F1', title='Sensitivity to one supported case', ylim=(0, .34))
ax.text(.02, .98, 'Solid: official 11 events\nHatched: omit atelectasis onset (n=1)\nPost hoc diagnostic; official table unchanged', transform=ax.transAxes, va='top', fontsize=8)
fig.suptitle('1,000 test pairs / 195 patients: output collapse and metric sensitivity', fontsize=13)
fig.savefig(OUT / 'diagnostics.png', dpi=180)
fig.savefig(OUT / 'diagnostics.pdf')
plt.close(fig)

(OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
print(json.dumps({k: v for k, v in summary.items() if k != 'models'}, indent=2))
for name, result in summary['models'].items():
    print(name, json.dumps({k: v for k, v in result.items() if k not in ['training', 'prediction_label_counts']}))
print('Artifacts:', OUT)
