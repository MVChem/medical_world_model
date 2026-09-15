"""Fill the shared paper tables from completed, protocol-matched local results.

Run from any directory. This writes aggregate values and source hashes only;
patient records and predictions stay in the original experiment directories.
"""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / '27cvpr'
OPEN = 'code/medworld_open_baselines/runs/comparators_20260913'
RAW = 'code/medworld_baselines/runs/raw_models_20260911'
FORECAST = 'code/medworld_table1/runs/qwen9b_ablation_20260914'
FROZEN = 'code/medworld_dense_baselines/runs/frozen_slots_20260913'
SHUFFLED = 'code/medworld_dense_baselines/runs/frozen_slots_shuffled_remaining_20260913'
NA, TBD = r'\na', r'\tbd'
sources = {}


def read(path):
    data = (ROOT / path).read_bytes()
    sources[path] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def green(path):
    data = read(path)
    assert data['status'] == 'complete'
    assert data['n'] == data['completed'] == 297
    return data['mean']


def row(label, values, origin, protocol):
    return dict(label=label, values=values, sources=origin, protocol=protocol)


def forecast(label, path, green_path, protocol):
    data = read(path)
    metrics = data.get('table1', data)
    assert metrics.get('n', metrics.get('report_n')) == 297
    values = [metrics[k] for k in ('ap', 'auroc', 'transition_f1')]
    values += [TBD, metrics['radgraph_f1'], green(green_path), metrics['brier'], metrics['ece']]
    return row(label, values, [path, green_path], protocol)


def qwen9b_forecast(label, condition):
    """Only publish completed 9B runs; never reuse the historical 0.8B values."""
    parent = f'{FORECAST}/{condition}'
    metadata_paths = [f'{parent}/config.json', f'{parent}/status.json',
                      f'{parent}/evaluation_test/generation.json']
    metrics_path = f'{parent}/evaluation_test/metrics.json'
    green_path = f'{FORECAST}/green/{condition}/test/green_metrics.json'
    protocol = ('Qwen3.5-9B forecast adaptation; final-language-layer state; '
                'new shared 9B Stage-1 initialization; 16000 training pairs; '
                '2400 updates x effective batch32; evaluation pending')
    pending = row(label, [TBD] * 8, [], protocol)
    if not all((ROOT / p).is_file() for p in metadata_paths + [metrics_path]):
        return pending
    cfg, status, generation = [read(p) for p in metadata_paths]
    if (status.get('state') != 'complete' or status.get('stage') != 2 or
            status.get('stage_step') != 2400 or generation.get('count') != 297):
        pending['sources'] = metadata_paths
        return pending
    assert any(part in ('Qwen3.5-9B', 'models--Qwen--Qwen3.5-9B')
               for part in Path(cfg['qwen']).parts), 'Wrong forecast backbone'
    assert cfg['state_condition'] == generation['state_condition'] == condition
    assert cfg['max_train_pairs'] == 16000 and cfg['max_stage2_steps'] == 2400
    assert cfg['batch_size'] * cfg.get('gradient_accumulation', 1) == 32
    assert generation['split'] == 'test'
    assert generation['teacher_forcing'] is False and generation['target_inputs'] is False
    expected_checkpoint = (ROOT / parent / 'checkpoint_final.pt').resolve()
    assert expected_checkpoint.is_file()
    assert Path(generation['checkpoint']).resolve() == expected_checkpoint
    assert len(generation['checkpoint_sha256']) == 64
    data = read(metrics_path)
    metrics = data.get('table1', data)
    assert metrics.get('n', metrics.get('report_n')) == 297
    assert metrics.get('patients', data.get('patients')) == 94
    assert data.get('state_condition', metrics.get('state_condition')) == condition
    values = [metrics[k] for k in ('ap', 'auroc', 'transition_f1')]
    values += [TBD, metrics['radgraph_f1'], TBD, metrics['brier'], metrics['ece']]
    origin = metadata_paths + [metrics_path]
    if (ROOT / green_path).is_file():
        green_status = read(green_path)
        origin.append(green_path)
        if (green_status.get('status') == 'complete' and
                green_status.get('n') == green_status.get('completed') == 297):
            values[5] = green(green_path)
    return row(label, values, origin, protocol.replace('; evaluation pending', ''))


def native_downstream(label, path):
    data = read(path)
    cls, report = data['table2_classification'], data['table2_report']
    assert cls['n'] == 353 and report['report_n'] == 507
    return row(label, [cls['auroc'], cls['ap'], TBD, TBD,
                      report['radgraph_f1'], report['chexbert_f1'], TBD, TBD,
                      NA, NA, NA, NA], [path], 'native pretrained zero shot')


def dense(path, train_n=4096):
    data = read(path)
    assert data['train_n'] == train_n and data['epochs'] == 20
    assert data.get('complete_requested_epochs', True)
    assert data['metrics']['test']['n'] == 447
    return data['metrics']


def dense_row(label, segmentation, sr, protocol):
    seg, sup = dense(segmentation), dense(sr)
    assert seg['human_test']['n'] == 138
    return row(label, [NA] * 8 + [seg['test']['dice'], seg['human_test']['dice'],
                                 sup['test']['psnr'], sup['test']['ssim']],
               [segmentation, sr], protocol)


def format_value(value, *, scale=100):
    if isinstance(value, str):
        assert value in (NA, TBD)
        return value
    assert isinstance(value, (int, float)) and math.isfinite(value)
    return f'{value * scale:.2f}'


def format_rows(rows, count, midrule_before=(), *, unscaled_columns=()):
    lines = []
    for i, item in enumerate(rows):
        assert len(item['values']) == count
        if i in midrule_before:
            lines.append(r'\midrule')
        values = [format_value(value, scale=1 if j in unscaled_columns else 100)
                  for j, value in enumerate(item['values'])]
        lines += ['% ' + ('; '.join(item['sources']) or item['protocol']),
                  item['label'] + ' & ' + ' & '.join(values) + r' \\']
    return '\n'.join(lines)


t1, t2 = [], []
native = [
    (r'Qwen3.5-9B$^{\mathrm{ZS}}$', f'{RAW}/qwen9b/test'),
    (r'LLaVA-v1.6-Mistral-7B$^{\mathrm{ZS}}$', f'{OPEN}/zero_shot/llava16mistral7b/llava16mistral7b/test'),
    (r'LLaVA-Med-v1.5-7B$^{\mathrm{ZS}}$', f'{OPEN}/zero_shot/llava_med7b/llava_med7b/test'),
    (r'CheXagent-8B$^{\mathrm{ZS}}$', f'{OPEN}/zero_shot/chexagent8b/chexagent8b/test'),
]
for label, parent in native:
    t1.append(forecast(label, parent + '/metrics.json', parent + '/green_metrics.json', 'native pretrained zero shot'))
    t2.append(native_downstream(label, parent + '/metrics.json'))
for model, name, green_model in [('biovil', 'BioViL-T', 'biovil_t_adapted'), ('chexworld', 'CheXWorld', 'chexworld_adapted')]:
    parent = f'code/medworld_open_baselines/{model}_runs/forecast_16k'
    t1.append(forecast(name + ' + predictor', parent + '/evaluation_test/metrics.json',
                       parent + f'/green/{green_model}/test/green_metrics.json',
                       'frozen official encoder swap; shared V-JEPA-derived Stage-1 warm start; 16000 training pairs; 2400 updates x batch32'))
copy_path = f'{OPEN}/simple_future/copy_current/test/metrics.json'
copy_green = f'{OPEN}/simple_future/copy_current/test/green_metrics.json'
copy = read(copy_path)['table1']
t1.append(row('Copy Current', [NA, NA, copy['transition_f1'], TBD,
                              copy['radgraph_f1'], green(copy_green), NA, NA],
              [copy_path, copy_green], 'copy source report; no continuous-score interface'))
prior_path = f'{OPEN}/simple_future/finding_prior/test/metrics.json'
prior = read(prior_path)
assert prior['status'] == 'complete' and prior['n_train'] == 16000
v = prior['table1']
t1.append(row('Finding-transition prior', [v['ap'], v['auroc'], TBD, NA, NA, NA, v['brier'], v['ece']],
              [prior_path], 'train-only prior; structured source finding labels; transition readout pending'))
for condition, label in [('no_slots', 'Full-token forecaster (Qwen3.5-9B)'),
                         ('slots', r'\method{} (Qwen3.5-9B slots)'),
                         ('shuffled', 'Qwen3.5-9B (shuffled state)')]:
    t1.append(qwen9b_forecast(label, condition))

for model, name in [('dinov2_vitb14', 'DINOv2'), ('chexworld', 'CheXWorld')]:
    parent = f'{OPEN}/dense_4096/{model}'
    item = dense_row(name + ' + heads', parent + '/segmentation_spatial/metrics.json',
                     parent + '/sr_spatial/metrics.json', 'frozen visual encoder + supervised task heads; dense4096/20epochs')
    path = parent + '/classification_spatial/metrics.json'
    data = read(path)
    assert data['complete_requested_epochs'] and data['epochs'] == 20 and data['train_n'] == 13681
    test = data['metrics']['test']
    assert test['n'] == 353
    item['values'][:2] = [test['macro_auroc'], test['macro_auprc']]
    item['sources'].append(path)
    item['protocol'] += '; classification13681/20epochs; macro_auprc is non-interpolated AP'
    t2.append(item)
t2.append(row('MAIRA-2', [NA] * 4 + [TBD] * 4 + [NA] * 4, [], 'gated weights; native reporting/grounding pending'))
sr_path = f'{OPEN}/swinir_4096/metrics.json'
sr = dense(sr_path)
t2.append(row('SwinIR (CXR-adapted)', [NA] * 10 + [sr['test']['psnr'], sr['test']['ssim']], [sr_path],
              'official pretrained SwinIR-M DF2K x4; full-parameter adaptation; 4096/20epochs; 8.313 process GPU-hours'))
t2 += [
    dense_row('Image-only + heads', f'{FROZEN}/image_only/segmentation_image_only/metrics.json',
              f'{FROZEN}/image_only/sr_image_only/metrics.json', 'matched dense4096/20epochs; no VLM features'),
    dense_row(r'Qwen3.5-9B vision + slots$^{\dagger}$', f'{FROZEN}/qwen9b/segmentation_slots/metrics.json',
              f'{FROZEN}/qwen9b/sr_slots/metrics.json', 'only frozen456M vision tower; four fixed depth-pooled tokens; train task heads; dense4096/20epochs'),
    dense_row(r'Qwen3.5-9B vision + shuffled$^{\dagger}$', f'{SHUFFLED}/qwen9b/segmentation_shuffled_slots/metrics.json',
              f'{SHUFFLED}/qwen9b/sr_shuffled_slots/metrics.json', 'same frozen vision protocol; cross-patient shuffled tokens; dense4096/20epochs'),
    row('Qwen3.5-0.8B no-slots', [TBD] * 12, [], 'planned matched six-task adaptation; no eligible complete run'),
    row(r'\textbf{\method{} (full model)}', [TBD] * 12, [], 'planned complete six-task evaluation of the proposed architecture; no eligible complete run'),
]

header1 = r'''% Generated by plans/populate_results.py; see results_20260914.json for raw sources.
\begin{table*}[t]
\caption{Available future-state results on 297 held-out pairs (94 patients).
All scores are multiplied by 100 and shown to two decimals; arrows indicate the preferred direction.
$\mathrm{ZS}$ denotes public-checkpoint zero-shot inference.}
\label{tab:future_prediction}
\centering
\begingroup
\small
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{1.14}
\begin{tabular}{@{}L{0.30\linewidth}*{4}{C{0.075\linewidth}}C{0.095\linewidth}*{3}{C{0.075\linewidth}}@{}}
\toprule
 & \multicolumn{2}{c}{\shortstack{Future clinical\\status}}
 & \multicolumn{2}{c}{\shortstack{Disease\\progression}}
 & \multicolumn{2}{c}{\shortstack{Future report\\fidelity}}
 & \multicolumn{2}{c}{\shortstack{Probabilistic\\reliability}} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Method & \shortstack{AP\\$\uparrow$} & \shortstack{AUROC\\$\uparrow$}
 & \shortstack{Trans.\\F1$\uparrow$} & \shortstack{Dir.\\F1$\uparrow$}
 & \shortstack{RadGraph\\F1$\uparrow$} & \shortstack{GREEN\\$\uparrow$}
 & \shortstack{Brier\\$\downarrow$} & \shortstack{ECE\\$\downarrow$} \\
\midrule
'''
footer1 = r'''
\bottomrule
\end{tabular}
\endgroup\par
\vspace{3pt}
\begin{minipage}{\linewidth}\fontsize{8}{9.5}\selectfont
AP is non-interpolated average precision; AP/AUROC/Brier/ECE are macro-averaged
over findings with common reference masks. RadGraph uses partial F1.
VLM scores use Yes/No likelihoods. Trained forecasters use 16,000 pairs,
2,400 updates and batch 32; actual GPU hours differ. ``+ predictor'' is an
official frozen-encoder adaptation with shared forecast modules, not an original
published forecasting system. The bottom three rows are legacy JEPA-path
Qwen3.5-9B forecasters with shared Stage-1 initialization. Their slots
use eight final-language-layer queries, distinct from the proposed multi-depth
fusion/vision architecture; the full-token control retains the same predictor.
Native-image-conditioned 4+4 results, Direction labels and the prior's
transition readout are pending. The prior reads structured source findings.
TBD: pending; ---: no score for this interface. Single-seed point estimates.
\end{minipage}
\end{table*}
'''
header2 = r'''% Generated by plans/populate_results.py; see results_20260914.json for raw sources.
\begin{table*}[t]
\caption{Available results and remaining evaluations on six downstream tasks.
Higher is better. All scores are multiplied by 100 except PSNR (dB); values are shown to two decimals.
$\mathrm{ZS}$ denotes no benchmark-specific adaptation.}
\label{tab:downstream_tasks}
\centering
\begingroup
\fontsize{8}{9.5}\selectfont
\setlength{\tabcolsep}{1.5pt}
\renewcommand{\arraystretch}{1.22}
\begin{tabular}{@{}L{0.273\linewidth}*{12}{C{0.053\linewidth}}@{}}
\toprule
 & \multicolumn{2}{c}{Classification}
 & \multicolumn{2}{c}{VQA}
 & \multicolumn{2}{c}{Report gen.}
 & \multicolumn{2}{c}{Grounding}
 & \multicolumn{2}{c}{Segmentation}
 & \multicolumn{2}{c}{SR $4\times$} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}
Method & AUC & AP & Acc. & $\mu$F1 & RG & CB & mIoU & A@.5 & \shortstack{Dice\\pseudo} & \shortstack{Dice\\human} & PSNR & SSIM \\
\midrule
'''
footer2 = r'''
\bottomrule
\end{tabular}
\endgroup\par
\vspace{3pt}
\begin{minipage}{\linewidth}\fontsize{8}{9.5}\selectfont
Classification/report tests contain 353/507 images; classification heads use
13,681 training images and 20 epochs. All filled segmentation/SR rows use
4,096/249/447 train/validation/test images, 20 epochs and batch 8.
Human Dice uses 138 external Montgomery lung masks; pseudo Dice uses CXAS.
$^{\dagger}$Only the frozen native vision tower is used, with four fixed
depth-pooled tokens and trained heads; the 9B language model is not used in
these rows. Shuffled tokens come from other patients. SwinIR is fully adapted;
its training/evaluation takes 8.31 GPU-h versus 0.16 for image-only SR on the
same GPU type, so this is not an equal-GPU-hour comparison.
RG/CB: RadGraph/CheXbert F1; A@.5: box accuracy at IoU $\geq0.5$.
TBD: pending; ---: not evaluated with this interface. Official VQA, MS-CXR
grounding and the full six-task adaptation remain pending; derived QA, anatomy
localization and report-assisted prototypes are excluded.
\end{minipage}
\end{table*}
'''
(PAPER / 'tables/table1_future.tex').write_text(header1 + format_rows(t1, 8, (8,)) + footer1)
# PSNR is the eleventh metric (index 10) and retains its dB unit.
(PAPER / 'tables/table2_downstream.tex').write_text(
    header2 + format_rows(t2, 12, (8, 11), unscaled_columns=(10,)) + footer2)
manifest = dict(snapshot_date='2026-09-15', qwen_model='Qwen/Qwen3.5-9B',
                qwen_revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a',
                display=dict(decimal_places=2, default_multiplier=100,
                             unscaled_metrics=['psnr'], raw_values_preserved=True),
                table1_columns=['ap', 'auroc', 'transition_f1', 'direction_f1', 'radgraph_f1', 'green', 'brier', 'ece'],
                table2_columns=['classification_auroc', 'classification_ap', 'vqa_accuracy', 'vqa_micro_f1',
                                'report_radgraph_f1', 'report_chexbert_f1', 'grounding_miou', 'grounding_acc05',
                                'dice_pseudo', 'dice_human', 'psnr', 'ssim'],
                table1=t1, table2=t2, source_sha256=sources)
(PAPER / 'tables/results_20260914.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
for name, rows in [('Table 1', t1), ('Table 2', t2)]:
    numeric = sum(isinstance(v, (int, float)) for r in rows for v in r['values'])
    print(f'{name}: {len(rows)} rows, {numeric} populated metric cells')
