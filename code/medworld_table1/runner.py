"""Persistent local experiment coordinator; launch inside tmux via launch.sh."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from common import ROOT, atomic_json, read_config


def launch(args):
    os.umask(0o077)
    root = Path(args.run).resolve()
    root.mkdir(parents=True, exist_ok=False)
    cfg = read_config(args.config)
    atomic_json(root / 'config.json', cfg)
    source = root / 'source'
    source.mkdir()
    for file in ROOT.glob('*.py'):
        shutil.copy2(file, source / file.name)
    shutil.copy2(ROOT / 'README.md', root / 'protocol.md')
    jobs, evaluations = {}, {}

    def start(name, gpu, command):
        directory = root / name
        directory.mkdir(exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM='false',
                   OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1')
        with (directory / 'console.log').open('a') as log:
            process = subprocess.Popen([sys.executable, *command], cwd=ROOT, env=env,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        print(f'{name} pid={process.pid} gpu={gpu}', flush=True)
        return process

    config_args = ['--config', str(root / 'config.json')]
    jobs['ours'] = start('ours', 0, ['train.py', *config_args, '--run', str(root / 'ours')])
    jobs['direct'] = start('direct', 1, ['train.py', *config_args, '--mode', 'direct', '--run', str(root / 'direct')])
    # Copy Current is scored independently; no training and no disease scores.
    while True:
        if 'copy' not in evaluations and (ROOT / 'weights/clinical_ready.json').exists():
            evaluations['copy'] = start('copy', 5, ['evaluate.py', *config_args, '--mode', 'copy', '--out', str(root / 'copy')])
        if 'matched' not in jobs and (root / 'ours/checkpoint_stage1.pt').exists():
            jobs['matched'] = start('matched', 2, ['train.py', *config_args, '--run', str(root / 'matched'), '--follow', str(root / 'ours')])
        for name, process in jobs.items():
            if process.poll() == 0 and name not in evaluations:
                evaluations[name] = start(name + '/evaluation', {'ours': 0, 'direct': 1, 'matched': 2}[name],
                    ['evaluate.py', *config_args, '--mode', name, '--checkpoint', str(root / name / 'checkpoint_final.pt'),
                     '--out', str(root / name / 'evaluation')])
        statuses = {name: dict(pid=p.pid, exit_code=p.poll()) for name, p in jobs.items()}
        scores = {name: dict(pid=p.pid, exit_code=p.poll()) for name, p in evaluations.items()}
        failed = any(v['exit_code'] not in (None, 0) for v in [*statuses.values(), *scores.values()])
        complete = all(p.poll() is not None for p in jobs.values()) and all(p.poll() is not None for p in evaluations.values())
        if complete and 'copy' not in evaluations:
            # Training completion is still surfaced; absent scoring prerequisites
            # are not an infinite silent wait.
            failed = True
        atomic_json(root / 'runner_status.json', dict(pid=os.getpid(), state='failed' if failed else ('complete' if complete else 'running'),
                    training=statuses, evaluations=scores, updated_unix=time.time(),
                    clinical_assets='ready' if 'copy' in evaluations else 'preparing'))
        render_table(root)
        if complete:
            print('All launched jobs exited; see runner_status.json.', flush=True)
            return 1 if failed else 0
        time.sleep(10)


def render_table(root):
    lines = ['# Table 1 small-model pilot (not final paper results)', '',
        '| Method | Future R@1 | Finding AUPRC | Transition F1 | RadGraph F1 | CheXbert F1 | Status |',
        '|---|---:|---:|---:|---:|---:|---|']
    for key, label in [('copy', 'Copy Current'), ('direct', 'Qwen3.5-0.8B direct'),
                        ('matched', 'Stage-1 state + matched LWM'), ('ours', 'MedWorld-JEPA (Qwen3.5-0.8B)')]:
        path = root / key / ('metrics.json' if key == 'copy' else 'evaluation/metrics.json')
        if path.exists():
            scores = json.loads(path.read_text())
            values = ['—' if scores.get(k) is None else f'{scores[k]:.4f}' for k in
                      ('future_r1', 'finding_auprc', 'transition_f1', 'radgraph_f1', 'chexbert_f1')]
            status = f'n={scores["n"]}; retrieval n={scores["retrieval_queries"]}; {scores["radgraph_status"]}'
        else:
            values, status = ['pending']*5, 'pending'
            status_path = root / key / 'status.json'
            if status_path.exists():
                s = json.loads(status_path.read_text())
                status = f'{s["state"]}; stage {s["stage"]}; step {s["stage_step"]}; {s["train_hours"]:.2f}h'
        lines.append('| ' + ' | '.join([label, *values, status]) + ' |')
    lines += ['', 'Macro AP and F1 use reference-supported findings/events; blank/uncertain references are excluded.',
              'Retrieval uses 32 candidates with fractional ties and reports its inclusion coverage. Unavailable scores remain —.',
              'Stage 1: image-only disease and report tasks. No segmentation/SR supervision in this pilot.',
              'Report availability timestamps are unknown. This is retrospective report-available forecasting.',
              'Direct baseline: 8h native Qwen vision + report SFT. Ours: 1h Stage 1 + 7h Stage 2.',
              'Matched: shared Stage-1 initialization, identical Stage-2 batches/updates/losses/LR; frozen source encoder. Its compute time can differ.']
    (root / 'table1.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/pilot.json')
    parser.add_argument('--run', required=True)
    sys.exit(launch(parser.parse_args()))
