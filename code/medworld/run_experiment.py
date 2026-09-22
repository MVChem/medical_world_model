"""Durable training -> full held-out evaluation -> native-Qwen comparison pipeline."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import shutil
from copy import deepcopy

from .config import PROJECT, load_config
from .launch_distributed import gpu_status, atomic


def registry(run, state, outcome=None, summary="Classification, segmentation and VQA training/evaluation."):
    root=PROJECT/'experiments';root.mkdir(exist_ok=True)
    with (root/'.registry.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=root/'registry.json'
        entries=json.loads(path.read_text()) if path.exists() else []
        identity=(run.parent.name + "_" + run.name) if run.name in ("baseline", "slots") else run.name;relative=str(run.relative_to(PROJECT))
        entries=[e for e in entries if e['id']!=identity]
        if outcome is None:
            entries.append({'id':identity,'summary':summary,'status':state,
                            'observed_at':datetime.now().astimezone().isoformat(),'run':relative,'live_status':relative+'/pipeline_status.json'})
        else:
            history=root/'README.md'
            row=f"| {datetime.now().astimezone().date()} | `{identity}` | {outcome} | [Run](../{relative}/) |\n"
            lines = [line for line in history.read_text().splitlines() if f'| `{identity}` |' not in line]
            history.write_text('\n'.join(lines).rstrip()+'\n'+row)
        atomic(path,entries)


def schedule(jobs, gpus, run, env):
    waiting=list(jobs);active={};finished=set();last_write=0
    try:
        while waiting or active:
            for gpu,(job,process,handle) in list(active.items()):
                code=process.poll()
                if code is not None:
                    handle.close();del active[gpu]
                    if code:raise RuntimeError(f"Evaluation {job['id']} exited {code}; see {job['log']}")
                    finished.add(job['id'])
            for gpu in gpus:
                if gpu in active:continue
                ready=next((j for j in waiting if set(j.get('after',[]))<=finished),None)
                if ready is None:continue
                status=gpu_status(gpu)
                if int(status['memory.used'])>1024 or int(status['utilization.gpu'])>5:continue
                log=run/ready['log'];log.parent.mkdir(parents=True,exist_ok=True);handle=log.open('a')
                command=[sys.executable,'-m',ready['module'],*ready['args'],'--gpu',gpu]
                process=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=handle,stderr=subprocess.STDOUT)
                active[gpu]=(ready,process,handle);waiting.remove(ready)
            if time.time()-last_write>15:
                atomic(run/'pipeline_status.json',{'phase':'evaluating','finished_jobs':sorted(finished),
                       'active_jobs':{gpu:{'id':j['id'],'pid':p.pid} for gpu,(j,p,h) in active.items()},
                       'pending_jobs':[j['id'] for j in waiting],'heartbeat_unix':time.time()})
                last_write=time.time()
            time.sleep(3)
    finally:
        for job,process,handle in active.values():
            if process.poll() is None:process.terminate()
        for job,process,handle in active.values():
            try:process.wait(timeout=60)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            handle.close()


def evaluation_jobs(run, cfg=None):
    cfg = cfg or load_config(run / 'config.json' if (run / 'config.json').exists() else None)
    testing = cfg['testing']
    if not testing['enabled']:
        return []
    pairs = [(task, 'test') for task in testing['tasks']]
    if 'segmentation' in testing['tasks'] and testing['human_segmentation']:
        pairs.append(('segmentation', 'human_test'))
    jobs = []
    for task, split in pairs:
        name = task if split == 'test' else 'segmentation_human'
        jobs.append({'id': name, 'module': 'medworld.evaluation.evaluate', 'log': f'evaluation/{name}.log',
                     'args': ['--checkpoint', str(run / 'final.pt'), '--out', str(run / 'evaluation' / name),
                              '--task', task, '--split', split,
                              '--vqa-per-type', str(testing['vqa_per_type']), '--vqa-seed', str(testing['vqa_seed'])]})
    return jobs


def arm_config(cfg, use_slots, steps=None):
    result = deepcopy(cfg)
    result['slot_conditioning'] = use_slots
    if steps is not None:
        result.update(total_hours=0, steps=steps)
    return result


def plan_updates(run, total_hours, calibration_seconds):
    """Estimate a common update budget; never stop one formal arm on time."""
    import math
    rates = {}
    for name in ('slots', 'baseline'):
        records = [json.loads(line) for line in (run / f'calibration_{name}/metrics.jsonl').read_text().splitlines()]
        warm = records[3:]
        if len(warm) < 6 or set(r['task'] for r in warm) != {'classification', 'segmentation', 'vqa'}:
            raise ValueError('Calibration must cover all three tasks after warmup')
        elapsed = (records[-1]['wall_unix'] - records[2]['wall_unix']) / len(warm)
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError('Invalid calibration timing')
        rates[name] = elapsed
    remaining = total_hours * 3600 - calibration_seconds
    # Allow 10% for startup, checkpoints and validation; retain complete task cycles.
    steps = int(remaining * .9 / sum(rates.values()) / 3) * 3
    if steps < 3:
        raise ValueError('Training budget is too short after calibration; use explicit steps instead')
    return {'mode': 'calibrated_equal_steps', 'requested_total_hours': total_hours,
            'calibration_seconds': calibration_seconds, 'seconds_per_step': rates,
            'target_steps_per_arm': steps,
            'estimated_training_seconds': {name: seconds * steps for name, seconds in rates.items()},
            'headroom_fraction': .1, 'evaluation_time_included': False}


def native_tasks(cfg):
    return [task for task in cfg['testing']['tasks'] if task in ('classification', 'vqa')]


def native_spec(cfg):
    from medworld_zero_shot_eval.models import models
    matches = [s for s in models() if s['family'] == 'qwen'
               and Path(s['path']).resolve() == Path(cfg['qwen']).resolve()]
    if len(matches) != 1 or matches[0]['id'] not in ('qwen08b', 'qwen4b', 'qwen9b'):
        raise ValueError('Native Qwen must match a supported pretrained backbone path')
    return matches[0]


def freeze_experiment(run, with_qwen):
    from .launch_distributed import snapshot
    from .datasets.protocol import _sha256
    snapshot(run)
    if with_qwen:
        source = Path(__file__).resolve().parent.parent / 'medworld_zero_shot_eval'
        target = run / 'source/medworld_zero_shot_eval'
        shutil.copytree(source, target, ignore=shutil.ignore_patterns('runs', '__pycache__', '.git'))
        atomic(run / 'native_source_manifest.json', {
            str(p.relative_to(target)): _sha256(p) for p in sorted(target.rglob('*')) if p.is_file()})


def write_report(run, cfg, budgets, no_slots_rows, qwen_rows, skipped):
    summaries = json.loads((run / 'slots/evaluation_summary.json').read_text()) if cfg['testing']['enabled'] else {}
    no_slots = {(r['task'], r['metric']): r['baseline'] for r in no_slots_rows}
    qwen = {(r['task'], r['metric']): r['qwen'] for r in qwen_rows}
    rows = []
    for folder, summary in summaries.items():
        task = 'segmentation' if folder == 'segmentation_human' else folder
        keys = {'classification': ('macro_auroc', 'macro_ap'), 'segmentation': ('mean_dice', 'mean_iou'),
                'vqa': ('exact_match', 'micro_f1')}[task]
        metrics = summary['tasks'][task]
        for metric in keys:
            row = {'task': folder, 'metric': metric, 'n': metrics['n'], 'slots': metrics[metric]}
            if (folder, metric) in no_slots:
                row['no_slots'] = no_slots[(folder, metric)]
            if (folder, metric) in qwen:
                row['qwen'] = qwen[(folder, metric)]
            rows.append(row)
        if task == 'segmentation':
            for dataset, entry in metrics.get('by_dataset', {}).items():
                label = f'{folder}/{dataset}'
                for metric in keys:
                    row = {'task': label, 'metric': metric, 'n': entry['n_volumes'],
                           'unit': 'MRI volumes or CXR images', 'slots': entry[metric]}
                    if (label, metric) in no_slots:
                        row['no_slots'] = no_slots[(label, metric)]
                    rows.append(row)
    atomic(run / 'comparison.json', {'training': budgets, 'baselines': cfg['baselines'],
                                    'skipped': skipped, 'qwen_training_steps': 0, 'rows': rows})
    columns = ['slots'] + (['no_slots'] if no_slots_rows else []) + (['qwen'] if qwen_rows else [])
    names = {'slots': 'With 8 slots', 'no_slots': 'No slots',
             'qwen': 'Native ' + native_spec(cfg)['label'] if qwen_rows else 'Native Qwen'}
    lines = ['# MedWorld experiment', '',
             'Both trained arms must finish exactly the same optimizer updates with matching batches. Time is an estimate; native Qwen is not trained.', '',
             '| Model | Target updates | Actual training (seconds) | Optimizer updates |',
             '|---|---:|---:|---:|']
    for name, budget in budgets.items():
        elapsed = budget['elapsed_seconds']
        lines.append(f"| {name} | {budget['target_steps']} | {elapsed:.1f} | {budget['steps']} |")
    lines += ['',
             '| Task | Metric | N | ' + ' | '.join(names[c] for c in columns) + ' |',
             '|---|---|---:|' + '---:|' * len(columns)]
    for row in rows:
        values = ['N/A' if row.get(c) is None else f'{row[c]:.5f}' for c in columns]
        lines.append('| ' + ' | '.join([row['task'], row['metric'], str(row['n']), *values]) + ' |')
    if skipped:
        lines += ['', 'Skipped: ' + '; '.join(f'{k}: {v}' for k, v in skipped.items()) + '.']
    (run / 'COMPARISON.md').write_text('\n'.join(lines) + '\n')


def run_experiment(cfg, run, gpus):
    run = Path(run).resolve()
    selectors = gpus.split(',')
    if len(selectors) < 2 or not all(selectors) or len(set(selectors)) != len(selectors):
        raise ValueError('Select at least two distinct GPUs')
    if run.exists() and any(run.iterdir()):
        raise ValueError('Choose a new experiment directory')
    run.mkdir(parents=True, exist_ok=True)
    cfg = deepcopy(cfg)
    cfg['slot_conditioning'] = True
    qwen_tasks = native_tasks(cfg)
    qwen_enabled = cfg['baselines']['qwen'] and cfg['testing']['enabled'] and bool(qwen_tasks)
    spec = native_spec(cfg) if qwen_enabled else None
    # Freeze all arms before training so later workspace edits cannot alter the comparison.
    freeze_experiment(run, qwen_enabled)
    atomic(run / 'config.json', cfg)
    registry(run, 'training', summary='Slots first; optional equal-step no-slots training and native Qwen tests.')
    env = dict(os.environ, PYTHONPATH=str(run / 'source'), MEDWORLD_PROJECT_ROOT=str(PROJECT),
               PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', PYTORCH_ALLOC_CONF='expandable_segments:True')
    child = None
    outcome = 'Failed; see pipeline_status.json and original run logs.'
    skipped = {}
    previous_handlers = {}

    def interrupted(*_):
        raise KeyboardInterrupt

    def train_arm(name, settings):
        nonlocal child
        config = run / f'{name}_config.json'
        atomic(config, settings)
        atomic(run / 'pipeline_status.json', {'phase': name, 'heartbeat_unix': time.time(),
               'active_run': name, 'active_status': f'{name}/status.json'})
        with (run / f'{name}.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-m', 'medworld.launch_distributed', '--config', str(config),
                                      '--out', str(run / name), '--gpus', gpus], cwd=PROJECT, env=env,
                                     stdout=log, stderr=subprocess.STDOUT)
            if child.wait():
                raise RuntimeError(f'{name} training/evaluation failed; see {name}.log')
        state = json.loads((run / name / 'status.json').read_text())
        if (not state.get('complete') or state.get('stopped') or not (run / name / 'final.pt').is_file()
                or type(state.get('step')) is not int or state['step'] <= 0):
            raise ValueError(f'{name} training did not complete successfully')
        if not settings['total_hours'] and state['step'] != settings['steps']:
            raise ValueError(f'{name} optimizer update count differs from the target')
        if settings['testing']['enabled']:
            pipeline = json.loads((run / name / 'pipeline_status.json').read_text())
            if not pipeline.get('evaluation_complete'):
                raise ValueError(f'{name} tests did not complete')
        return {'steps': state['step'], 'requested_hours': settings['total_hours'],
                'target_steps': settings['steps'] if not settings['total_hours'] else None,
                'elapsed_seconds': state['heartbeat_unix'] - state['started_unix']}

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.signal(sig, interrupted)
        target_steps = cfg['steps'] if not cfg['total_hours'] else None
        if cfg['total_hours'] and cfg['baselines']['no_slots']:
            calibration_start = time.monotonic()
            for name, use_slots in (('slots', True), ('baseline', False)):
                probe = arm_config(cfg, use_slots, 9)
                probe['testing']['enabled'] = False
                probe.update(validate_every=10, validation_samples=1, save_every=10)
                train_arm(f'calibration_{name}', probe)
            plan = plan_updates(run, cfg['total_hours'], time.monotonic() - calibration_start)
            target_steps = plan['target_steps_per_arm']
        else:
            plan = {'mode': 'equal_steps' if target_steps is not None else 'single_arm_time',
                    'requested_total_hours': cfg['total_hours'], 'target_steps_per_arm': target_steps}
        atomic(run / 'budget_plan.json', plan)
        # Formal arms start from the same original initialization, never probe weights.
        budgets = {'slots': train_arm('slots', arm_config(cfg, True, target_steps))}
        atomic(run / 'training_budget.json', {'total_hours': cfg['total_hours'], 'arms': budgets})
        no_slots_rows, qwen_rows = [], []
        if cfg['baselines']['no_slots']:
            budgets['no_slots'] = train_arm('baseline', arm_config(cfg, False, target_steps))
            if budgets['no_slots']['steps'] != budgets['slots']['steps']:
                raise ValueError('No-slots and slots optimizer update counts must match exactly')
            atomic(run / 'training_budget.json', {'total_hours': cfg['total_hours'], 'arms': budgets})
            if cfg['testing']['enabled']:
                from .evaluation.compare_run import compare
                no_slots_rows = compare(run / 'baseline', run / 'slots', run / 'no_slots_comparison')
        else:
            skipped['no_slots'] = 'disabled by baselines.no_slots'
        if qwen_enabled:
            jobs = [{'id': 'qwen', 'module': 'medworld_zero_shot_eval.evaluate', 'log': 'qwen.log',
                     'args': ['--config', str(run / 'slots/config.json'), '--model', spec['id'],
                              '--tasks', *qwen_tasks, '--out', str(run / 'qwen')]}]
            registry(run, 'evaluating', summary='Native Qwen on matching selected tests after trained arms.')
            schedule(jobs, selectors, run, env)
            from .evaluation.compare_native import compare_native
            qwen_rows = compare_native(run / 'slots', run / 'qwen', run, qwen_tasks)
            if 'segmentation' in cfg['testing']['tasks']:
                skipped['qwen_segmentation'] = 'native Qwen has no segmentation interface'
        else:
            skipped['qwen'] = ('disabled by baselines.qwen' if not cfg['baselines']['qwen'] else
                               'testing.enabled=false' if not cfg['testing']['enabled'] else
                               'no supported classification/VQA task selected')
        write_report(run, cfg, budgets, no_slots_rows, qwen_rows, skipped)
        atomic(run / 'pipeline_status.json', {'phase': 'complete', 'comparison': 'COMPARISON.md',
               'training': budgets, 'baselines': cfg['baselines'], 'skipped': skipped,
               'testing_enabled': cfg['testing']['enabled'], 'heartbeat_unix': time.time()})
        outcome = 'Completed: slots' + (' and no-slots' if cfg['baselines']['no_slots'] else '')
        outcome += (' training; selected tests' if cfg['testing']['enabled'] else ' training; testing disabled')
        outcome += (' and native Qwen comparison.' if qwen_enabled else '.')
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=360)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        atomic(run / 'pipeline_status.json', {'phase': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
               'error': repr(error), 'heartbeat_unix': time.time()})
        raise
    finally:
        registry(run, 'finished', outcome)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main():
    p = argparse.ArgumentParser(description='Train/test slots first, then optional no-slots and native Qwen baselines.')
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--gpus', default='1,2')
    a = p.parse_args()
    run_experiment(load_config(a.config), a.out, a.gpus)


if __name__ == '__main__':
    main()
