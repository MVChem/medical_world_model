"""Freeze source/configs and emit a central-scheduler manifest, without launch."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from common import ROOT, atomic_json, digest, load_rows, read_config


def main(args):
    os.umask(0o077)
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    source = run / 'source'
    if source.exists():
        raise FileExistsError('Source snapshot exists; use a new run or audit explicitly')
    own = source / 'code/medworld_table1'
    own.mkdir(parents=True)
    for path in ROOT.glob('*.py'):
        shutil.copy2(path, own / path.name)
    shutil.copytree(ROOT / 'tests', own / 'tests', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT.parent / 'medworld_common', source / 'code/medworld_common', ignore=shutil.ignore_patterns('__pycache__'))
    baseline = source / 'code/medworld_baselines'
    baseline.mkdir()
    for name in ('base.py', 'stats.py', 'green_eval.py'):
        shutil.copy2(ROOT.parent / 'medworld_baselines' / name, baseline / name)
    shutil.copytree(ROOT.parent / 'medworld_baselines/vendor/GREEN/green_score', baseline / 'green_official', ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('weights', 'vendor', 'metric_vendor'):
        (own / name).symlink_to(ROOT / name, target_is_directory=True)
    (source / 'code/vjepa2').symlink_to(ROOT.parent / 'vjepa2', target_is_directory=True)
    atomic_json(run / 'source_manifest.json', {str(p.relative_to(source)): digest(p)
                for p in source.rglob('*.py') if not any(parent.is_symlink() for parent in p.parents)})
    configs = {}
    for condition in ('slots', 'no_slots', 'shuffled'):
        cfg = read_config(ROOT / f'configs/overnight_20260913_{condition}.json')
        configs[condition] = run / f'configs/{condition}.json'
        atomic_json(configs[condition], cfg)
    cache = Path(cfg['cache'])
    old = Path(cfg['feature_reuse_cache'])
    if not (cache / 'manifest.json').exists():
        raise RuntimeError('CPU cohort preparation must finish before queue publication')
    audit = {}
    for split in ('train', 'validate', 'test'):
        previous = load_rows(old / f'{split}.jsonl')
        expanded = load_rows(cache / f'{split}.jsonl')
        if split == 'train':
            assert {r['id'] for r in previous} <= {r['id'] for r in expanded}
        else:
            assert previous == expanded, 'Official held-out cohort changed'
        audit[split] = dict(previous_pairs=len(previous), expanded_pairs=len(expanded), sha256=digest(cache / f'{split}.jsonl'))
    atomic_json(run / 'cohort_audit.json', audit)
    python = str(ROOT / '.venv/bin/python')
    def job(name, script, arguments, artifacts, deps=(), minimum=60, success=None):
        return dict(id='table1_'+name, argv=[python, str(own/script), *map(str, arguments)],
            artifacts=list(map(str, artifacts)), deps=list(deps), gpus=1, allowed_gpus=[3], priority=20,
            minimum_seconds=minimum, max_attempts=1, success_fields=success or {})
    jobs = [job('features', 'features.py', ['--config', configs['slots']], [cache/'features.json'], minimum=1800)]
    prior = 'table1_features'
    for condition in ('no_slots', 'slots', 'shuffled'):
        train = run / condition
        config = configs[condition]
        status = str(train/'status.json')
        jobs.append(job(condition+'_train', 'train.py', ['--config', config, '--run', train],
            [train/'checkpoint_final.pt', train/'status.json', train/'task_gradient_audit.json'], [prior], minimum=4*3600,
            success={status:dict(state='complete', stage=2, stage_step=cfg['max_stage2_steps'])}))
        prior = 'table1_'+condition+'_train'
        output = train/'evaluation_test'
        jobs.append(job(condition+'_eval', 'evaluate.py', ['--config', config, '--mode', 'ours', '--checkpoint', train/'checkpoint_final.pt', '--out', output],
            [output/'metrics.json', output/'generation.json'], [prior], minimum=1500,
            success={str(output/'generation.json'):dict(count=297, teacher_forcing=False, target_inputs=False)}))
        prior = 'table1_'+condition+'_eval'
    jobs.append(job('green', 'overnight_green.py', ['--run', run],
        [run/'green'/condition/'test/green_metrics.json' for condition in configs], [prior], minimum=3600,
        success={str(run/'green'/condition/'test/green_metrics.json'):dict(status='complete', n=297, completed=297) for condition in configs}))
    atomic_json(run/'jobs.json', dict(jobs=jobs, report_commands=[], protocol=dict(
        train_pairs=16000, test_pairs=297, patient_cap=4, matched_stage2_steps=cfg['max_stage2_steps'],
        state_conditions=list(configs), initial_checkpoint=cfg['initialize_stage1'],
        no_slots='All valid image/report/EHR hidden tokens; masked predictor and readouts; eight-bin pooling only for latent loss',
        shuffled='Deterministic source-state donor from another patient in the same split; own target/horizon unchanged',
        deadline=cfg['train_deadline'])))
    print(run/'jobs.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args())
