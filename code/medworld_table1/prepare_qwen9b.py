"""Prepare the real 9B matched forecast experiment; never reuse 0.8B scores."""
import argparse
import json
from pathlib import Path
import sys

from common import ROOT, atomic_json, digest, read_config


def prepare(run):
    run = run.resolve()
    if (run / 'plan.json').exists():
        raise FileExistsError('Use the existing immutable plan or a new run')
    run.mkdir(parents=True, exist_ok=True)
    project = ROOT.parents[1]
    qwen = Path('/home/data2/chk/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a').resolve()
    assert (qwen / 'model.safetensors.index.json').is_file()
    original = ROOT / 'runs/linked_20260909_overnight/ours'
    cfg = read_config(original / 'config.json')
    cfg.update(qwen=str(qwen), share_frozen_backbone=True, batch_size=4,
               gradient_accumulation=2, max_stage1_steps=1694, stage1_only=True,
               total_hours=9999, periodic_checkpoint_minutes=30, ce_chunk=32,
               state_condition='slots', validation_every_steps=800,
               validation_pairs=32, audit_task_gradients=False)
    for key in ('train_deadline', 'initialize_stage1', 'max_stage2_steps'):
        cfg.pop(key, None)
    atomic_json(run / 'configs/stage1.json', cfg)
    configs = {}
    for condition in ('no_slots', 'slots', 'shuffled'):
        c = read_config(ROOT / f'configs/overnight_20260913_{condition}.json')
        c.update(qwen=str(qwen), share_frozen_backbone=True, batch_size=4,
                 gradient_accumulation=8, ce_chunk=32,
                 initialize_stage1=str(run / 'stage1/checkpoint_stage1.pt'),
                 total_hours=9999, max_stage2_steps=2400,
                 periodic_checkpoint_minutes=30)
        c.pop('train_deadline', None)
        configs[condition] = c
        atomic_json(run / f'configs/{condition}.json', c)
    # Both source cohorts are immutable and already have complete V-JEPA features.
    for split in ('validate', 'test'):
        assert digest(Path(cfg['cache']) / f'{split}.jsonl') == digest(Path(c['cache']) / f'{split}.jsonl')
    atomic_json(run / 'protocol.json', dict(
        qwen=str(qwen), qwen_revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a',
        stage1=dict(updates=1694, effective_batch=8, presentations=13552,
                    train_observations=11332, cache=cfg['cache'], seed=42),
        stage2=dict(updates=2400, effective_batch=32, presentations=76800,
                    train_pairs=16000, validate_pairs=230, test_pairs=297, test_patients=94, seed=42),
        precision='BF16 frozen pretrained weights; FP32 custom modules; no quantization',
        memory='share immutable pretrained weights across distinct encoder/decoder/target modules; independent LoRA; official untied output head',
        comparison='same data order and update budget within 9B ablation; actual GPU hours not fixed',
        old_08b_stage1=str(original / 'checkpoint_stage1.pt'),
        old_08b_checkpoint_reused=False))
    python = sys.executable
    jobs = []
    allowed = [3, 4, 5, 6, 7]
    def job(name, script, argv, artifacts, deps, success, gpus=1):
        jobs.append(dict(id=name, argv=[python, str(ROOT / script), *map(str, argv)],
                         artifacts=list(map(str, artifacts)), deps=deps,
                         success_fields=success, gpus=gpus, allowed_gpus=allowed,
                         priority=30, max_attempts=1,
                         env=dict(PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')))
    s = run / 'stage1'
    job('qwen9b_stage1', 'train.py', ['--config', run / 'configs/stage1.json', '--run', s],
        [s / 'checkpoint_stage1.pt', s / 'status.json'], [],
        {str(s / 'status.json'): dict(state='complete', stage=1, stage_step=1694)})
    evaluations = []
    for condition in configs:
        train, config = run / condition, run / f'configs/{condition}.json'
        train_job, eval_job = f'qwen9b_{condition}_train', f'qwen9b_{condition}_eval'
        job(train_job, 'train.py', ['--config', config, '--run', train],
            [train / 'checkpoint_final.pt', train / 'status.json', train / 'task_gradient_audit.json'],
            ['qwen9b_stage1'], {str(train / 'status.json'): dict(state='complete', stage=2, stage_step=2400)})
        output = train / 'evaluation_test'
        job(eval_job, 'evaluate.py', ['--config', config, '--mode', 'ours',
                                    '--checkpoint', train / 'checkpoint_final.pt', '--out', output],
            [output / 'metrics.json', output / 'generation.json'], [train_job],
            {str(output / 'generation.json'): dict(count=297, teacher_forcing=False, target_inputs=False,
                                                  state_condition=condition)})
        evaluations.append(eval_job)
    green_paths = [run / f'green/{condition}/test/green_metrics.json' for condition in configs]
    job('qwen9b_green', 'overnight_green.py', ['--run', run], green_paths, evaluations,
        {str(p): dict(status='complete', n=297, completed=297) for p in green_paths})
    atomic_json(run / 'plan.json', dict(project=str(project), jobs=jobs))
    print(json.dumps(dict(run=str(run), jobs=len(jobs), status='prepared_not_launched')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    prepare(parser.parse_args().run)
