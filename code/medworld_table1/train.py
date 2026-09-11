"""Wall-time training, atomic resumable checkpoints, and matched-step control."""
import argparse
import json
import os
import signal
import time
from datetime import datetime
from pathlib import Path

from common import atomic_json, atomic_torch, digest, read_config, seed_all
import torch
from data import Corpus
from model import MedWorld

STOP = False


def stop_handler(signum, frame):
    global STOP
    STOP = True


def optimizer_for(model, cfg):
    lora, other = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (lora if 'lora_' in name else other).append(p)
    return torch.optim.AdamW([{'params': lora, 'lr': cfg['lora_learning_rate'], 'base_lr': cfg['lora_learning_rate']},
                             {'params': other, 'lr': cfg['learning_rate'], 'base_lr': cfg['learning_rate']}],
                             betas=(.9, .95), weight_decay=.01)


@torch.no_grad()
def validate(model, corpus, stage, count):
    model.eval()
    totals, n = {}, 0
    rows = corpus.pairs['validate'][:count]
    for i in range(0, len(rows), corpus.cfg['batch_size']):
        chosen = rows[i:i+corpus.cfg['batch_size']]
        batch = corpus.batch(chosen)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, parts = model.losses(batch, stage)
        for key, value in dict(loss=loss, **parts).items():
            totals[key] = totals.get(key, 0.) + float(value) * len(chosen)
        n += len(chosen)
    model.train()
    return {k: v/n for k, v in totals.items()}


def train(args, cfg):
    os.umask(0o077)
    seed_all(cfg['seed'])
    run = Path(args.run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'status.json').exists() and not args.resume:
        raise FileExistsError('Run exists. Use --resume or a new --run.')
    atomic_json(run / 'config.json', cfg)
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    signatures = {f: digest(Path(cfg['cache']) / f) for f in ['manifest.json', 'observations.jsonl', 'train.jsonl', 'validate.jsonl', 'test.jsonl', 'features.json']}
    signatures['qwen_config'] = digest(Path(cfg['qwen']) / 'config.json')
    signatures['qwen_weights'] = digest(Path(cfg['qwen']) / 'model.safetensors-00001-of-00001.safetensors')
    atomic_json(run / 'inputs.json', signatures)
    atomic_json(run / 'software.json', dict(torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), source_hashes={p.name: digest(p) for p in Path(__file__).parent.glob('*.py')}))
    if args.mode == 'direct':
        from direct import DirectQwen
        model = DirectQwen(cfg).to('cuda')
    else:
        model = MedWorld(cfg).to('cuda')
    corpus = Corpus(cfg, model.tokenizer)
    stage, step, stage_step, elapsed, save4_step = 1, 0, 0, 0., None
    if args.mode == 'direct':
        stage = 2
    followed = Path(args.follow).resolve() if args.follow else None
    if followed:
        checkpoint = torch.load(followed / 'checkpoint_stage1.pt', map_location='cpu', weights_only=False)
        if checkpoint['signatures'] != signatures:
            raise ValueError('Matched control input hashes differ.')
        model.load_compact(checkpoint['model'])
        model.begin_stage2(frozen_encoder=True)
        stage, step, elapsed = 2, checkpoint['step'], checkpoint['elapsed_seconds']
    optimizer = optimizer_for(model, cfg)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        if checkpoint['signatures'] != signatures or checkpoint['config'] != cfg:
            raise ValueError('Resume inputs/config differ.')
        stage, step, stage_step = checkpoint['stage'], checkpoint['step'], checkpoint['stage_step']
        elapsed, save4_step = checkpoint['elapsed_seconds'], checkpoint['save4_step']
        if args.mode != 'direct' and stage == 2 and model.target_encoder is None:
            model.begin_stage2(frozen_encoder=bool(followed))
        model.load_compact(checkpoint['model'])
        optimizer = optimizer_for(model, cfg)
        optimizer.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['torch_rng'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    model.train()
    started = time.time()
    history = run / 'metrics.jsonl'
    deadline = datetime.fromisoformat(cfg['train_deadline']).timestamp() if cfg.get('train_deadline') else None
    last_periodic = time.monotonic()
    best_validation = {}

    def status(state='running', **extra):
        record = dict(state=state, pid=os.getpid(), stage=stage, step=step, stage_step=stage_step,
                      elapsed_seconds=elapsed, train_hours=elapsed/3600, save4_step=save4_step,
                      started_unix=started, updated_unix=time.time(), mode='matched' if followed else args.mode, **extra)
        atomic_json(run / 'status.json', record)
        return record

    def save(name):
        atomic_torch(run / name, dict(format_version=1, model=model.compact_state(), optimizer=optimizer.state_dict(),
            stage=stage, step=step, stage_step=stage_step, elapsed_seconds=elapsed, save4_step=save4_step,
            config=cfg, signatures=signatures, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            followed=str(followed) if followed else None, mode='matched' if followed else args.mode))
        print(f'saved {name} step={step} stage_step={stage_step} hours={elapsed/3600:.4f}', flush=True)

    status()
    print(f'training mode={"matched" if followed else args.mode} params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}', flush=True)
    try:
        while not STOP:
            reference = None
            if followed:
                reference = json.loads((followed / 'status.json').read_text())
                limit = reference['stage_step'] if reference['stage'] == 2 else 0
                if stage_step >= limit:
                    if reference['state'] == 'complete':
                        break
                    if reference['state'] in ('failed', 'interrupted'):
                        raise RuntimeError('Primary run stopped; refusing to silently shorten matched control.')
                    time.sleep(1)
                    continue
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            multiplier = min(1., (stage_step+1) / cfg['warmup_steps'])
            for group in optimizer.param_groups:
                group['lr'] = group['base_lr'] * multiplier
            totals = {}
            for micro in range(cfg['gradient_accumulation']):
                batch = corpus.training_batch(stage, stage_step, micro)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, parts = model.losses(batch, stage)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss')
                (loss / cfg['gradient_accumulation']).backward()
                for k, v in dict(loss=loss, **parts).items():
                    totals[k] = totals.get(k, 0.) + float(v.detach()) / cfg['gradient_accumulation']
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            duration = time.monotonic() - tick
            elapsed += duration
            step, stage_step = step+1, stage_step+1
            # Scheduled writes happen before publishing progress, so the matched
            # run cannot advance beyond the corresponding save boundary.
            boundary = reference['save4_step'] if reference else None
            if save4_step is None and ((followed and boundary is not None and stage_step == boundary) or
                    (not followed and stage == 2 and (elapsed >= cfg['checkpoint_hours']*3600 or (args.smoke_steps and stage_step == 1)))):
                save4_step = stage_step
                save('checkpoint_4h.pt')
            row = status(loss=totals['loss'], step_seconds=duration)
            row.update(totals, grad_norm=float(norm), lr_multiplier=multiplier)
            with history.open('a') as f:
                f.write(json.dumps(row, allow_nan=False) + '\n')
            if step % 10 == 0 or args.smoke_steps or step == 1:
                print(f'stage={stage} step={stage_step} hours={elapsed/3600:.3f} sec/step={duration:.2f} loss={totals}', flush=True)
            if stage == 1 and (elapsed >= cfg['stage1_hours']*3600 or (args.smoke_steps and stage_step >= args.smoke_steps)):
                save('checkpoint_stage1.pt')
                model.begin_stage2()
                stage, stage_step = 2, 0
                optimizer = optimizer_for(model, cfg)
                status()
            elif stage == 2 and not followed and (elapsed >= cfg['total_hours']*3600 or
                    (deadline is not None and time.time() >= deadline) or (args.smoke_steps and stage_step >= args.smoke_steps)):
                break
            if cfg.get('periodic_checkpoint_minutes') and time.monotonic()-last_periodic >= cfg['periodic_checkpoint_minutes']*60:
                save('checkpoint_latest.pt')
                last_periodic = time.monotonic()
            if cfg['validation_every_steps'] and stage_step and stage_step % cfg['validation_every_steps'] == 0:
                metrics = validate(model, corpus, stage, cfg['validation_pairs'])
                record = dict(stage=stage, step=step, metrics=metrics, updated_unix=time.time())
                atomic_json(run / 'validation_latest.json', record)
                with (run / 'validation_history.jsonl').open('a') as f:
                    f.write(json.dumps(record)+'\n')
                if metrics['text'] < best_validation.get(stage, float('inf')):
                    best_validation[stage] = metrics['text']
                    save(f'checkpoint_best_stage{stage}.pt')
                print(f'validation step={step}: {metrics}', flush=True)
        save('checkpoint_interrupted.pt' if STOP else 'checkpoint_final.pt')
        metrics = validate(model, corpus, stage, min(4, cfg['validation_pairs']) if args.smoke_steps else cfg['validation_pairs'])
        atomic_json(run / 'validation_final.json', dict(stage=stage, step=step, metrics=metrics))
        status('interrupted' if STOP else 'complete',
            stopping_reason='signal' if STOP else ('wall_clock_deadline' if deadline and time.time() >= deadline else 'training_budget_or_matched_steps'))
    except Exception as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/pilot.json')
    parser.add_argument('--run', required=True)
    parser.add_argument('--mode', choices=['ours', 'direct'], default='ours')
    parser.add_argument('--follow', help='Ours run directory; starts from its Stage-1 checkpoint and follows its exact Stage-2 steps.')
    parser.add_argument('--resume')
    parser.add_argument('--smoke-steps', type=int, default=0)
    args = parser.parse_args()
    train(args, read_config(args.config))
