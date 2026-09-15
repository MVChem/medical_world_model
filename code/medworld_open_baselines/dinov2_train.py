"""Train matched supervised heads on frozen DINOv2 final spatial tokens.

Dense tasks retain the September 13 image decoder, trainable parameter count,
initialization, losses, and scorer. Only fixed token coordinates change from
four depth positions to the spatial grid. Classification is an independent
image-only supervised head on the exact C0 353-image test set.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinov2_features import DENSE, DENSE_SOURCE, ROOT, PROJECT, MODEL_ID, MODELS, atomic, digest, read_rows, write_rows
import frozen_slots_train as shared

STAGE_SOURCE = ROOT.parent / 'medworld_stage1'
if not STAGE_SOURCE.is_dir():
    STAGE_SOURCE = PROJECT / 'code/medworld_stage1'


class SpatialHead(shared.FrozenSlotHead):
    def __init__(self, task, grid=8):
        super().__init__(task)
        self.grid = grid
        # Same trainable modules as the reference head; only a fixed buffer changes.
        self.depth_position = shared.positional_encoding(grid, grid)

    def forward(self, image, tokens):
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (self.grid ** 2, 1024):
            raise ValueError('invalid final spatial token shape')
        z = F.interpolate(self.stem(image), (32, 32), mode='bilinear', align_corners=False)
        query = self.query_normalize(z.flatten(2).transpose(1, 2) + self.image_position)
        values = self.slot_project(self.slot_normalize(tokens.detach().float())) + self.depth_position
        attended, _ = self.attention(query, values, values, need_weights=False)
        attended = attended.transpose(1, 2).reshape(image.shape[0], 64, 32, 32)
        result = self.decode(self.fuse(torch.cat([z, attended], dim=1)))
        if self.task == 'sr':
            return F.interpolate(image, scale_factor=4, mode='bicubic', align_corners=False) + .1 * result
        return result


class DenseCorpus(shared.SlotCorpus):
    def __init__(self, args):
        super().__init__(args.data_run, args.run, args.model, args.task, 'image_only',
                         args.device, args.train_limit, args.seed, args.pseudo_path)
        self.condition = 'spatial_tokens'
        root = args.run / args.model
        path = root / 'dense_feature_contract.json'
        self.feature_contract = json.loads(path.read_text())
        self.slot_contract_hash = digest(path)  # shared scorer's generic representation hash field
        if self.feature_contract['cohort_sha256'] != self.cohort_hash:
            raise ValueError('feature cohort mismatch')
        for k in ('image_sha256', 'lr_image_sha256'):
            if self.feature_contract[k] != self.manifest[k]:
                raise ValueError('feature image fingerprint mismatch')
        col, branch = (1, 'lr') if args.task == 'sr' else (0, 'hr')
        self.feature = np.load(root / f'{branch}_features.npy', mmap_mode='r')
        done = np.load(root / 'dense_features_done.npy')
        required = [i for ids in self.pool.values() for i in ids]
        if self.feature.shape != tuple(self.feature_contract['shape']) or not done[required, col].all():
            raise ValueError('incomplete/wrong-shaped spatial features')
        marker = json.loads((root / 'dense_features_complete.json').read_text())
        if marker['contract_sha256'] != digest(path) or digest(root / f'{branch}_features.npy') != marker['feature_sha256'][branch]:
            raise ValueError('feature completion fingerprint mismatch')
        self.grid = self.feature_contract['model']['grid']

    def batch(self, ids):
        image, _, target, mask = super().batch(ids)
        feature = torch.from_numpy(np.array(self.feature[ids], copy=True)).to(self.device).float()
        if not torch.isfinite(feature).all():
            raise ValueError('nonfinite spatial features')
        return image, F.pad(feature, (0, 256)), target, mask


def classification_components():
    # Import actual historical head/loss/scorer, rather than a similar reimplementation.
    sys.path.insert(0, str(STAGE_SOURCE))
    from networks import Classification, masked_classification
    from evaluation import classification_metrics
    return Classification, masked_classification, classification_metrics


class ClassificationCorpus:
    def __init__(self, args):
        self.device = torch.device(args.device)
        root = args.run / args.model
        self.rows = read_rows(root / 'classification_observations.jsonl')
        self.pool = {s: [i for i, r in enumerate(self.rows) if r['split'] == s]
                     for s in ('train', 'validate', 'test')}
        if args.train_limit:
            self.pool['train'] = self.pool['train'][:args.train_limit]
        self.feature = np.load(root / 'classification_features.npy', mmap_mode='r')
        done = np.load(root / 'classification_features_done.npy')
        self.feature_contract = json.loads((root / 'classification_feature_contract.json').read_text())
        self.cohort_hash = digest(root / 'classification_observations.jsonl')
        if self.cohort_hash != self.feature_contract['cohort_sha256'] or not done.all():
            raise ValueError('classification feature cohort incomplete/mismatched')
        marker = json.loads((root / 'classification_features_complete.json').read_text())
        if marker['contract_sha256'] != digest(root / 'classification_feature_contract.json'):
            raise ValueError('classification feature contract mismatch')
        if marker['feature_sha256']['classification'] != digest(root / 'classification_features.npy'):
            raise ValueError('classification feature cache changed')
        labels = np.asarray([self.rows[i]['labels'] for i in self.pool['train']])
        self.pos_weight = torch.tensor(np.clip((labels == 0).sum(0) / np.maximum((labels == 1).sum(0), 1),
                                              .25, 10), dtype=torch.float32, device=self.device)

    def batch(self, ids):
        features = torch.from_numpy(np.array(self.feature[ids], copy=True)).to(self.device).float()
        labels = torch.tensor([self.rows[i]['labels'] for i in ids], device=self.device)
        if not torch.isfinite(features).all():
            raise ValueError('nonfinite classification features')
        return F.pad(features, (0, 256)), labels


@torch.inference_mode()
def evaluate_classification(model, corpus, out, args, contract):
    _, _, scorer = classification_components()
    model.eval()
    result = {}
    for split in ('validate', 'test'):
        records = []
        for start in range(0, len(corpus.pool[split]), args.microbatch):
            ids = corpus.pool[split][start:start + args.microbatch]
            feature, labels = corpus.batch(ids)
            with shared.autocast(corpus.device):
                scores = model(feature).float().sigmoid().cpu().tolist()
            records.extend(dict(id=corpus.rows[i]['id'], subject_id=corpus.rows[i]['subject_id'],
                                labels=corpus.rows[i]['labels'], scores=score) for i, score in zip(ids, scores))
        if len(records) != len(corpus.pool[split]):
            raise ValueError('classification evaluation lost records')
        write_rows(out / f'{split}_per_sample.jsonl', records)
        result[split] = scorer([r['labels'] for r in records], [r['scores'] for r in records])
    atomic(out / 'metrics.json', dict(model=args.model, task='classification', epochs=args.epochs,
           train_n=len(corpus.pool['train']), metrics=result, complete_requested_epochs=True,
           cohort_sha256=corpus.cohort_hash, contract_sha256=digest(out / 'contract.json'),
           scope='image-only supervised head; exact existing C0 evaluation IDs; official 13 CheXpert labels',
           checkpoint='final epoch; no test-based checkpoint selection'))
    return result


def train(args):
    out = args.out or args.run / args.model / f'{args.task}_spatial'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with shared.StopRequest(args.stop_at) as stop:
            return _train(args, out, stop)


def _train(args, out, stop):
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    classification = args.task == 'classification'
    corpus = ClassificationCorpus(args) if classification else DenseCorpus(args)
    if classification:
        head, cls_loss, _ = classification_components()
        model = head(1024).to(corpus.device)
    else:
        model = SpatialHead(args.task, corpus.grid).to(corpus.device)
    contract = dict(model=args.model, task=args.task, condition='spatial_tokens', seed=args.seed,
                    epochs=args.epochs, batch_size=args.batch_size, microbatch=args.microbatch,
                    train_n=len(corpus.pool['train']), split_counts={k: len(v) for k, v in corpus.pool.items()},
                    learning_rate=args.learning_rate, weight_decay=.01, optimizer='AdamW',
                    learning_rate_schedule='epoch cosine decay to 10% floor',
                    cohort_sha256=corpus.cohort_hash, features=corpus.feature_contract,
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    loss='class-balanced masked BCE' if classification else
                    ('valid-pixel MSE' if args.task == 'sr' else 'valid-pixel BCE plus soft Dice'),
                    architecture='stage1 Classification(1024) applied to 64 final spatial tokens' if classification else
                    'September13 image CNN + query cross attention; fixed 2D token positions; same trainable modules',
                    initialization_parameters_sha256=hashlib.sha256(b''.join(
                        p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest(),
                    source_sha256=digest(Path(__file__)), shared_train_sha256=digest(Path(shared.__file__)),
                    shared_heads_sha256=digest(DENSE_SOURCE / 'heads.py'), backbone_frozen=True,
                    input='image-only', amp='bfloat16' if corpus.device.type == 'cuda' else 'float32',
                    split_indices_sha256={s: hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                                          for s, ids in corpus.pool.items()})
    if classification:
        contract.update(pos_weight=corpus.pos_weight.cpu().tolist(),
                        classification_head_sha256=digest(STAGE_SOURCE / 'networks.py'),
                        classification_scorer_sha256=digest(STAGE_SOURCE / 'evaluation.py'),
                        budget_scope='independent 20-epoch head; matched C0 cohort, not historical multitask update budget')
    else:
        contract.update(pseudo_sha256=shared.target_fingerprint(corpus.pseudo_path, args.run)
                        if args.task == 'segmentation' else None,
                        human_mask_sha256=corpus.manifest.get('human_mask_sha256'),
                        data_run=str(args.data_run.resolve()))
    if (out / 'contract.json').exists() and json.loads((out / 'contract.json').read_text()) != contract:
        raise ValueError('training contract changed; use fresh output')
    atomic(out / 'contract.json', contract)
    if (out / 'metrics.json').exists():
        print('already complete', out, flush=True)
        return
    atomic(out / 'initialization.json', dict(parameters=contract['trainable_parameters'],
           state_sha256=shared.state_hash(model), parameters_sha256=contract['initialization_parameters_sha256'],
           effective_batch=args.batch_size, microbatch=args.microbatch))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    first, cursor, step, prior_seconds, history = 0, 0, 0, 0., []
    partial_total, partial_n = 0., 0
    checkpoint = out / 'checkpoint.pt'
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['contract'] != contract:
            raise ValueError('checkpoint contract mismatch')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        shared.restore_rng(saved['rng'])
        first, cursor, step = saved['epoch'], saved['cursor'], saved['step']
        prior_seconds, history = saved['seconds'], saved['history']
        partial_total, partial_n = saved['partial_total'], saved['partial_n']
    started, last_checkpoint = time.time(), time.time()

    def save(epoch, next_cursor=0, total=0., n=0):
        nonlocal last_checkpoint
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), contract=contract,
                        epoch=epoch, cursor=next_cursor, step=step, history=history, rng=shared.capture_rng(),
                        seconds=prior_seconds + time.time() - started, partial_total=total, partial_n=n),
                   out / 'checkpoint.tmp.pt')
        os.replace(out / 'checkpoint.tmp.pt', checkpoint)
        last_checkpoint = time.time()

    def loss_for(ids):
        batch = corpus.batch(ids)
        with shared.autocast(corpus.device):
            prediction = model(batch[0]) if classification else model(batch[0], batch[1])
        return cls_loss(prediction, batch[1], corpus.pos_weight) if classification else shared.objective(
            args.task, prediction, batch[2], batch[3])

    for epoch in range(first, args.epochs):
        model.train()
        ids = np.random.default_rng(args.seed + epoch).permutation(corpus.pool['train']).tolist()
        total, n = (partial_total, partial_n) if epoch == first else (0., 0)
        lr = args.learning_rate * (.1 + .9 * .5 * (1 + np.cos(np.pi * epoch / args.epochs)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        for start in range(cursor if epoch == first else 0, len(ids), args.batch_size):
            if stop.requested():
                save(epoch, start, total, n)
                atomic(out / 'progress.json', dict(status='stopped', epoch=epoch, step=step, reason=stop.reason))
                raise SystemExit(124)
            chunk = ids[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            # Masked classification balances classes over the whole effective batch.
            # Compute its full batch together; dense gradients can use microbatches.
            microbatch = len(chunk) if classification else args.microbatch
            for pos in range(0, len(chunk), microbatch):
                ii = chunk[pos:pos + microbatch]
                loss = loss_for(ii)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite training loss')
                (loss * len(ii) / len(chunk)).backward()
                total += float(loss.detach()) * len(ii)
                n += len(ii)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if time.time() - last_checkpoint >= args.checkpoint_seconds:
                save(epoch, start + len(chunk), total, n)
            if step % 50 == 0:
                atomic(out / 'progress.json', dict(status='training', epoch=epoch + 1, step=step,
                       samples_in_epoch=n, training_loss=total / n, seconds=prior_seconds + time.time() - started))
        model.eval()
        val_total, val_n = 0., 0
        with torch.inference_mode():
            for start in range(0, len(corpus.pool['validate']), args.microbatch):
                ii = corpus.pool['validate'][start:start + args.microbatch]
                val_total += float(loss_for(ii)) * len(ii)
                val_n += len(ii)
        record = dict(epoch=epoch + 1, step=step, train_loss=total / n, validate_loss=val_total / val_n,
                      learning_rate=lr, order_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                      seconds=prior_seconds + time.time() - started)
        history.append(record)
        save(epoch + 1)
        write_rows(out / 'epochs.jsonl', history)
        if epoch + 1 in (5, 10, 20):
            torch.save(dict(model=model.state_dict(), epoch=epoch + 1, contract=contract), out / f'epoch_{epoch+1}.pt')
        atomic(out / 'progress.json', dict(**record, status='epoch_complete'))
        print(args.model, args.task, json.dumps(record), flush=True)
    result = evaluate_classification(model, corpus, out, args, contract) if classification else shared.evaluate(
        model, corpus, out, args.microbatch, args.epochs, args.seed, contract)
    atomic(out / 'progress.json', dict(status='complete', epoch=args.epochs, step=step,
                                      seconds=prior_seconds + time.time() - started))
    print(json.dumps(result), flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--model', choices=MODELS, default=MODEL_ID)
    p.add_argument('--data-run', type=Path, default=DENSE / 'runs/dense_20260912')
    p.add_argument('--task', choices=['segmentation', 'sr', 'classification'], required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--microbatch', type=int, default=4)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=20260913)
    p.add_argument('--train-limit', type=int)
    p.add_argument('--device', default='cuda')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--pseudo-path', type=Path)
    p.add_argument('--stop-at')
    p.add_argument('--checkpoint-seconds', type=float, default=120.)
    a = p.parse_args()
    if min(a.epochs, a.batch_size, a.microbatch) < 1 or a.checkpoint_seconds <= 0:
        p.error('epochs, batch sizes and checkpoint seconds must be positive')
    return a


if __name__ == '__main__':
    train(parse_args())
