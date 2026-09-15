"""Official frozen DINOv2-B/14, final spatial features for Table 2 heads.

No language model, learned slots, or target image is used by the SR encoder.
The default interface averages the 37x37 final patch grid to an 8x8 spatial grid.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get('MEDWORLD_PROJECT', ROOT.parent.parent)).resolve()
DENSE = PROJECT / 'code/medworld_dense_baselines'
DENSE_SOURCE = ROOT.parent / 'medworld_dense_baselines'
if not DENSE_SOURCE.is_dir():
    DENSE_SOURCE = DENSE
sys.path.insert(0, str(DENSE_SOURCE))
from common import atomic, digest, read_rows, write_rows
from frozen_slots_extract import branch_image

ASSETS = ROOT / 'dinov2_assets'
MODEL_ID = 'dinov2_vitb14'
MODELS = (MODEL_ID, 'chexworld')
OFFICIAL_REPO = 'https://github.com/facebookresearch/dinov2'
OFFICIAL_WEIGHTS = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth'


def provenance(assets=ASSETS, image_size=518, grid=8):
    if image_size % 14 or grid < 1 or grid > image_size // 14:
        raise ValueError('image size must divide by 14; output grid cannot exceed patch grid')
    repo, weight = Path(assets) / 'repo', Path(assets) / 'dinov2_vitb14_pretrain.pth'
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
        raise ValueError('official DINOv2 repository has local modifications')
    return dict(model_id=MODEL_ID, official_repository=OFFICIAL_REPO, official_commit=commit,
                official_weights=OFFICIAL_WEIGHTS, weights_sha256=digest(weight),
                weights_bytes=weight.stat().st_size, image_size=image_size, grid=grid,
                native_grid=[image_size // 14] * 2, native_width=768, registers=0,
                backbone_frozen=True, learned_slot_parameters=0, language_model_loaded=False,
                output='final x_norm_patchtokens; class token excluded; adaptive average pool spatial grid',
                preprocessing='PIL RGB; bicubic whole-square resize; ImageNet mean/std; no crop/augmentation',
                interface='8x8 spatial tokens by default; native 768 channels; decoder pads 256 zeros',
                code_sha256=digest(Path(__file__)), torch_version=torch.__version__)


class Extractor:
    def __init__(self, assets=ASSETS, device='cuda', image_size=518, grid=8):
        self.metadata = provenance(assets, image_size, grid)
        self.device = torch.device(device)
        self.image_size, self.grid = image_size, grid
        os.environ['XFORMERS_DISABLED'] = '1'
        self.model = torch.hub.load(str(Path(assets) / 'repo'), MODEL_ID, source='local', pretrained=False)
        state = torch.load(Path(assets) / 'dinov2_vitb14_pretrain.pth', map_location='cpu', weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False).to(self.device)
        self.metadata['backbone_parameters'] = sum(p.numel() for p in self.model.parameters())
        self.metadata['dtype'] = 'float32 parameters; bfloat16 autocast on CUDA, float32 on CPU'

    @torch.inference_mode()
    def batch(self, images):
        if not images:
            raise ValueError('empty image batch')
        tensors = [torch.from_numpy(np.asarray(im.convert('RGB').resize(
            (self.image_size, self.image_size), Image.Resampling.BICUBIC)).copy()).permute(2, 0, 1)
            for im in images]
        pixels = torch.stack(tensors).to(self.device, dtype=torch.float32) / 255
        mean = pixels.new_tensor([.485, .456, .406])[None, :, None, None]
        std = pixels.new_tensor([.229, .224, .225])[None, :, None, None]
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
            tokens = self.model.forward_features((pixels - mean) / std)['x_norm_patchtokens']
        side = self.image_size // 14
        spatial = tokens.float().transpose(1, 2).reshape(len(images), 768, side, side)
        result = F.adaptive_avg_pool2d(spatial, (self.grid, self.grid)).flatten(2).transpose(1, 2)
        if result.requires_grad or not torch.isfinite(result).all():
            raise ValueError('features must be frozen and finite')
        return result.cpu().numpy().astype(np.float16)


def chexworld_provenance(grid=8):
    import chexworld_encoder as official
    vendor = ROOT / 'chexworld_vendor'
    commit = subprocess.check_output(['git', '-C', str(vendor), 'rev-parse', 'HEAD'], text=True).strip()
    return dict(model_id='chexworld', official_repository='https://github.com/LeapLabTHU/CheXWorld',
                official_commit=commit, weights_sha256=digest(official.WEIGHTS),
                official_weights='https://drive.google.com/file/d/1QKUhIWIicl65UXJIGSh7_rYaq5i08iVn/view',
                image_size=224, grid=grid, native_grid=[14, 14], native_width=768,
                backbone_frozen=True, learned_slot_parameters=0, language_model_loaded=False,
                output='official target_encoder final patchtokens; adaptive average pool spatial grid',
                preprocessing='official resize-short256, center-crop224, grayscale3, ImageNet mean/std',
                interface='spatial tokens; native 768 channels; decoder pads 256 zeros',
                code_sha256=digest(Path(__file__)), encoder_code_sha256=digest(Path(official.__file__)),
                torch_version=torch.__version__, license_note='upstream repository lacks top-level LICENSE')


class CheXWorldExtractor:
    def __init__(self, device='cuda', grid=8):
        import chexworld_encoder as official
        self.metadata = chexworld_provenance(grid)
        self.device, self.grid = torch.device(device), grid
        self.model = official.load_encoder().to(self.device)
        self.transform, self.feature_map = official.transform(), official.feature_map
        self.metadata['backbone_parameters'] = sum(p.numel() for p in self.model.parameters())
        self.metadata['dtype'] = 'float32 parameters; bfloat16 autocast on CUDA, float32 on CPU'

    @torch.inference_mode()
    def batch(self, images):
        pixels = torch.stack([self.transform(im) for im in images]).to(self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
            spatial = self.feature_map(self.model, pixels)
        tokens = F.adaptive_avg_pool2d(spatial.float(), (self.grid, self.grid)).flatten(2).transpose(1, 2)
        if not torch.isfinite(tokens).all() or tokens.requires_grad:
            raise ValueError('CheXWorld tokens must be frozen and finite')
        return tokens.cpu().numpy().astype(np.float16)


def array(path, shape):
    if path.exists():
        result = np.load(path, mmap_mode='r+')
        if result.shape != shape or result.dtype != np.float16:
            raise ValueError(f'incompatible feature cache {path}')
        return result
    result = np.lib.format.open_memmap(path, mode='w+', dtype=np.float16, shape=shape)
    result[:] = np.nan
    result.flush()
    return result


def prepare_classification(out, source=None, selection=None, raw_run=None):
    """Use exactly raw-model C0 split overrides, without loading any report text."""
    source = Path(source or PROJECT / 'code/medworld_stage1/data/overnight_20260910')
    selection = Path(selection or PROJECT / 'code/medworld_stage1/data/slot44_20260911_derived_v2/selection.json')
    raw_run = Path(raw_run or PROJECT / 'code/medworld_baselines/runs/raw_models_20260911')
    overrides = json.loads(selection.read_text())['patient_split_overrides']
    rows = []
    for r in read_rows(source / 'observations.jsonl'):
        if not r['tasks']['classification']:
            continue
        rows.append(dict(id=r['id'], index=len(rows), image_index=r['index'],
                         subject_id=r['subject_id'], split=overrides.get(str(r['subject_id']), r['split']),
                         labels=r['labels']))
    patient_sets = [{str(r['subject_id']) for r in rows if r['split'] == s}
                    for s in ('train', 'validate', 'test')]
    if any(a & b for i, a in enumerate(patient_sets) for b in patient_sets[:i]):
        raise ValueError('classification patient leakage')
    for split in ('validate', 'test'):
        expected = {r['id'] for r in read_rows(raw_run / 'cohort' / f'table2_inputs_{split}.jsonl')
                    if r['classification']}
        if expected != {r['id'] for r in rows if r['split'] == split}:
            raise ValueError(f'classification {split} IDs differ from existing raw baseline')
    contract = dict(source=str(source.resolve()), observations_sha256=digest(source / 'observations.jsonl'),
                    selection_sha256=digest(selection), image_sha256=digest(source / 'images.npy'),
                    counts={s: sum(r['split'] == s for r in rows) for s in ('train', 'validate', 'test')},
                    test_alignment='exact existing raw baseline C0 image IDs and official labels',
                    input='image only; current report excluded', labels='official 13 CheXpert; blank/uncertain masked')
    out = Path(out)
    if (out / 'classification_data.json').exists() and json.loads((out / 'classification_data.json').read_text()) != contract:
        raise ValueError('classification data contract changed')
    atomic(out / 'classification_data.json', contract)
    write_rows(out / 'classification_observations.jsonl', rows)
    return rows, contract


def extract(args):
    torch.set_num_threads(args.threads)
    out = args.run / args.model
    out.mkdir(parents=True, exist_ok=True)
    with (out / f'{args.task}_extraction.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _extract(args, out)


def _extract(args, out):
    meta = provenance(args.assets, args.image_size, args.grid) if args.model == MODEL_ID else chexworld_provenance(args.grid)
    if args.task == 'dense':
        data = args.data_run / 'data'
        rows = read_rows(data / 'observations.jsonl')
        manifest = json.loads((data / 'manifest.json').read_text())
        if digest(data / 'observations.jsonl') != manifest['cohort_sha256']:
            raise ValueError('dense cohort mismatch')
        for name, key in [('images.npy', 'image_sha256'), ('lr_images.npy', 'lr_image_sha256')]:
            if digest(data / name) != manifest[key]:
                raise ValueError(f'dense input mismatch {name}')
        images = np.load(data / 'images.npy', mmap_mode='r')
        lr_images = np.load(data / 'lr_images.npy', mmap_mode='r')
        branches = ['hr', 'lr']
        required = np.asarray([['segmentation' in r['tasks'], 'sr' in r['tasks']] for r in rows])
        contract = dict(cohort_sha256=manifest['cohort_sha256'], image_sha256=manifest['image_sha256'],
                        lr_image_sha256=manifest['lr_image_sha256'], data_run=str(args.data_run.resolve()),
                        sr_input='ONLY shared prepared uint8 LR; HR used only as reconstruction target')
        def get_image(branch, i):
            return branch_image(branch, i, images, lr_images)
    else:
        rows, data_contract = prepare_classification(out)
        images = np.load(Path(data_contract['source']) / 'images.npy', mmap_mode='r')
        branches, required = ['classification'], np.ones((len(rows), 1), dtype=bool)
        contract = dict(classification_data_sha256=digest(out / 'classification_data.json'),
                        cohort_sha256=digest(out / 'classification_observations.jsonl'))
        def get_image(branch, i):
            return Image.fromarray(np.array(images[rows[i]['image_index']], copy=True)).convert('RGB')
    if any(r['index'] != i for i, r in enumerate(rows)):
        raise ValueError('row order differs from cache index')
    contract.update(model=meta, shape=[len(rows), args.grid ** 2, 768], dtype='float16',
                    branches=branches, required_counts=dict(zip(branches, required.sum(0).tolist())),
                    extraction_batch_size=args.batch_size)
    contract_path = out / f'{args.task}_feature_contract.json'
    done_path = out / f'{args.task}_features_done.npy'
    complete_path = out / f'{args.task}_features_complete.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('feature contract changed; use a fresh run')
    if done_path.exists() and not contract_path.exists():
        raise ValueError('feature completion exists without provenance')
    atomic(contract_path, contract)
    done = np.load(done_path) if done_path.exists() else np.zeros_like(required)
    if done.shape != required.shape or done.dtype != bool or np.any(done & ~required):
        raise ValueError('invalid feature completion mask')
    caches = {b: array(out / f'{b}_features.npy', tuple(contract['shape'])) for b in branches}
    for col, b in enumerate(branches):
        if not np.isfinite(caches[b][done[:, col]]).all():
            raise ValueError('completed features contain nonfinite values')
    if complete_path.exists():
        marker = json.loads(complete_path.read_text())
        if not done[required].all() or marker['contract_sha256'] != digest(contract_path):
            raise ValueError('invalid completed feature contract')
        for b in branches:
            if digest(out / f'{b}_features.npy') != marker['feature_sha256'][b]:
                raise ValueError('completed feature bytes changed')
        print('features already complete', args.task, flush=True)
        return
    encoder = Extractor(args.assets, args.device, args.image_size, args.grid) if args.model == MODEL_ID else CheXWorldExtractor(args.device, args.grid)
    atomic(out / 'model_metadata.json', encoder.metadata)
    started, completed = time.time(), 0
    for col, branch in enumerate(branches):
        pending = np.flatnonzero(required[:, col] & ~done[:, col]).tolist()
        for start in range(0, len(pending), args.batch_size):
            ids = pending[start:start + args.batch_size]
            if args.limit:
                ids = ids[:max(0, args.limit - completed)]
            if not ids:
                return
            caches[branch][ids] = encoder.batch([get_image(branch, i) for i in ids])
            caches[branch].flush()
            done[ids, col] = True
            np.save(done_path.with_suffix('.tmp.npy'), done)
            os.replace(done_path.with_suffix('.tmp.npy'), done_path)
            completed += len(ids)
            progress = dict(done=int(done[required].sum()), total=int(required.sum()),
                            seconds=time.time() - started, physical_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'))
            atomic(out / f'{args.task}_features_progress.json', progress)
            if completed == len(ids) or completed % 128 < args.batch_size:
                print(args.task, json.dumps(progress), flush=True)
            if args.limit and completed >= args.limit and not done[required].all():
                return
    if not done[required].all():
        raise ValueError('incomplete feature extraction')
    atomic(complete_path, dict(complete=True, contract_sha256=digest(contract_path),
                              feature_sha256={b: digest(out / f'{b}_features.npy') for b in branches},
                              seconds=time.time() - started, count=int(required.sum())))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--model', choices=MODELS, default=MODEL_ID)
    p.add_argument('--data-run', type=Path, default=DENSE / 'runs/dense_20260912')
    p.add_argument('--task', choices=['dense', 'classification'], default='dense')
    p.add_argument('--assets', type=Path, default=ASSETS)
    p.add_argument('--image-size', type=int, default=518)
    p.add_argument('--grid', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    if a.batch_size < 1 or a.limit is not None and a.limit < 1:
        p.error('batch size and limit must be positive')
    return a


if __name__ == '__main__':
    extract(parse_args())
