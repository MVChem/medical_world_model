"""Cache the frozen V-JEPA 2.1 image branch, retaining an 8 x 8 grid."""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from common import atomic_json, digest, load_rows, read_config, seed_all


class Images(Dataset):
    def __init__(self, rows, size):
        self.rows, self.size = rows, size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        with Image.open(self.rows[i]['image']) as image:
            image = ImageOps.pad(image.convert('RGB'), (self.size, self.size), method=Image.Resampling.BICUBIC)
            array = np.array(image, dtype=np.float32) / 255.
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensor = (tensor - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
        return i, tensor


def extract(cfg):
    seed_all(cfg['seed'])
    cache = Path(cfg['cache'])
    rows = load_rows(cache / 'observations.jsonl')
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'vjepa2'))
    from app.vjepa_2_1.models.vision_transformer import vit_base
    model = vit_base(img_size=(384, 384), patch_size=16, num_frames=64, tubelet_size=2,
                     use_sdpa=True, use_rope=True, img_temporal_dim_size=1, interpolate_rope=True)
    weights = torch.load(cfg['vjepa_checkpoint'], map_location='cpu', weights_only=True)['ema_encoder']
    weights = {k.replace('module.', '').replace('backbone.', ''): v for k, v in weights.items()}
    model.load_state_dict(weights, strict=True)
    del weights
    model.requires_grad_(False).eval().to('cuda', dtype=torch.bfloat16)
    grid = cfg['visual_grid']
    feature_path = cache / 'vjepa_features.npy'
    if (cache / 'features.json').exists():
        raise FileExistsError('Features already complete.')
    features = np.lib.format.open_memmap(feature_path.with_suffix('.tmp.npy'), mode='w+',
                                         dtype=np.float16, shape=(len(rows), grid * grid, 768))
    missing = list(range(len(rows)))
    reused = 0
    if cfg.get('feature_reuse_cache'):
        import json
        old = Path(cfg['feature_reuse_cache'])
        metadata = json.loads((old / 'features.json').read_text())
        old_cfg = json.loads((old / 'manifest.json').read_text())['config']
        if metadata['checkpoint_sha256'] != digest(cfg['vjepa_checkpoint']) or any(old_cfg[k] != cfg[k] for k in ('image_size', 'visual_grid')):
            raise ValueError('Feature reuse checkpoint or preprocessing differs')
        if metadata['observations_sha256'] != digest(old / 'observations.jsonl'):
            raise ValueError('Feature reuse observation manifest differs')
        old_rows = load_rows(old / 'observations.jsonl')
        lookup = {(r['id'], r['image']): i for i, r in enumerate(old_rows)}
        old_features = np.load(old / 'vjepa_features.npy', mmap_mode='r')
        missing = []
        for i, row in enumerate(rows):
            j = lookup.get((row['id'], row['image']))
            if j is None:
                missing.append(i)
            else:
                features[i] = old_features[j]
                reused += 1
        del old_features
        print(f'Reused {reused} immutable features; extracting {len(missing)} new images', flush=True)
    loader = DataLoader(torch.utils.data.Subset(Images(rows, cfg['image_size']), missing), batch_size=cfg['feature_batch_size'],
                        num_workers=cfg['feature_workers'], pin_memory=True)
    start = time.monotonic()
    with torch.inference_mode():
        for batch, (idx, pixels) in enumerate(loader):
            # Upstream RoPE constructs float32 frequencies; autocast reconciles
            # Q/K with BF16 V at SDPA without changing the upstream checkout.
            with torch.autocast('cuda', dtype=torch.bfloat16):
                tokens = model(pixels.to('cuda', dtype=torch.bfloat16).unsqueeze(2))
            assert tokens.shape[1:] == (576, 768), tokens.shape
            tokens = tokens.transpose(1, 2).reshape(-1, 768, 24, 24)
            tokens = F.adaptive_avg_pool2d(tokens, (grid, grid)).flatten(2).transpose(1, 2)
            assert torch.isfinite(tokens).all()
            features[idx.numpy()] = tokens.float().cpu().numpy()
            if batch % 50 == 0:
                print(f'features {int(idx[-1])+1}/{len(rows)} elapsed={time.monotonic()-start:.0f}s', flush=True)
    features.flush()
    del features
    os.replace(feature_path.with_suffix('.tmp.npy'), feature_path)
    atomic_json(cache / 'features.json', dict(shape=[len(rows), grid * grid, 768],
        checkpoint_sha256=digest(cfg['vjepa_checkpoint']), observations_sha256=digest(cache / 'observations.jsonl'),
        reused_observations=reused, reuse_cache=cfg.get('feature_reuse_cache'),
        encoder='V-JEPA 2.1 ViT-B EMA image branch, strict checkpoint load, last normalized layer',
        preprocessing='384px aspect-preserving black letterbox, bicubic, ImageNet mean/std, image temporal size 1',
        pooling='24x24 to 8x8 adaptive spatial average pooling', dtype='float16'))
    print('feature cache complete', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/pilot.json')
    extract(read_config(p.parse_args().config))
