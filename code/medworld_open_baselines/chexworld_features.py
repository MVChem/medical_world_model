"""Shared BioViL-T/CheXWorld cache extractor for immutable Table-1 manifests.

The legacy vjepa_features.npy basename is only Corpus's storage ABI; metadata
and run configuration always name the real pretrained representation.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent
TABLE1 = ROOT.parent / 'medworld_table1'
sys.path.insert(0, str(TABLE1))
from common import atomic_json, digest, load_rows


class Images(Dataset):
    def __init__(self, rows, transform, load_image=None):
        self.rows, self.transform, self.load_image = rows, transform, load_image

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.load_image:
            return i, self.transform(self.load_image(self.rows[i]['image']))
        with Image.open(self.rows[i]['image']) as image:
            pixels = self.transform(image.convert('RGB'))
        return i, pixels


def extract(args):
    module = __import__('biovil_encoder' if args.encoder == 'biovil_t' else 'chexworld_encoder')
    checkpoint = Path(args.checkpoint or module.WEIGHTS).resolve()
    cfg = json.loads(Path(args.config).read_text())
    cache = Path(cfg['cache'])
    observations = cache / 'observations.jsonl'
    rows = load_rows(observations)
    if args.limit:
        rows = rows[:args.limit]
    output = Path(args.out) if args.out else cache
    output.mkdir(parents=True, exist_ok=True)
    feature_path = output / 'vjepa_features.npy'
    metadata = output / 'features.json'
    signature = dict(encoder=args.encoder, checkpoint_sha256=digest(checkpoint),
                     observations_sha256=digest(observations), count=len(rows))
    if metadata.exists():
        prior = json.loads(metadata.read_text())
        if all(prior.get(k) == v for k, v in signature.items()) and feature_path.exists():
            print('Matching complete feature cache already exists', flush=True)
            return
        raise FileExistsError('Existing feature cache has different provenance')
    torch.set_num_threads(4)
    model = module.load_encoder(checkpoint).to(args.device)
    loader = DataLoader(Images(rows, module.transform(), getattr(module, 'load_image', None)), batch_size=args.batch_size,
                        num_workers=args.workers, pin_memory=args.device.startswith('cuda'))
    features = None
    started = time.monotonic()
    with torch.inference_mode():
        for idx, pixels in loader:
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.device.startswith('cuda')):
                maps = module.feature_map(model, pixels.to(args.device))
                tokens = F.adaptive_avg_pool2d(maps, (8, 8)).flatten(2).transpose(1, 2)
            if not torch.isfinite(tokens).all():
                raise FloatingPointError('Non-finite pretrained features')
            if features is None:
                shape = (len(rows), 64, tokens.shape[-1])
                features = np.lib.format.open_memmap(feature_path.with_suffix('.partial.npy'),
                                                     mode='w+', dtype=np.float16, shape=shape)
            features[idx.numpy()] = tokens.float().cpu().numpy()
            atomic_json(output / 'feature_status.json', dict(state='running', encoder=args.encoder,
                        done=int(idx[-1])+1, total=len(rows), elapsed_seconds=time.monotonic()-started))
            if int(idx[0]) % (args.batch_size*20) == 0:
                print(f'{args.encoder} features {int(idx[-1])+1}/{len(rows)} elapsed={time.monotonic()-started:.1f}s', flush=True)
    if features is None:
        raise ValueError('Empty image cohort')
    features.flush()
    del features
    os.replace(feature_path.with_suffix('.partial.npy'), feature_path)
    atomic_json(metadata, dict(**signature, checkpoint=str(checkpoint), shape=shape, dtype='float16',
                pretrained=True, strict_load=True, frozen=True, source_only=True,
                feature_sha256=digest(feature_path), pooling='native spatial grid to 8x8 adaptive average',
                preprocessing='official inference transform; see pinned encoder module',
                elapsed_seconds=time.monotonic()-started,
                adapted_interface='Frozen public visual encoder, matched Qwen fusion/predictor/readouts',
                partial=bool(args.limit)))
    atomic_json(output / 'feature_status.json', dict(state='complete', encoder=args.encoder,
                done=len(rows), total=len(rows), elapsed_seconds=time.monotonic()-started))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--encoder', choices=['biovil_t', 'chexworld'], required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--out')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, default=0)
    extract(p.parse_args())
