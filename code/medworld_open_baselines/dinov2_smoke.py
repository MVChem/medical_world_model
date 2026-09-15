"""Short real-image, LR-isolation and save/load checks, under an external GPU lock."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from dinov2_features import Extractor, CheXWorldExtractor, DENSE, ROOT, atomic, branch_image
from dinov2_train import SpatialHead, classification_components, shared


def main(args):
    torch.set_num_threads(4)
    images = np.load(DENSE / 'runs/dense_20260912/data/images.npy', mmap_mode='r')
    lr = np.load(DENSE / 'runs/dense_20260912/data/lr_images.npy', mmap_mode='r')
    class NoHR:
        def __getitem__(self, key):
            raise AssertionError('SR extraction tried to read HR')
    branch_image('lr', 0, NoHR(), lr)
    device = torch.device(args.device)
    output = dict(device=str(device), checks=[])
    for name, factory in [('dinov2_vitb14', Extractor), ('chexworld', CheXWorldExtractor)]:
        started = time.time()
        encoder = factory(device=args.device)
        source = [branch_image('hr', 0, images, lr), branch_image('lr', 0, NoHR(), lr)]
        features = encoder.batch(source)
        assert features.shape == (2, 64, 768) and np.isfinite(features).all()
        assert not np.array_equal(features[0], features[1]), 'HR/LR features unexpectedly identical'
        del encoder
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        for task in ('segmentation', 'sr'):
            torch.manual_seed(20260913)
            head = SpatialHead(task).to(device)
            torch.manual_seed(20260913)
            reference = shared.FrozenSlotHead(task)
            assert all(torch.equal(p.cpu(), q) for p, q in zip(head.parameters(), reference.parameters()))
            del reference
            col = 1 if task == 'sr' else 0
            feature = F.pad(torch.from_numpy(features[col:col+1].copy()).to(device).float(), (0, 256))
            arr = lr[0] if task == 'sr' else images[0]
            image = torch.from_numpy(arr.copy()).to(device).float()[None, None] / 255
            if task == 'segmentation':
                image = F.interpolate(image, (256, 256), mode='area')
            target = torch.zeros((1, 1, 512, 512) if task == 'sr' else (1, 3, 256, 256), device=device)
            mask = torch.ones_like(target[:, :1])
            optim = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=.01)
            with shared.autocast(device):
                prediction = head(image, feature)
            loss = shared.objective(task, prediction, target, mask)
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())
            optim.step()
            head.eval()
            with torch.inference_mode(), shared.autocast(device):
                expected = head(image, feature).float()
            checkpoint = args.out / f'{name}_{task}.pt'
            torch.save(dict(model=head.state_dict(), optimizer=optim.state_dict()), checkpoint)
            restored = SpatialHead(task).to(device).eval()
            restored.load_state_dict(torch.load(checkpoint, weights_only=True)['model'])
            with torch.inference_mode(), shared.autocast(device):
                actual = restored(image, feature).float()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            output['checks'].append(dict(model=name, task=task, loss=float(loss.detach()),
                parameters=sum(p.numel() for p in head.parameters()), train_step=True, exact_reload=True))
            del restored, head, optim, image, feature, target, mask, prediction, expected, actual, loss
        output[name] = dict(seconds=time.time() - started, feature_shape=list(features.shape))
    head_cls, criterion, scorer = classification_components()
    head = head_cls(1024).to(device)
    feature = F.pad(torch.from_numpy(features.copy()).to(device).float(), (0, 256))
    labels = torch.zeros((2, 13), dtype=torch.long, device=device)
    labels[0, 0], labels[0, 1] = 1, -1
    loss = criterion(head(feature), labels, torch.ones(13, device=device))
    loss.backward()
    assert torch.isfinite(loss)
    output['classification'] = dict(loss=float(loss.detach()), masked_labels=True,
                                     parameters=sum(p.numel() for p in head.parameters()))
    output['sr_hr_access_guard'] = True
    if device.type == 'cuda':
        output['peak_gpu_memory_bytes'] = torch.cuda.max_memory_allocated()
    atomic(args.out / 'smoke.json', output)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', type=Path, default=ROOT / 'dinov2_assets/smoke')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    main(a)
