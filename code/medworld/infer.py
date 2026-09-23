"""Predict from an image and/or supplied report, with a question for VQA."""
import argparse
from pathlib import Path
from .gpu import acquire_gpu
from .downstream_tasks.registry import TASKS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--image')
    p.add_argument('--report', help='Optional raw report text')
    p.add_argument('--task', choices=TASKS, required=True)
    p.add_argument('--question')
    p.add_argument('--out', required=True)
    p.add_argument('--gpu', default='auto')
    a = p.parse_args()
    if not a.image and not a.report:
        p.error('Supply --image and/or --report')
    if a.task == 'segmentation' and not a.image:
        p.error('Segmentation requires --image')
    if a.task == 'vqa' and not a.question:
        p.error('VQA requires --question')
    if Path(a.out).exists():
        p.error('Output exists')
    lock, device = acquire_gpu(a.gpu)
    try:
        import torch
        from PIL import Image, ImageOps
        from .runtime import load_model, atomic_json
        model, _ = load_model(a.checkpoint, device)
        batch = {'questions': [a.question]}
        if a.image:
            with Image.open(a.image) as im:
                image = ImageOps.pad(im.convert('RGB'), (512, 512), method=Image.Resampling.BICUBIC, color='black')
            batch['images'] = [image]
        if a.report is not None:
            batch['reports'] = [a.report]
        prediction = model.predict(a.task, batch)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        if a.task == 'segmentation':
            torch.save({'mask_probabilities': prediction.sigmoid().cpu()}, a.out)
        else:
            atomic_json(a.out, {'task': a.task, 'prediction': prediction.cpu().tolist() if torch.is_tensor(prediction) else prediction})
    finally:
        if lock is not None:
            lock.close()


if __name__ == '__main__':
    main()
