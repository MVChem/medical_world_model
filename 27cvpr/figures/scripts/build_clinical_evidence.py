#!/usr/bin/env python3
"""Real input-space blur occlusion for existing image-only classification baselines.

This is a baseline-only candidate, not a visualization of the proposed slots.
No disease localization annotations are available for this selected case.
Rebuild: /home/data2/chk/workspace/2026/.venv/bin/python <this file>
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / '27cvpr/figures/generated'
RUN = ROOT / 'code/medworld_open_baselines/runs/comparators_20260913/dense_4096'
sys.path.insert(0, str(ROOT / 'code/medworld_open_baselines'))
from dinov2_features import Extractor, CheXWorldExtractor
from dinov2_train import classification_components

FINDINGS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
            'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
MODELS = [('dinov2_vitb14', 'DINOv2-B/14'), ('chexworld', 'CheXWorld')]
SEED = 20260914


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def select_case():
    source = RUN / MODELS[0][0]
    data = json.loads((source / 'classification_data.json').read_text())
    records = rows(source / 'classification_observations.jsonl')
    other = rows(RUN / MODELS[1][0] / 'classification_observations.jsonl')
    assert records == other, 'baseline cohorts differ'
    finding_ids = [1, 8]
    eligible = [r for r in records if r['split'] == 'test' and all(r['labels'][j] == 1 for j in finding_ids)]
    case = min(eligible, key=lambda r: hashlib.sha256(f'{SEED}:{r["id"]}'.encode()).hexdigest())
    images = np.load(Path(data['source']) / 'images.npy', mmap_mode='r')
    pixels = np.array(images[case['image_index']], copy=True)
    return case, finding_ids, Image.fromarray(pixels).convert('RGB'), data, len(eligible)


def occluded(image, blurred, regions):
    result = image.copy()
    for box in regions:
        result.paste(blurred.crop(box), box[:2])
    return result


def infer(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)
    case, finding_ids, image, data, count = select_case()
    width, height = image.size
    grid = 8
    xs = np.linspace(0, width, grid + 1, dtype=int)
    ys = np.linspace(0, height, grid + 1, dtype=int)
    boxes = [(int(xs[x]), int(ys[y]), int(xs[x + 1]), int(ys[y + 1])) for y in range(grid) for x in range(grid)]
    blurred = image.filter(ImageFilter.GaussianBlur(radius=24))
    perturbations = [image] + [occluded(image, blurred, [b]) for b in boxes]
    result = dict(scope='Baseline-only image-space class-conditioned attribution; proposed slot model pending',
                  attribution='Signed original logit minus logit after local Gaussian blur; NOT attention or grounding',
                  localization_reference='Unavailable; official image-level labels only; no localization accuracy claim',
                  input='Image only, no current report, no clinical history',
                  selection=dict(rule='Minimum SHA256(seed:image_id) among test images positive for both target findings',
                                 seed=SEED, eligible=count, independent_of_predictions=True),
                  case=case, findings=[FINDINGS[j] for j in finding_ids], data=data,
                  image_pixel_sha256=hashlib.sha256(np.asarray(image).tobytes()).hexdigest(),
                  grid=[grid, grid], blur_radius_pixels=24, boxes_xyxy=boxes,
                  control='Top 4 positive-drop grid cells vs 16 sets of 4 uniformly sampled grid cells; equal area',
                  control_note='Exploratory same-image check: equal input canvas area, not matched retained pixels or anatomy; native crop differs; not held-out evidence validation', models={})
    cls, _, _ = classification_components()
    for mid, name in MODELS:
        started = time.monotonic()
        model_dir = RUN / mid
        cp = model_dir / 'classification_spatial/checkpoint.pt'
        saved = torch.load(cp, map_location='cpu', weights_only=False)
        assert saved['epoch'] == saved['contract']['epochs'] == 20
        head = cls(1024).eval().to(args.device)
        head.load_state_dict(saved['model'], strict=True)
        del saved
        encoder = Extractor(device=args.device) if mid == 'dinov2_vitb14' else CheXWorldExtractor(device=args.device)

        @torch.inference_mode()
        def score(batch):
            features = torch.from_numpy(encoder.batch(batch)).to(args.device).float()
            with torch.autocast(torch.device(args.device).type, dtype=torch.bfloat16,
                                enabled=torch.device(args.device).type == 'cuda'):
                return head(F.pad(features, (0, 256))).float().cpu().numpy()

        logits = []
        for start in range(0, len(perturbations), args.batch_size):
            logits.extend(score(perturbations[start:start + args.batch_size]))
            print(name, 'occlusions', min(start + args.batch_size, len(perturbations)), '/', len(perturbations), flush=True)
        logits = np.asarray(logits)
        assert np.isfinite(logits).all()
        drops = logits[0, finding_ids][None] - logits[1:, finding_ids]
        cached = np.load(model_dir / 'classification_features.npy', mmap_mode='r')[case['index']]
        with torch.inference_mode():
            cached_logits = head(F.pad(torch.tensor(np.array(cached))[None].to(args.device).float(), (0, 256))).float().cpu().numpy()[0]
        stored = next(r for r in rows(model_dir / 'classification_spatial/test_per_sample.jsonl') if r['id'] == case['id'])
        controls = []
        for j, target in enumerate(finding_ids):
            top = np.argsort(-drops[:, j], kind='stable')[:4].tolist()
            rng = np.random.default_rng(SEED + j)
            random_sets = [rng.choice(64, 4, replace=False).tolist() for _ in range(16)]
            sets = [top] + random_sets
            ims = [occluded(image, blurred, [boxes[k] for k in ids]) for ids in sets]
            scores = []
            for start in range(0, len(ims), args.batch_size):
                scores.extend(score(ims[start:start + args.batch_size])[:, target])
            d = logits[0, target] - np.asarray(scores)
            controls.append(dict(finding=FINDINGS[target], top_cells=top, random_cells=random_sets,
                                 top_logit_drop=float(d[0]), random_logit_drops=d[1:].tolist(),
                                 random_mean=float(d[1:].mean()), random_sd=float(d[1:].std())))
        result['models'][mid] = dict(name=name, checkpoint=str(cp.relative_to(ROOT)), checkpoint_sha256=sha(cp),
            contract_sha256=sha(model_dir / 'classification_spatial/contract.json'),
            encoder=encoder.metadata, logits=logits[:, finding_ids].tolist(), logit_drops=drops.T.reshape(2,8,8).tolist(),
            probabilities=(1 / (1 + np.exp(-logits[0, finding_ids]))).tolist(),
            cached_probabilities=(1 / (1 + np.exp(-cached_logits[finding_ids]))).tolist(),
            saved_bf16_probabilities=[stored['scores'][j] for j in finding_ids],
            recomputed_vs_cached_logit_max_abs=float(np.max(np.abs(logits[0] - cached_logits))),
            controls=controls, inference_device=args.device, seconds=time.monotonic()-started)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'clinical_evidence.provenance.json').write_text(json.dumps(result, indent=2) + '\n')
        print('completed', name, result['models'][mid]['seconds'], flush=True)
        del encoder, head
        gc.collect()
    return result, image


def draw(result, image):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.cm import ScalarMappable
    result['script'] = str(Path(__file__).relative_to(ROOT))
    result['script_sha256'] = sha(__file__)
    result['display'] = 'Shared signed logit scale; cell opacity proportional to absolute drop; no smoothing'
    result['control_note'] = 'Exploratory same-image check: equal input canvas area, not matched retained pixels or anatomy; native crop differs; not held-out evidence validation'
    result['probability_note'] = 'Displayed probabilities are CPU FP32 re-encodings; saved BF16 probabilities are retained separately for comparison'
    (OUT / 'clinical_evidence.provenance.json').write_text(json.dumps(result, indent=2) + '\n')
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'pdf.fonttype':42, 'svg.fonttype':'none'})
    fig = plt.figure(figsize=(14, 8.3), facecolor='white')
    gs = fig.add_gridspec(2, 4, left=.025, right=.985, top=.845, bottom=.135,
                          width_ratios=[1,1,1,1.04], wspace=.12, hspace=.29)
    fig.text(.025,.965,'Clinical evidence: a first attribution check',fontsize=19,weight='bold',color='#172d45')
    fig.text(.025,.925,'Existing image-only classifiers  |  Same held-out CXR, two findings  |  Final 20-epoch heads',fontsize=11,color='#465566')
    vmax = max(abs(np.asarray(m['logit_drops'])).max() for m in result['models'].values())
    vmax = max(float(vmax), 1e-5)
    norm = TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax)
    titles = ['Input + image-level reference','DINOv2-B/14','CheXWorld','Equal input-area blur']
    pixels = np.asarray(image.convert('L'))
    for row, finding in enumerate(result['findings']):
        ax = fig.add_subplot(gs[row,0])
        ax.imshow(pixels,cmap='gray',vmin=0,vmax=255);ax.axis('off')
        ax.set_title(titles[0] if row==0 else '',fontsize=11,pad=10,weight='bold')
        ax.text(.02,.03,f'{finding}: positive',transform=ax.transAxes,color='white',fontsize=11,
                bbox=dict(facecolor='#172d45',alpha=.87,edgecolor='none',pad=5))
        for col,(mid,_) in enumerate(MODELS,1):
            model = result['models'][mid]
            ax=fig.add_subplot(gs[row,col])
            ax.imshow(pixels,cmap='gray',vmin=0,vmax=255)
            heat=np.asarray(model['logit_drops'][row])
            ax.imshow(heat,cmap='RdBu_r',norm=norm,alpha=.75*np.clip(np.abs(heat)/vmax,0,1),interpolation='nearest',
                      extent=(-.5,pixels.shape[1]-.5,pixels.shape[0]-.5,-.5))
            ax.axis('off');ax.set_title(titles[col] if row==0 else '',fontsize=11,pad=10,weight='bold')
            ax.text(.02,.03,f'p = {model["probabilities"][row]:.3f}',transform=ax.transAxes,color='white',fontsize=11,
                    bbox=dict(facecolor='#172d45',alpha=.87,edgecolor='none',pad=5))
        ax=fig.add_subplot(gs[row,3])
        if row==0:ax.set_title(titles[3],fontsize=11,pad=10,weight='bold')
        xpos=np.arange(2)
        high=[result['models'][m]['controls'][row]['top_logit_drop'] for m,_ in MODELS]
        rnd=[result['models'][m]['controls'][row]['random_mean'] for m,_ in MODELS]
        sd=[result['models'][m]['controls'][row]['random_sd'] for m,_ in MODELS]
        ax.bar(xpos-.17,high,.32,color='#b34d41',label='Top-response cells')
        ax.bar(xpos+.17,rnd,.32,color='#8695a4',yerr=sd,capsize=3,label='Random cells')
        ax.axhline(0,color='#4a5662',lw=.8);ax.set_xticks(xpos,['DINOv2','CheXWorld'],fontsize=10)
        ax.set_ylabel('Logit drop',fontsize=10);ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
        if row==0:ax.legend(loc='best',fontsize=8,frameon=False)
    cax=fig.add_axes([.10,.083,.48,.014])
    cb=fig.colorbar(ScalarMappable(norm=norm,cmap='RdBu_r'),cax=cax,orientation='horizontal')
    cb.set_label('Signed logit drop after local blur (shared scale; 8 × 8 cells)',fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.text(.625,.057,'4 / 64 cells blurred per control; random mean ± SD (16 draws).',fontsize=8.7,color='#465566')
    fig.text(.625,.039,'Equal canvas area; retained image / anatomical area differs.',fontsize=8.7,color='#465566')
    fig.text(.025,.017,'Baseline pilot only. No proposed-slot result or lesion-region annotation is available; these maps are attribution, not grounding or attention.',fontsize=9,color='#465566')
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/f'clinical_evidence.{ext}',dpi=190,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--render-only',action='store_true')
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=2);args=p.parse_args()
    if args.render_only:
        result=json.loads((OUT/'clinical_evidence.provenance.json').read_text());_,_,im,_,_=select_case()
    else: result,im=infer(args)
    draw(result,im)
    print('wrote',OUT/'clinical_evidence.pdf',flush=True)
