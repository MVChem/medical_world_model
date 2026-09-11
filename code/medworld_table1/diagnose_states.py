"""Held-out state/readout interventions; oracle futures are diagnostics only."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

from common import ROOT, atomic_json, digest, load_rows, seed_all, write_rows
import numpy as np
import torch


def state_statistics(states):
    x = states.float().flatten(1)
    centered = x - x.mean(0)
    singular = torch.linalg.svdvals(centered)
    p = singular.square() / singular.square().sum().clamp_min(1e-12)
    rank = torch.exp(-(p*p.clamp_min(1e-12).log()).sum())
    normalized = torch.nn.functional.normalize(x, dim=-1)
    cosines = normalized @ normalized.T
    offdiag = ~torch.eye(len(x), dtype=torch.bool, device=x.device)
    return dict(centered_rms=float(centered.square().mean().sqrt()),
        effective_rank=float(rank), max_rank=len(x)-1,
        mean_pairwise_cosine=float(cosines[offdiag].mean()))


def generate(a):
    from data import Corpus
    from model import MedWorld, masked_bce
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cfg = checkpoint['config']
    seed_all(cfg['seed'])
    for name in ('observations.jsonl', 'train.jsonl', 'validate.jsonl', 'test.jsonl', 'features.json'):
        assert digest(Path(cfg['cache'])/name) == checkpoint['signatures'][name]
    model = MedWorld(cfg)
    if checkpoint['stage'] == 2:
        model.begin_stage2(frozen_encoder=checkpoint.get('mode') == 'matched')
    model.load_compact(checkpoint['model'])
    model.eval().cuda()
    corpus = Corpus(cfg, model.tokenizer)
    rows, patients = [], set()
    for r in corpus.pairs['validate']:
        if r['patient'] not in patients:
            rows.append(r)
            patients.add(r['patient'])
        if len(rows) >= a.limit:
            break
    assert len(rows) > 1
    states = {'current_image': [], 'current_multimodal': [], 'future_online': []}
    if checkpoint['stage'] == 2:
        states.update(forecast=[], future_fixed=[])
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, len(rows), cfg['batch_size']):
            batch = corpus.batch(rows[i:i+cfg['batch_size']])
            current = model.state(batch)
            states['current_multimodal'].append(current)
            states['current_image'].append(model.state(batch, image_only=True))
            states['future_online'].append(model.state(batch, 'target'))
            if checkpoint['stage'] == 2:
                forecast = model.world(current, batch['horizon'])
                states['forecast'].append(forecast)
                states['future_fixed'].append(model.state(batch, 'target', target=True))
                # Causal forecast tensors must be exactly invariant to changes
                # in every future modality and label.
                for key in list(batch):
                    if key.startswith('target_'):
                        batch[key] = torch.zeros_like(batch[key])
                altered = model.world(model.state(batch), batch['horizon'])
                torch.testing.assert_close(forecast, altered, rtol=0, atol=0)
        states = {k: torch.cat(v) for k, v in states.items()}
        base = 'forecast' if checkpoint['stage'] == 2 else 'current_multimodal'
        states['shuffled_'+base] = states[base].roll(1, 0)
        states['mean_'+base] = states[base].mean(0, keepdim=True).expand_as(states[base])
        metrics, outputs = {}, []
        for variant, state in states.items():
            current_ref = variant.startswith('current') or 'current_multimodal' in variant
            side = 'source' if current_ref else 'target'
            generated, scores, ces, bces = [], [], [], []
            for i in range(0, len(rows), cfg['batch_size']):
                selected = rows[i:i+cfg['batch_size']]
                batch = corpus.batch(selected)
                s = state[i:i+len(selected)]
                generated.extend(model.decoder.generate(s, cfg['generation_tokens']))
                scores.extend(model.scores(s).sigmoid().float().cpu().tolist())
                ces.append((float(model.decoder.loss(s, batch[side+'_target_ids'], batch[side+'_target_mask'])), len(selected)))
                bces.append((float(masked_bce(model.scores(s), batch[side+'_labels'])), len(selected)))
            counts = Counter(generated)
            metrics[variant] = dict(reference=side, n=len(rows), unique_reports=len(counts),
                most_common_count=max(counts.values()),
                text_ce_batch_weighted=sum(v*n for v,n in ces)/len(rows),
                finding_bce_batch_weighted=sum(v*n for v,n in bces)/len(rows),
                state=state_statistics(state))
            for r, report, score in zip(rows, generated, scores):
                reference = corpus.observations[corpus.lookup[r[side]]]
                source = corpus.observations[corpus.lookup[r['source']]]
                outputs.append(dict(id=r['id'], variant=variant, report=report, scores=score,
                    reference=reference['report'], current_report=source['report']))
            atomic_json(a.out/'summary.json', dict(checkpoint=str(a.checkpoint),
                checkpoint_sha256=digest(a.checkpoint), stage=checkpoint['stage'],
                stage_step=checkpoint['stage_step'], findings=cfg['findings'], split='validate',
                patient_disjoint_sample=True, variants=metrics,
                future_input_invariance='passed' if checkpoint['stage']==2 else 'not_applicable',
                purpose='Readout diagnostics. True future states and cohort-mean intervention are NOT forecast methods.'))
            write_rows(a.out/'predictions.jsonl', outputs)
            print(variant, metrics[variant], flush=True)


def score(a):
    # Load metric-only Transformers after common and before clinical.
    sys.path.insert(0, str(ROOT/'metric_vendor'))
    from clinical import CheXbert
    from metrics import clinical_metrics
    summary = json.loads((a.out/'summary.json').read_text())
    rows = load_rows(a.out/'predictions.jsonl')
    extractor = CheXbert()
    for variant, result in summary['variants'].items():
        selected = [r for r in rows if r['variant'] == variant]
        current = extractor.labels([r['current_report'] for r in selected], summary['findings'])
        target = extractor.labels([r['reference'] for r in selected], summary['findings'])
        predicted = extractor.labels([r['report'] for r in selected], summary['findings'])
        result['clinical'] = clinical_metrics(current, target, predicted,
            [r['scores'] for r in selected], summary['findings'])
    summary['chexbert'] = extractor.provenance
    atomic_json(a.out/'summary.json', summary)
    lines = ['# State/readout diagnostic (validation only)', '',
        '| State input | Reference | Unique reports | Largest repeat | Report CE | Finding AP | Report CheXbert F1 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for name, m in summary['variants'].items():
        def f(x): return '—' if x is None else f'{x:.4f}'
        lines.append(f"| {name} | {m['reference']} | {m['unique_reports']}/{m['n']} | {m['most_common_count']} | {m['text_ce_batch_weighted']:.4f} | {f(m['clinical']['finding_auprc'])} | {f(m['clinical']['chexbert_f1'])} |")
    lines += ['', summary['purpose'], '', 'One pair per validation patient. Same decoder/head within a checkpoint; no comparison across different references without qualification.']
    (a.out/'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    os.umask(0o077)
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit', type=int, default=24)
    p.add_argument('--score-only', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.score_only:
        score(a)
    else:
        generate(a)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--score-only'], check=True)
