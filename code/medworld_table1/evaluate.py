"""Greedy held-out generation and official clinical scoring; never fabricate unavailable scores."""
import argparse
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from common import ROOT, atomic_json, digest, load_rows, read_config, seed_all, write_rows
import numpy as np
import torch
from clinical import CheXbert
from data import Corpus
from metrics import candidate_pools, clinical_metrics, retrieval


def predict(args, cfg):
    from transformers import AutoTokenizer
    if args.mode == 'qwen9b':
        raise ValueError('Use evaluate_qwen9b.py for source-only local server generation, then --score-only.')
    seed_all(cfg['seed'])
    if args.mode == 'copy':
        tokenizer = AutoTokenizer.from_pretrained(cfg['qwen'], local_files_only=True)
        model = None
    else:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if checkpoint['stage'] != 2:
            raise ValueError('Stage-1 checkpoint alone is not a trained forecast.')
        for name, expected in checkpoint['signatures'].items():
            if name == 'qwen_config':
                path = Path(cfg['qwen']) / 'config.json'
            elif name == 'qwen_weights':
                path = Path(cfg['qwen']) / 'model.safetensors-00001-of-00001.safetensors'
            else:
                path = Path(cfg['cache']) / name
            if digest(path) != expected:
                raise ValueError(f'Checkpoint input differs: {name}')
        if args.mode == 'direct':
            from direct import DirectQwen
            model = DirectQwen(cfg)
        else:
            from model import MedWorld
            model = MedWorld(cfg)
            model.begin_stage2(frozen_encoder=checkpoint.get('mode') == 'matched')
        model.load_compact(checkpoint['model'])
        model.eval().to('cuda')
        tokenizer = model.tokenizer
    corpus = Corpus(cfg, tokenizer)
    rows = corpus.pairs[args.split][:args.limit or None]
    generated = []
    for i in range(0, len(rows), cfg['batch_size']):
        selected = rows[i:i+cfg['batch_size']]
        if model is None:
            reports = tokenizer.batch_decode([corpus.tokens[corpus.lookup[r['source']]] for r in selected], skip_special_tokens=True)
            scores = [None] * len(selected)
        else:
            batch = corpus.batch(selected)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                reports, scores = model.predict(batch)
            scores = scores.float().cpu().tolist()
        generated.extend(dict(id=r['id'], report=text, scores=score) for r, text, score in zip(selected, reports, scores))
        if i % 50 == 0:
            print(f'generated {i+len(selected)}/{len(rows)}', flush=True)
            write_rows(Path(args.out) / 'predictions.partial.jsonl', generated)
    write_rows(Path(args.out) / 'predictions.jsonl', generated)
    atomic_json(Path(args.out) / 'generation.json', dict(mode=args.mode, split=args.split, count=len(rows),
        checkpoint=str(args.checkpoint), checkpoint_sha256=digest(args.checkpoint) if args.checkpoint else None,
        decoding='greedy', max_new_tokens=cfg['generation_tokens'], teacher_forcing=False, target_inputs=False))
    counts = Counter(r['report'] for r in generated)
    atomic_json(Path(args.out) / 'diversity.json', dict(n=len(generated), unique_reports=len(counts),
        most_common_count=max(counts.values(), default=0),
        top5_fraction=sum(v for _, v in counts.most_common(5))/max(1, len(generated)),
        empty_reports=sum(not r['report'].strip() for r in generated)))
    del model
    torch.cuda.empty_cache()


def score(args, cfg):
    output = Path(args.out)
    generated = load_rows(output / 'predictions.jsonl')
    rows = load_rows(Path(cfg['cache']) / f'{args.split}.jsonl')
    obs = {r['id']: r for r in load_rows(Path(cfg['cache']) / 'observations.jsonl')}
    expected = rows[:args.limit or None]
    if [r['id'] for r in expected] != [r['id'] for r in generated]:
        raise ValueError('Prediction IDs/order differ from evaluation cohort')
    rows = expected
    current_reports = [obs[r['source']]['report'] for r in rows]
    target_reports = [obs[r['target']]['report'] for r in rows]
    predicted_reports = [r['report'] for r in generated]
    extractor = CheXbert()
    current = extractor.labels(current_reports, cfg['findings'])
    target = extractor.labels(target_reports, cfg['findings'])
    predicted = extractor.labels(predicted_reports, cfg['findings'])
    provenance = extractor.provenance
    del extractor
    torch.cuda.empty_cache()
    scores = [r['scores'] for r in generated] if all(r['scores'] is not None for r in generated) else None
    result = clinical_metrics(current, target, predicted, scores, cfg['findings'])
    negatives = cfg.get('retrieval_negatives', 31)
    pools = candidate_pools(rows, current, target, cfg['seed'], negatives=negatives)
    result.update(retrieval(predicted, target, pools))
    atomic_json(output / 'candidate_pools.json', pools)
    atomic_json(output / 'chexbert_labels.json', dict(current=current, target=target, predicted=predicted, provenance=provenance))
    result.update(mode=args.mode, split=args.split, n=len(rows), patients=len({r['patient'] for r in rows}),
        radgraph_f1=None, radgraph_status='pending', chexbert=provenance,
        protocol=dict(unknown='Reference blank/uncertain excluded; prediction unknown never alters coverage',
                      zero_support='Truth-unsupported diseases/events marked null and excluded from a fixed reference-defined denominator',
                      auprc='non-interpolated macro Average Precision, genuine model disease scores; training CheXpert, eval CheXbert',
                      retrieval=f'{negatives+1} candidates; other patients; matched horizon, view, coarse current-positive count, exact reference coverage mask; fractional ties',
                      transition='Per-finding onset/resolution F1 on all truth-evaluable pairs, including stable cases'))
    # Stable/changed diagnostics use reference labels only.
    c, t, p = map(np.asarray, (current, target, predicted))
    known = ((c == 0) | (c == 1)) & ((t == 0) | (t == 1))
    changed = ((c != t) & known).any(1)
    groups = {}
    for name, mask in [('changed', changed), ('stable_observed', ~changed & known.any(1))] + [(f'horizon_{h}', np.array([r['horizon'] == h for r in rows])) for h in range(4)]:
        sub = clinical_metrics(c[mask], t[mask], p[mask], np.asarray(scores)[mask] if scores is not None else None, cfg['findings'])
        sub['n'] = int(mask.sum())
        groups[name] = sub
    result['groups'] = groups
    atomic_json(output / 'metrics.json', result)
    if not args.skip_radgraph:
        try:
            from radgraph import F1RadGraph
            scorer = F1RadGraph(reward_level='all', model_type='radgraph-xl', cuda=0,
                               model_cache_dir=str(ROOT / 'weights/radgraph'))
            reward, _, _, _ = scorer(hyps=predicted_reports, refs=target_reports)
            result['radgraph_f1'] = float(reward[1])
            result['radgraph_status'] = 'radgraph 0.1.18, radgraph-xl, RG_ER/partial, report-mean'
        except Exception as exc:
            result['radgraph_status'] = f'unavailable: {type(exc).__name__}: {exc}'
    atomic_json(output / 'metrics.json', result)
    header = '| Method | Future R@1 | Finding AUPRC | Transition F1 | RadGraph F1 | CheXbert F1 |\n|---|---:|---:|---:|---:|---:|\n'
    keys = ['future_r1', 'finding_auprc', 'transition_f1', 'radgraph_f1', 'chexbert_f1']
    values = ['—' if result[k] is None else f'{result[k]:.4f}' for k in keys]
    (output / 'table1_row.md').write_text(header + '| ' + args.mode + ' | ' + ' | '.join(values) + ' |\n\nPilot only; see metrics.json for coverage, support and unavailable metrics.\n')
    print({k: result[k] for k in keys}, flush=True)


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/pilot.json')
    parser.add_argument('--mode', choices=['ours', 'matched', 'direct', 'copy', 'qwen9b'], required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--split', choices=['validate', 'test'], default='test')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--out', required=True)
    parser.add_argument('--score-only', action='store_true')
    parser.add_argument('--skip-radgraph', action='store_true')
    args = parser.parse_args()
    cfg = read_config(args.config)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if not args.score_only:
        predict(args, cfg)
    if os.environ.get('MEDWORLD_METRIC_WORKER') == '1':
        score(args, cfg)
    else:
        # RadGraph 0.1.18 uses the Transformers 4 tokenizer API. Run clinical
        # extraction in its isolated dependency path after Qwen generation.
        subprocess.run([sys.executable, str(ROOT / 'score.py'), *sys.argv[1:], '--score-only'], check=True)
