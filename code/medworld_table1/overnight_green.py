"""Adapt locally saved forecast reports to the existing official GREEN runner."""
import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from common import atomic_json, load_rows, read_config, write_rows


def main(args):
    root = args.run.resolve()
    cfg = read_config(root / 'configs/slots.json')
    bridge = root / 'green'
    (bridge / 'cohort').mkdir(parents=True, exist_ok=True)
    obs = {r['id']: r for r in load_rows(Path(cfg['cache']) / 'observations.jsonl')}
    rows = load_rows(Path(cfg['cache']) / 'test.jsonl')
    expected = [r['id'] for r in rows]
    write_rows(bridge / 'cohort/table1_references_test.jsonl', [dict(id=r['id'], target_report=obs[r['target']]['report']) for r in rows])
    models = []
    for condition in ('slots', 'no_slots', 'shuffled'):
        (bridge / condition / 'test').mkdir(parents=True, exist_ok=True)
        predicted = load_rows(root / condition / 'evaluation_test/predictions.jsonl')
        if [r['id'] for r in predicted] != expected:
            raise ValueError('GREEN bridge prediction cohort mismatch')
        write_rows(bridge / condition / 'test/responses.jsonl', [dict(key='table1_report|'+r['id']+'|', ok=True, text=r['report']) for r in predicted])
        atomic_json(bridge / condition / 'inference_finished.json', dict(status='complete'))
        models.append(dict(id=condition))
    atomic_json(bridge / 'models.json', models)
    baseline = Path(__file__).resolve().parent.parent / 'medworld_baselines'
    sys.path.insert(0, str(baseline))
    import green_eval
    green_eval.main(SimpleNamespace(run=bridge, gpu=os.environ['CUDA_VISIBLE_DEVICES'], prepare=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args())
