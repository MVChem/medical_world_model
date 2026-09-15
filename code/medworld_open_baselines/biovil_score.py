"""Isolated Transformers-4 clinical scorer for adapted representation rows.

This process must never import biovil_forecast/Qwen before the metric dependency
path is selected. Generation remains in the parent's Transformers-5 process.
"""
import argparse
import os
from pathlib import Path
import sys

TABLE1 = Path(__file__).resolve().parent.parent / 'medworld_table1'
sys.path.insert(0, str(TABLE1))
sys.path.insert(0, str(TABLE1 / 'metric_vendor'))
os.environ['MEDWORLD_METRIC_WORKER'] = '1'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config')
    p.add_argument('--out', required=True)
    p.add_argument('--mode', default='representation_adapted')
    p.add_argument('--split', choices=['validate', 'test'], default='test')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--skip-radgraph', action='store_true')
    p.add_argument('--import-smoke', action='store_true')
    args = p.parse_args()
    from common import atomic_json, read_config
    import transformers
    if transformers.__version__.split('.')[0] != '4':
        raise RuntimeError(f'Clinical scoring requires isolated Transformers 4, got {transformers.__version__}')
    from importlib.metadata import version
    radgraph_error = None
    try:
        from radgraph import F1RadGraph
        radgraph_version = version('radgraph')
    except Exception as exc:
        if args.import_smoke:
            raise
        radgraph_error = f'{type(exc).__name__}: {exc}'
        radgraph_version = None
    atomic_json(Path(args.out)/'metric_environment.json', dict(
        transformers=transformers.__version__, transformers_path=transformers.__file__,
        radgraph=radgraph_version, worker=str(Path(__file__).resolve()),
        isolated_process=True, radgraph_import_success=radgraph_error is None,
        radgraph_import_error=radgraph_error))
    if args.import_smoke:
        print(f'Isolated metric imports passed: Transformers {transformers.__version__}, RadGraph {version("radgraph")}', flush=True)
        return
    if not args.config:
        p.error('--config is required for scoring')
    from evaluate import score
    score(args, read_config(args.config))


if __name__ == '__main__':
    main()
