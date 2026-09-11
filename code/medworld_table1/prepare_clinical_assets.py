"""Finish local scoring prerequisites and verify the actual clinical pipelines."""
import json
import os
from pathlib import Path
import sys
import tarfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / 'metric_vendor'))
from common import ROOT, atomic_json


def prepare():
    os.umask(0o077)
    out = ROOT / 'weights/radgraph/radgraph-xl'
    archive = ROOT / 'weights/radgraph-xl.tar.gz'
    checkpoint = ROOT / 'weights/chexbert.pth'
    deadline = time.monotonic() + 4*3600
    while not archive.exists() or not checkpoint.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('Clinical weight downloads did not finish in 4h')
        time.sleep(10)
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(out, filter='data')
    config = json.loads((out / 'config.json').read_text())
    names = {config['dataset_reader']['token_indexers']['bert']['model_name'],
             config['model']['embedder']['token_embedders']['bert']['model_name']}
    from huggingface_hub import snapshot_download
    paths = {}
    for name in names:
        print('preparing RadGraph text backbone', name, flush=True)
        local = ROOT / 'weights' / name.replace('/', '__')
        if (local / 'config.json').exists() and (local / 'vocab.txt').exists():
            paths[name] = str(local)
            continue
        paths[name] = snapshot_download(name, local_dir=local,
            # RadGraph initializes this backbone from config and then strictly
            # restores its own checkpoint, which already contains all weights.
            allow_patterns=['*.json', '*.txt', '*.model'])
    config['dataset_reader']['token_indexers']['bert']['model_name'] = paths[config['dataset_reader']['token_indexers']['bert']['model_name']]
    config['model']['embedder']['token_embedders']['bert']['model_name'] = paths[config['model']['embedder']['token_embedders']['bert']['model_name']]
    atomic_json(out / 'config.json', config)
    from clinical import CheXbert
    from radgraph import F1RadGraph
    # Synthetic non-patient reports only; tests run locally.
    reports = ['No pleural effusion or pneumothorax.', 'Moderate right pleural effusion.']
    chex = CheXbert()
    labels = chex.labels(reports, ['Pleural Effusion', 'Pneumothorax'])
    assert labels[0] == [0, 0] and labels[1][0] == 1, labels
    scorer = F1RadGraph(reward_level='all', model_type='radgraph-xl', cuda=0,
                       model_cache_dir=str(ROOT / 'weights/radgraph'))
    reward, _, _, _ = scorer(hyps=reports, refs=reports)
    assert float(reward[1]) > .99, reward
    atomic_json(ROOT / 'weights/clinical_ready.json', dict(chexbert_labels=labels, radgraph_identity_f1=float(reward[1]),
                                                         chexbert=chex.provenance, radgraph_text_paths=paths))
    print('Clinical scoring smoke test passed.', flush=True)


if __name__ == '__main__':
    try:
        prepare()
    except Exception as exc:
        atomic_json(ROOT / 'weights/clinical_failed.json', dict(error=f'{type(exc).__name__}: {exc}'))
        raise
