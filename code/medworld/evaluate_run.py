"""Config-controlled held-out tests using each run's frozen source."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
from .config import PROJECT, load_config
from .launch_distributed import atomic
from .run_experiment import evaluation_jobs, registry, schedule


def completed_job(run, job, checkpoint_hash, fingerprint):
    directory = run / 'evaluation' / job['id']
    summary_path = directory / 'summary.json'
    if not summary_path.exists():
        return False
    summary = json.loads(summary_path.read_text())
    args = job['args']
    task, split = args[args.index('--task') + 1], args[args.index('--split') + 1]
    predictions = directory / f'{task}.jsonl'
    if (summary.get('checkpoint_sha256') != checkpoint_hash or summary.get('data_fingerprint') != fingerprint
            or summary.get('limit') is not None or summary.get('split') != split or task not in summary.get('tasks', {})):
        raise ValueError(f'{job["id"]}: existing results use a different checkpoint/protocol')
    if not predictions.exists():
        raise ValueError(f'{job["id"]}: predictions are missing')
    from .datasets.protocol import _sha256
    if _sha256(predictions) != summary.get('predictions_sha256', {}).get(task):
        raise ValueError(f'{job["id"]}: prediction contents changed or lack integrity metadata')
    with predictions.open() as handle:
        count = sum(1 for line in handle if line.strip())
    if count <= 0 or count != summary['tasks'][task]['n']:
        raise ValueError(f'{job["id"]}: incomplete predictions')
    return True


def evaluate_run(run, gpus):
    run = Path(run).resolve()
    cfg = load_config(run / 'config.json')
    jobs = evaluation_jobs(run, cfg)
    if not jobs:
        atomic(run / 'pipeline_status.json', {'phase': 'training_complete', 'evaluation_complete': False,
               'evaluation_skipped': True, 'reason': 'testing.enabled=false', 'heartbeat_unix': time.time()})
        return
    state = json.loads((run / 'status.json').read_text())
    if not state.get('complete') or state.get('stopped') or not (run / 'final.pt').is_file():
        raise ValueError('Tests require successfully completed training and final.pt')
    source = run / 'source'
    for name, digest in json.loads((run / 'source_manifest.json').read_text()).items():
        if hashlib.sha256((source / 'medworld' / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen source was modified: {name}')
    from .datasets.protocol import _sha256
    fingerprint = hashlib.sha256(json.dumps(json.loads((run / 'data_protocol.json').read_text()), sort_keys=True).encode()).hexdigest()
    checkpoint_hash = _sha256(run / 'final.pt')
    pending, reused = [], []
    for job in jobs:
        directory = run / 'evaluation' / job['id']
        if cfg['testing']['reuse_completed'] and completed_job(run, job, checkpoint_hash, fingerprint):
            reused.append(job['id'])
        else:
            if directory.exists() and any(directory.iterdir()):
                raise ValueError(f'{job["id"]}: existing partial results; preserve them before retrying')
            pending.append(job)
    atomic(run / 'evaluation_plan.json', {'jobs': jobs, 'gpus': gpus, 'testing': cfg['testing'], 'reused': reused,
           'test_selection': 'final.pt; no test-based checkpoint selection'})
    env = dict(os.environ, MEDWORLD_PROJECT_ROOT=str(PROJECT), PYTHONPATH=str(source),
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1',
               PYTORCH_ALLOC_CONF='expandable_segments:True')
    registry(run, 'evaluating')
    outcome = 'Failed during configured evaluation; checkpoint preserved.'
    try:
        schedule(pending, gpus, run, env)
        summary = {j['id']: json.loads((run / 'evaluation' / j['id'] / 'summary.json').read_text()) for j in jobs}
        atomic(run / 'evaluation_summary.json', summary)
        atomic(run / 'pipeline_status.json', {'phase': 'complete', 'training_complete': True,
               'evaluation_complete': True, 'tested_jobs': [j['id'] for j in jobs], 'reused_jobs': reused,
               'summary': 'evaluation_summary.json', 'heartbeat_unix': time.time()})
        outcome = 'Completed configured tests: ' + ', '.join(j['id'] for j in jobs) + '.'
    except BaseException as error:
        atomic(run / 'pipeline_status.json', {'phase': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
               'training_complete': True, 'evaluation_complete': False, 'error': repr(error), 'heartbeat_unix': time.time()})
        raise
    finally:
        registry(run, 'finished', outcome)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--gpus', default='1,2,3,6')
    a = p.parse_args()
    gpus = a.gpus.split(',')
    if not all(gpus) or len(set(gpus)) != len(gpus):
        p.error('Select distinct GPUs')
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    evaluate_run(a.run, gpus)


if __name__ == '__main__':
    main()
