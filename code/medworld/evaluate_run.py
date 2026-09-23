"""Config-controlled held-out tests using each run's frozen source."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
from .config import PROJECT, load_config
from .launch_distributed import atomic
from .run_experiment import evaluation_jobs, registry, schedule
from .evaluation.protocol import FUTURE_TASKS, TABLE1_METRICS, validate_metric_protocol


def completed_job(run, job, checkpoint_hash, fingerprint):
    directory = run / 'evaluation' / job['id']
    summary_path = directory / 'summary.json'
    if not summary_path.exists():
        return False
    summary = json.loads(summary_path.read_text())
    args = job['args']
    task, split = args[args.index('--task') + 1], args[args.index('--split') + 1]
    validate_metric_protocol(summary, 'table1' if task in FUTURE_TASKS else 'table2')
    if task == 'vqa':
        for flag, key, default in [('--vqa-per-type', 'per_type', 0), ('--vqa-seed', 'seed', 42)]:
            expected = int(args[args.index(flag) + 1]) if flag in args else default
            if summary.get('vqa_selection', {}).get(key, default) != expected:
                raise ValueError('VQA sampling protocol changed')
    predictions = directory / f'{task}.jsonl'
    if (summary.get('checkpoint_sha256') != checkpoint_hash or summary.get('data_fingerprint') != fingerprint
            or summary.get('limit') is not None or summary.get('partial')
            or summary.get('split') != split or task not in summary.get('tasks', {})):
        raise ValueError(f'{job["id"]}: existing results use a different checkpoint/protocol')
    if not predictions.exists():
        raise ValueError(f'{job["id"]}: predictions are missing')
    from .datasets.protocol import _sha256
    if _sha256(predictions) != summary.get('predictions_sha256', {}).get(task):
        raise ValueError(f'{job["id"]}: prediction contents changed or lack integrity metadata')
    with predictions.open() as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    count = len(records)
    if count <= 0 or count != summary['tasks'][task]['n']:
        raise ValueError(f'{job["id"]}: incomplete predictions')
    if task in FUTURE_TASKS:
        from .evaluation.future_metrics import SCHEMA, reference_fingerprint, validate_radgraph_provenance
        from .evaluation.future_common import scoring_protocol
        fields = {'id', 'patient', 'target', 'prediction'} | ({'finding'} if task == 'progression' else set())
        if any(set(row) != fields for row in records) or len({row['id'] for row in records}) != count:
            raise ValueError('Future-task prediction records have missing fields or duplicate IDs')
        references = [{key: value for key, value in row.items() if key != 'prediction'} for row in records]
        reference_hash = reference_fingerprint(references)
        scores = summary['tasks'][task]
        if (scores.get('complete') is not True
                or summary.get('references', {}).get(task) != {'n': count, 'references_sha256': reference_hash}):
            raise ValueError('Future-task reference cohort or completion state changed')
        protocol_path = directory / f'{task}_protocol.json'
        if not protocol_path.is_file():
            raise ValueError('Future-task scorer protocol file is missing')
        cfg = load_config(run / 'config.json')
        protocol = json.loads(protocol_path.read_text())
        expected = scoring_protocol(task, references, cfg)
        protocol_hash = hashlib.sha256(json.dumps(expected, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if (protocol != expected or scores.get('schema') != SCHEMA or scores.get('task') != task
                or scores.get('unit') != expected['unit'] or scores.get('reference_sha256') != reference_hash
                or scores.get('protocol_sha256') != protocol_hash):
            raise ValueError('Future-task reference/scorer protocol differs from the current definition')
        if task == 'future_report':
            validate_radgraph_provenance(protocol['radgraph'])
        data_protocol = json.loads((run / 'data_protocol.json').read_text())
        future_metadata = data_protocol.get('future')
        if (not isinstance(future_metadata, dict)
                or summary.get('future_data_fingerprint') != hashlib.sha256(json.dumps(future_metadata, sort_keys=True).encode()).hexdigest()):
            raise ValueError('Future-task data fingerprint differs from the training protocol')
        value = scores.get(TABLE1_METRICS[task])
        if (isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
                or value < 0 or (task != 'remaining_los' and value > 1)):
            raise ValueError('Future-task metric is incomplete or nonfinite')
    else:
        from .evaluation.selection import validate_references
        validate_references(summary, task, records)
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
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
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
