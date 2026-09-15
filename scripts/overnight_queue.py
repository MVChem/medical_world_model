"""Durable job queue with independent per-worker deadline enforcement.

Job manifests contain argv arrays, dependencies and required JSON artifacts.
No shell interpolation; only run-owned process groups are signalled.
"""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def timestamp(value):
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError('Deadlines must include a time zone')
    return parsed.timestamp()


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def inventory():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True, timeout=15)
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                     '--format=csv,noheader,nounits'], text=True, timeout=15)
    busy = {line.split(',')[0].strip() for line in active.splitlines() if ',' in line}
    rows = []
    for line in raw.splitlines():
        index, uid, memory, util = [part.strip() for part in line.split(',')]
        rows.append(dict(index=int(index), uuid=uid, idle=uid not in busy and int(memory) < 512 and int(util) < 5))
    return rows


def validate_plan(plan):
    jobs = plan['jobs']
    ids = [job['id'] for job in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate job IDs')
    for job in jobs:
        if not job.get('argv') or not all(isinstance(s, str) for s in job['argv']):
            raise ValueError('argv must be a nonempty string array')
        if not job.get('artifacts'):
            raise ValueError('Each job requires explicit artifacts')
        if not set(job.get('deps', [])) <= set(ids):
            raise ValueError('Unknown dependency')
        if int(job.get('gpus', 1)) not in range(0, 6):
            raise ValueError('GPU request must be 0..5')
    visited = set()
    while len(visited) < len(ids):
        new = {j['id'] for j in jobs if set(j.get('deps', [])) <= visited} - visited
        if not new:
            raise ValueError('Dependency cycle')
        visited |= new


def valid_artifacts(job):
    hashes = {}
    for filename in job['artifacts']:
        path = Path(filename)
        if not path.is_file():
            return None
        if path.suffix == '.json':
            data = read(path)
            if not isinstance(data, (dict, list)):
                return None
            if isinstance(data, dict) and data.get('status') in ('interrupted', 'deadline', 'stopped', 'failed', 'partial'):
                return None
            for key, expected in job.get('success_fields', {}).get(filename, {}).items():
                value = data
                for component in key.split('.'):
                    value = value.get(component) if isinstance(value, dict) else None
                if value != expected:
                    return None
        hashes[filename] = sha(path)
    return hashes


def signal_owned(pid, start, sig):
    if start is not None and identity(pid) == start:
        try:
            os.killpg(pid, sig)
            return True
        except ProcessLookupError:
            pass
    return False


def worker(args):
    record = read(args.record)
    job = record['job']
    locks, child = [], None
    requested = False
    soft, hard = timestamp(record['soft_deadline']), timestamp(record['hard_deadline'])
    result = dict(attempt_id=record['attempt_id'], job_id=job['id'], started=time.time(), returncode=1)
    def stop(signum, frame):
        nonlocal requested
        requested = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        if time.time() >= soft:
            result.update(returncode=124, reason='deadline before launch')
            return
        for card in record['cards']:
            lock = Path(f'/tmp/medworld-frozen-slots-{card["uuid"]}.lock').open('a')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                result.update(returncode=75, reason='card lock busy')
                return
            locks.append(lock)
        if record['cards']:
            available = {c['index'] for c in inventory() if c['idle']}
            if not set(c['index'] for c in record['cards']) <= available:
                result.update(returncode=75, reason='card occupied before launch')
                return
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
                   CUDA_VISIBLE_DEVICES=','.join(str(c['index']) for c in record['cards']),
                   MEDWORLD_SOFT_DEADLINE=record['soft_deadline'], MEDWORLD_HARD_DEADLINE=record['hard_deadline'])
        env.update(job.get('env', {}))
        child = subprocess.Popen(job['argv'], cwd=job.get('cwd', record['project']), env=env,
                                 start_new_session=True, stdin=subprocess.DEVNULL)
        child_start = identity(child.pid)
        record.update(child_pid=child.pid, child_start=child_start)
        atomic(args.record, record)
        soft_sent = False
        while child.poll() is None:
            now = time.time()
            if now >= hard:
                signal_owned(child.pid, child_start, signal.SIGKILL)
                result['reason'] = 'hard deadline'
                break
            if (now >= soft or requested) and not soft_sent:
                signal_owned(child.pid, child_start, signal.SIGTERM)
                result['reason'] = 'graceful deadline or requested stop'
                soft_sent = True
            time.sleep(min(1.0, max(.01, hard - now)))
        result['returncode'] = child.wait(timeout=30)
        result['artifact_sha256'] = valid_artifacts(job) if result['returncode'] == 0 else None
        if result['returncode'] == 0 and result['artifact_sha256'] is None:
            result.update(returncode=124 if soft_sent else 1, reason='required complete artifacts missing')
    except BaseException as exc:
        result.update(returncode=1, reason=f'{type(exc).__name__}: {exc}')
        if child and child.poll() is None:
            signal_owned(child.pid, identity(child.pid), signal.SIGKILL)
            child.wait(timeout=30)
    finally:
        result['finished'] = time.time()
        atomic(record['result'], result)
        for lock in locks:
            lock.close()


def reconcile(job):
    if job['status'] != 'running':
        return
    if job.get('worker_start') is not None and identity(job['worker_pid']) == job['worker_start']:
        return
    result = read(job['result'], {})
    if result.get('attempt_id') == job['attempt_id']:
        rc = result['returncode']
        job.update(finished=result['finished'], returncode=rc, reason=result.get('reason'))
        if rc == 0 and result.get('artifact_sha256') == valid_artifacts(job['spec']):
            job.update(status='complete', artifact_sha256=result['artifact_sha256'])
        elif rc == 75:
            job.update(status='queued', attempts=job['attempts'] - 1, retry_after=time.time() + 30)
        elif rc == 124:
            job['status'] = 'deadline_partial'
        else:
            job.update(status='queued' if job['attempts'] < job['spec'].get('max_attempts', 2) else 'failed', retry_after=time.time() + 30)
    else:
        # An independent worker normally survives coordinator termination. If the
        # worker itself died, retain the card claim until its owned child exits.
        record = read(job['record'], {})
        if record.get('child_start') and identity(record['child_pid']) == record['child_start']:
            job['reason'] = 'worker missing; owned child being stopped'
            signal_owned(record['child_pid'], record['child_start'], signal.SIGTERM)
            return
        job.update(status='failed', reason='worker exited without matching receipt')


def run(args):
    os.umask(0o077)
    root = args.run.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = read(args.plan)
    validate_plan(plan)
    if timestamp(args.soft_deadline) >= timestamp(args.hard_deadline):
        raise ValueError('Soft deadline must precede hard deadline')
    config = dict(plan_sha256=signature(plan), soft_deadline=args.soft_deadline, hard_deadline=args.hard_deadline,
                  max_gpus=args.max_gpus, gpu_order=args.gpu_order, project=str(Path.cwd().resolve()),
                  executor_sha256=sha(__file__))
    old = read(root / 'config.json')
    if old and old != config:
        raise ValueError('Immutable run configuration changed; use a new run')
    atomic(root / 'config.json', config)
    atomic(root / 'plan.json', plan)
    jobs = read(root / 'queue.json')
    if jobs is None:
        jobs = [dict(id=j['id'], spec=j, status='queued', attempts=0) for j in plan['jobs']]
    for job in jobs:
        if job['status'] == 'complete' and job.get('artifact_sha256') != valid_artifacts(job['spec']):
            raise ValueError(f'Completed artifacts changed for {job["id"]}')
    atomic(root / 'queue.json', jobs)
    if args.init_only:
        return
    stop = False
    children = []
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    gpu_order = [int(i) for i in args.gpu_order.split(',')]
    last_report = 0
    while True:
        now = time.time()
        children = [p for p in children if p.poll() is None]
        for job in jobs:
            reconcile(job)
        complete = {j['id'] for j in jobs if j['status'] == 'complete'}
        bad = {j['id'] for j in jobs if j['status'] in ('failed', 'blocked', 'deadline_partial', 'deadline_skipped')}
        for job in jobs:
            if job['status'] == 'queued' and set(job['spec'].get('deps', [])) & bad:
                job.update(status='blocked', reason='dependency incomplete')
            elif job['status'] == 'queued' and (now >= timestamp(args.soft_deadline) or stop):
                job.update(status='deadline_skipped', reason='no new jobs after stopping deadline')
        if now >= timestamp(args.hard_deadline) or stop:
            for job in jobs:
                if job['status'] == 'running':
                    rec = read(job['record'], {})
                    sig = signal.SIGKILL if now >= timestamp(args.hard_deadline) else signal.SIGTERM
                    if rec.get('child_start'):
                        signal_owned(rec['child_pid'], rec['child_start'], sig)
        running = [j for j in jobs if j['status'] == 'running']
        held = {i for j in running for i in j.get('gpus', [])}
        available, error = [], None
        if now < timestamp(args.soft_deadline) and not stop:
            try:
                cards = inventory()
                available = [c for i in gpu_order for c in cards if c['index'] == i and c['idle'] and i not in held]
            except Exception as exc:
                error = str(exc)
        capacity = args.max_gpus - len(held)
        cpu_capacity = 2 - sum(not j.get('gpus') for j in running)
        for job in sorted(jobs, key=lambda j: j['spec'].get('priority', 100)):
            spec = job['spec']
            need = spec.get('gpus', 1)
            eligible = [c for c in available if c['index'] in spec.get('allowed_gpus', gpu_order)]
            if job['status'] != 'queued' or now < job.get('retry_after', 0) or not set(spec.get('deps', [])) <= complete:
                continue
            if (need and (need > capacity or len(eligible) < need)) or (not need and cpu_capacity <= 0):
                continue
            if now + spec.get('minimum_seconds', 0) >= timestamp(args.soft_deadline):
                job.update(status='deadline_skipped', reason='insufficient remaining planned runtime')
                continue
            assigned = eligible[:need]
            job.update(status='running', attempts=job['attempts'] + 1, started=now,
                       gpus=[c['index'] for c in assigned], attempt_id=str(uuid.uuid4()))
            for c in assigned:
                available.remove(c)
            capacity -= need
            cpu_capacity -= not need
            folder = root / 'attempts' / f'{job["id"]}-{job["attempt_id"]}'
            folder.mkdir(parents=True)
            job.update(record=str(folder / 'record.json'), result=str(folder / 'result.json'), log=str(folder / 'worker.log'))
            atomic(job['record'], dict(job=spec, attempt_id=job['attempt_id'], cards=assigned, result=job['result'],
                   soft_deadline=args.soft_deadline, hard_deadline=args.hard_deadline, project=config['project']))
            with Path(job['log']).open('a') as log:
                child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker', '--record', job['record']],
                                         stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            children.append(child)
            job.update(worker_pid=child.pid, worker_start=identity(child.pid))
            atomic(root / 'queue.json', jobs)
            print('launched', job['id'], job['gpus'], child.pid, flush=True)
        counts = {state: sum(j['status'] == state for j in jobs) for state in sorted({j['status'] for j in jobs})}
        finished = not any(j['status'] in ('running', 'queued') for j in jobs)
        atomic(root / 'queue.json', jobs)
        atomic(root / 'status.json', dict(phase='finished' if finished else 'running', counts=counts, updated=time.time(),
               pid=os.getpid(), max_gpus=args.max_gpus, soft_deadline=args.soft_deadline, hard_deadline=args.hard_deadline,
               inventory_error=error, all_complete=all(j['status'] == 'complete' for j in jobs)))
        lines = ['# Overnight experiment queue', '', f'Updated: {dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()}',
                 '', f'GPU limit: {args.max_gpus}; graceful stop {args.soft_deadline}; hard stop {args.hard_deadline}.',
                 '', '| Job | Status | GPUs | Log |', '| --- | --- | --- | --- |']
        for j in jobs:
            lines.append(f'| {j["id"]} | {j["status"]} | {j.get("gpus", [])} | {j.get("log", "")} |')
        (root / 'QUEUE.md').write_text('\n'.join(lines) + '\n')
        if finished or time.time() - last_report >= 120:
            for command in plan.get('report_commands', []):
                try:
                    result = subprocess.run(command, cwd=config['project'], timeout=45, capture_output=True, text=True,
                                            env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
                    if result.returncode:
                        print('report failed:', result.stderr[-1000:], flush=True)
                except Exception as exc:
                    print('report error:', exc, flush=True)
            last_report = time.time()
        if finished:
            return
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('run')
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--max-gpus', type=int, choices=range(1, 6), default=5)
    p.add_argument('--gpu-order', default='0,1,2,3,4')
    p.add_argument('--soft-deadline', default='2026-09-14T07:45:00+08:00')
    p.add_argument('--hard-deadline', default='2026-09-14T08:00:00+08:00')
    p.add_argument('--init-only', action='store_true')
    p = sub.add_parser('worker')
    p.add_argument('--record', type=Path, required=True)
    args = parser.parse_args()
    worker(args) if args.mode == 'worker' else run(args)


if __name__ == '__main__':
    main()
