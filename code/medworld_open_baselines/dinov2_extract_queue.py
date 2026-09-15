"""Serial feature extraction on one idle GPU, sharing the existing project lock."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from dinov2_features import DENSE, ROOT, atomic


def main(args):
    uuid = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=uuid',
                                    '--format=csv,noheader'], text=True).strip()
    args.run.mkdir(parents=True, exist_ok=True)
    state = args.run / 'feature_queue_status.json'
    with (Path('/tmp') / f'medworld-frozen-slots-{uuid}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                        '--format=csv,noheader'], text=True)
        usage = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
                '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        memory, utilization = [int(x.strip()) for x in usage.split(',')]
        if uuid in apps or memory > 512 or utilization >= 5:
            raise RuntimeError('allocated GPU is no longer idle')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
        for task in ('dense', 'classification'):
            for model in ('dinov2_vitb14', 'chexworld'):
                cmd = [sys.executable, str(ROOT / 'dinov2_features.py'), '--run', str(args.run),
                       '--data-run', str(args.data_run), '--model', model, '--task', task,
                       '--batch-size', '8']
                log = args.run / f'{model}_{task}_features.log'
                with log.open('a') as output:
                    child = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT, env=env,
                                             pass_fds=(lock.fileno(),))
                    atomic(state, dict(status='running', parent_pid=os.getpid(), pid=child.pid,
                                       gpu=args.gpu, gpu_uuid=uuid, model=model, task=task,
                                       command=cmd, log=str(log), started=time.time()))
                    code = child.wait()
                if code:
                    atomic(state, dict(status='failed', model=model, task=task, returncode=code, log=str(log)))
                    raise SystemExit(code)
        atomic(state, dict(status='complete', gpu=args.gpu, gpu_uuid=uuid, finished=time.time()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--data-run', type=Path, default=DENSE / 'runs/dense_20260912')
    main(p.parse_args())
