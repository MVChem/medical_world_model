"""Run budget-limited baseline jobs on any idle GPU using the shared GPU locks.

The established executor owns worker process groups, validates completed artifact
hashes, and resumes work after coordinator restarts. This sweep has no 09-14 08:00
deadline; each training command has an explicit sample/update budget.
"""
import argparse
import importlib.util
import os
from pathlib import Path

def main():
    project=Path(os.environ.get('MEDWORLD_PROJECT', Path(__file__).resolve().parents[2]))
    adjacent=Path(__file__).with_name('executor.py')
    executor=adjacent if adjacent.exists() else project/'scripts/overnight_queue.py'
    spec=importlib.util.spec_from_file_location('baseline_executor',executor)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='mode',required=True)
    run=sub.add_parser('run')
    run.add_argument('--run',type=Path,required=True)
    run.add_argument('--plan',type=Path,required=True)
    run.add_argument('--max-gpus',type=int,choices=range(1,9),default=8)
    run.add_argument('--gpu-order',default='7,3,0,1,2,4,5,6')
    run.add_argument('--soft-deadline',default='2099-01-01T00:00:00+08:00')
    run.add_argument('--hard-deadline',default='2099-01-01T00:15:00+08:00')
    run.add_argument('--init-only',action='store_true')
    worker=sub.add_parser('worker')
    worker.add_argument('--record',type=Path,required=True)
    args=p.parse_args()
    # Worker argv must use this wrapper so source resolution remains immutable.
    module.__file__=str(Path(__file__).resolve())
    module.worker(args) if args.mode=='worker' else module.run(args)

if __name__=='__main__':main()
