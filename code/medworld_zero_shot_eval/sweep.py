"""Run per-model GPU preflights followed by full matched baseline evaluations."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from .common import atomic


def register(root, run, outcome=None):
    folder = root / 'experiments'
    with (folder / '.registry.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = folder / 'registry.json'
        entries = json.loads(path.read_text())
        entries = [e for e in entries if e['id'] != run.name]
        rel = str(run.relative_to(root))
        if outcome is None:
            entries.append({'id': run.name, 'summary': 'Native Qwen 0.8B/4B/9B and MedGemma 4B classification/VQA.',
                            'status': 'running', 'observed_at': datetime.now().astimezone().isoformat(),
                            'run': rel, 'live_status': rel + '/status.json'})
        else:
            with (folder / 'README.md').open('a') as f:
                f.write(f'| {datetime.now().astimezone().date()} | `{run.name}` | {outcome} | [Run](../{rel}/) |\n')
        atomic(path, entries)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--config', required=True)
    p.add_argument('--gpus', default='2')
    p.add_argument('--skip-smoke', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--adopt', action='append', default=[], help='Existing worker as model:pid')
    a=p.parse_args()
    from medworld.config import PROJECT, load_config
    root=PROJECT
    run=a.run.resolve(); run.mkdir(parents=True,exist_ok=True)
    # Resolve legacy asset names before launching any child process. Keep the
    # resolved configuration with the run so retries do not reuse stale paths.
    config_path = run / 'evaluation_config.json'
    config = load_config(a.config, root=root)
    atomic(config_path, config)
    gpus=a.gpus.split(',')
    if len(gpus) not in (1,4) or len(set(gpus))!=len(gpus):
        raise ValueError('Provide one GPU for serial evaluation or four distinct GPUs')
    assignments=list(zip(['qwen08b','qwen4b','qwen9b','medgemma4b'], gpus*4 if len(gpus)==1 else gpus))
    register(root,run)
    adopted = {m: int(pid) for m,pid in (v.split(':') for v in a.adopt)}
    state={m:{'status':'pending','gpu':g} for m,g in assignments}
    atomic(run/'status.json',state)
    import threading
    guard=threading.Lock()
    def worker(item):
        model,gpu=item
        if state[model]['status'] == 'failed':
            return
        for phase,extra in [active_phase]:
            status_path = run/model/phase/'status.json'
            if a.resume and status_path.exists():
                while True:
                    saved = json.loads(status_path.read_text())
                    if saved['status'] == 'complete': break
                    if model not in adopted:
                        raise RuntimeError(f'Cannot resume partial worker {model}/{phase}')
                    try: os.kill(adopted[model], 0)
                    except ProcessLookupError:
                        raise RuntimeError(f'Adopted worker stopped: {model}')
                    with guard:
                        state[model]['status']=phase;atomic(run/'status.json',state)
                    time.sleep(10)
                continue
            with guard:
                state[model]['status']=phase;atomic(run/'status.json',state)
            command=[sys.executable,'-m','medworld_zero_shot_eval.evaluate','--config',str(config_path),
                     '--model',model,'--gpu',gpu,'--out',str(run/model/phase),*extra]
            log_path = run/f'{model}_{phase}.log'
            for attempt in range(60):
                with log_path.open('a') as log:
                    result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode == 0 or (run/model/phase).exists(): break
                if 'No idle unlocked GPU' not in log_path.read_text()[-2000:]: break
                time.sleep(10)
            if result.returncode:
                with guard:
                    state[model].update(status='failed',phase=phase,exit_code=result.returncode)
                    atomic(run/'status.json',state)
                return
        with guard:
            state[model]['status']='complete' if phase == 'test' else 'smoke_complete'
            atomic(run/'status.json',state)
    try:
        for active_phase in ([('test',[])] if a.skip_smoke else [('smoke',['--limit','2']),('test',[])]):
            with ThreadPoolExecutor(max_workers=len(gpus)) as pool:list(pool.map(worker,assignments))
        outcome='Completed: all four native models evaluated.' if all(s['status']=='complete' for s in state.values()) else 'Failed or partial; see per-model status and logs.'
    except BaseException:
        register(root,run,'Interrupted; inspect original logs.');raise
    register(root,run,outcome)

if __name__=='__main__':main()
