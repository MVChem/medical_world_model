"""Run one shared Stage 1, then matched slots/native/shuffled Stage-2 branches.

Only explicitly selected idle GPUs from 0,1,2,3,4,7 are eligible. Global UUID
locks match the dense/slot extraction experiments. This queue never stops an
unrelated process. Formal training requires an explicit --train argument.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run import atomic_json, digest

HERE = Path(__file__).resolve().parent
ALLOWED_GPUS = frozenset((0, 1, 2, 3, 4, 7))
STOP = False


def inventory():
    text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                                    "--format=csv,noheader,nounits"], text=True)
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                      "--format=csv,noheader,nounits"], text=True)
    occupied = {line.split(",")[0].strip() for line in active.splitlines() if line.strip()}
    result = {}
    for line in text.splitlines():
        index, uid, memory, utilization = [item.strip() for item in line.split(",")]
        result[int(index)] = dict(uuid=uid, memory_mib=int(memory), utilization=int(utilization),
                                  idle=uid not in occupied and int(memory) < 512 and int(utilization) < 10)
    return result


def acquire_idle(gpus):
    for gpu, info in inventory().items():
        if gpu not in gpus or not info["idle"]:
            continue
        lock = Path(f"/tmp/medworld-frozen-slots-{info['uuid']}.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            continue
        # Verify again after acquiring the shared lock.
        if not inventory().get(gpu, {}).get("idle", False):
            lock.close()
            continue
        return gpu, info["uuid"], lock
    return None


def stop_handler(signum, frame):
    global STOP
    STOP = True


def jobs_for(root, config, smoke, evaluate, score=False):
    if smoke and score:
        raise ValueError("Smoke runs cannot launch clinical scoring")
    evaluate = evaluate or score
    common = [sys.executable, str(HERE / "run.py"), "--config", str(config)]
    if smoke:
        common += ["--smoke"]
    jobs = {"stage1": dict(depends=[], command=common + ["--stage", "stage1", "--condition", "slots",
                            "--out", str(root / "stage1")])}
    for condition in ("native", "slots", "shuffled"):
        jobs[condition] = dict(depends=["stage1"], command=common + ["--stage", "stage2", "--condition", condition,
            "--init-checkpoint", str(root / "stage1/checkpoint_stage1.pt"), "--out", str(root / condition)])
        if evaluate and not smoke:
            name = condition + "_test"
            jobs[name] = dict(depends=[condition], command=[sys.executable, str(HERE / "evaluate.py"),
                "--checkpoint", str(root / condition / "checkpoint_final.pt"),
                "--out", str(root / condition / "evaluation_test"), "--split", "test"])
        if score:
            jobs[condition + "_clinical"] = dict(depends=[condition + "_test"], command=[sys.executable,
                str(HERE / "score_results.py"), "--out", str(root / condition / "evaluation_test"),
                "--condition", condition])
    if score:
        jobs["green"] = dict(depends=[condition + "_clinical" for condition in ("native", "slots", "shuffled")],
                             command=[sys.executable, str(HERE / "green_bridge.py"), "--run", str(root)])
    return jobs


def launch(args):
    global STOP
    STOP = False
    os.umask(0o077)
    if not args.gpus or len(args.gpus) != len(set(args.gpus)) or not set(args.gpus) <= ALLOWED_GPUS:
        raise ValueError("GPU list must be unique and contained in 0,1,2,3,4,7; GPUs 5 and 6 are excluded")
    root = Path(args.out).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Queue requires a fresh output directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir()
    cfg = json.loads(Path(args.config).read_text())
    config = root / "config.json"
    atomic_json(config, cfg)
    jobs = jobs_for(root, config, args.smoke, args.evaluate, args.score)
    atomic_json(root / "queue_plan.json", dict(smoke_only=args.smoke, gpus=args.gpus,
        excluded_gpus=[5, 6], max_parallel=args.max_parallel, jobs=jobs,
        source_hashes={path.name: digest(path) for path in HERE.glob("*.py")}))
    active, states = {}, {name: dict(state="pending") for name in jobs}
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop_handler)
    started, stop_sent = time.monotonic(), False

    def publish(state):
        atomic_json(root / "queue_status.json", dict(state=state, pid=os.getpid(), jobs=states,
                    smoke_only=args.smoke, updated_unix=time.time(), table1_ready=False))

    publish("running")
    while True:
        for name, (process, lock) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            lock.close()
            states[name].update(state="complete" if code == 0 else "failed", exit_code=code,
                                finished_unix=time.time())
            del active[name]
        failed = any(state["state"] == "failed" for state in states.values())
        if failed:
            # Already running independent branches may finish; failed dependencies
            # remain explicitly blocked and no further jobs are launched.
            for name, state in states.items():
                if state["state"] == "pending":
                    state.update(state="blocked", reason="An earlier queue job failed")
        if STOP and not stop_sent:
            for process, _ in active.values():
                process.send_signal(signal.SIGTERM)
            for state in states.values():
                if state["state"] == "pending":
                    state.update(state="interrupted", reason="Queue stop requested")
            stop_sent = True
        if not STOP and not failed:
            for name, job in jobs.items():
                if len(active) >= args.max_parallel:
                    break
                if states[name]["state"] != "pending" or any(states[d]["state"] != "complete" for d in job["depends"]):
                    continue
                allocation = acquire_idle(args.gpus)
                if allocation is None:
                    break
                gpu, uid, lock = allocation
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1",
                           TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1")
                try:
                    with (root / "logs" / f"{name}.log").open("a") as log:
                        process = subprocess.Popen(job["command"], cwd=HERE.parents[1], env=env,
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            pass_fds=(lock.fileno(),))
                except BaseException:
                    lock.close()
                    raise
                active[name] = process, lock
                states[name] = dict(state="running", pid=process.pid, gpu=gpu, gpu_uuid=uid,
                                    started_unix=time.time(), command=job["command"])
                print(json.dumps(dict(job=name, **states[name])), flush=True)
        pending = any(state["state"] == "pending" for state in states.values())
        if not active and not pending:
            state = "interrupted" if STOP else "failed" if failed else "complete"
            publish(state)
            return 130 if STOP else 1 if failed else 0
        if args.max_wait_seconds and not active and time.monotonic() - started >= args.max_wait_seconds:
            for state in states.values():
                if state["state"] == "pending":
                    state.update(state="blocked", reason="No eligible idle GPU before queue wait limit")
            publish("blocked")
            return 2
        publish("stopping" if STOP else "running")
        time.sleep(10)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--train", action="store_true")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 7])
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--max-wait-seconds", type=float, default=0)
    parser.add_argument("--evaluate", action="store_true", help="Generate final held-out predictions after each formal branch")
    parser.add_argument("--score", action="store_true", help="Implies --evaluate; clinical scoring then official GREEN for all three branches")
    args = parser.parse_args(argv)
    if args.max_parallel < 1 or args.max_wait_seconds < 0:
        parser.error("Parallelism must be positive; wait limit must be nonnegative")
    if args.smoke and args.score:
        parser.error("Smoke checkpoints cannot be clinically scored")
    return launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
