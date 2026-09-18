"""Eight-hour on-demand seed groups, supervised outside the launching session."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

from medworld.config import PROJECT
from medworld.datasets.current import _sha256
from medworld.gpu import ALLOWED_GPUS
from medworld.runtime import atomic_json
from .train import TASKS, VARIANTS
from .online import sr_geometry
from .registration import record as record_experiment


def idle(gpu):
    if gpu not in ALLOWED_GPUS:
        return False
    try:
        text = subprocess.check_output(["nvidia-smi", "-i", gpu,
            "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
        memory, utilization = (int(v.strip()) for v in text.split(","))
        active = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-compute-apps=pid",
            "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
        return memory < 1024 and utilization <= 5 and not active
    except (subprocess.SubprocessError, ValueError):
        return False


def plan(run, cfg):
    jobs = []
    for seed in cfg["seeds"]:
        command = ["-m", "medworld_spatial.train", "--checkpoint", cfg["checkpoint"],
            "--semantic-teacher", cfg["semantic_teacher"], "--out", str(run / "groups" / f"seed{seed}"),
            "--jobs-root", str(run / "jobs"), "--seed", str(seed), "--deadline", str(cfg["deadline"]),
            "--finish-reserve", str(cfg["finish_reserve"]), "--resume"]
        for key in ("steps", "batch_size", "semantic_batch", "train_n", "val_n", "test_n", "human_n", "save_every"):
            command += ["--" + key.replace("_", "-"), str(cfg[key])]
        command += ["--tasks", *cfg.get("tasks", TASKS)]
        command += ["--sr-scale", str(cfg.get("sr_scale", 4))]
        jobs.append({"id": f"seed{seed}", "kind": "matched_training_group", "state": "pending",
                     "variants": list(VARIANTS), "command": command})
    return jobs


def worker(run):
    from .report import build
    run = Path(run).resolve()
    cfg = json.loads((run / "plan.json").read_text())
    jobs, active, last_report = plan(run, cfg), {}, 0
    print(json.dumps({"event": "coordinator_started", "pid": os.getpid(), "time": time.time(),
                      "input_mode": "on_demand_no_disk_cache"}), flush=True)
    while True:
        now = time.time()
        for identity, (process, handle, gpu) in list(active.items()):
            rc = process.poll()
            if rc is None:
                continue
            handle.close()
            job = next(j for j in jobs if j["id"] == identity)
            log = (run / "logs" / f"{identity}.log").read_text(errors="replace")
            if rc and "No idle unlocked GPU" in log and job.get("attempts", 0) < 10:
                job.update(state="pending", returncode=rc)
            else:
                final = run / "groups" / identity / "summary.json"
                result = json.loads(final.read_text()) if final.exists() else {}
                state = result.get("state", "interrupted" if rc == 0 else "failed")
                job.update(state=state, returncode=rc, ended_unix=now)
                print(json.dumps({"event": "worker_exited", "job": identity, "state": state, "returncode": rc}), flush=True)
            del active[identity]
        if now >= cfg["deadline"] - cfg["finish_reserve"]:
            for job in jobs:
                if job["state"] == "pending":
                    job.update(state="not_started_budget", reason="Training window expired")
        occupied = {item[2] for item in active.values()}
        for gpu in ALLOWED_GPUS:
            candidates = [j for j in jobs if j["state"] == "pending"]
            if not candidates:
                break
            if gpu in occupied or not idle(gpu):
                continue
            job = candidates[0]
            command = [sys.executable, *job["command"], "--gpu", gpu]
            handle = (run / "logs" / f"{job['id']}.log").open("a", buffering=1)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                       env=env, cwd=PROJECT)
            active[job["id"]] = (process, handle, gpu)
            job.update(state="running", gpu=gpu, pid=process.pid, started_unix=now, attempts=job.get("attempts", 0) + 1)
            print(json.dumps({"event": "worker_started", "job": job["id"], "pid": process.pid, "gpu": gpu}), flush=True)
        for job in jobs:
            progress = run / "groups" / job["id"] / "status.json"
            if progress.exists():
                job["progress"] = json.loads(progress.read_text())
        state = {"state": "running", "heartbeat_unix": time.time(), "deadline_unix": cfg["deadline"],
                 "allowed_gpus": list(ALLOWED_GPUS), "input_mode": "on_demand_no_disk_cache", "jobs": jobs}
        if not active and not any(j["state"] == "pending" for j in jobs):
            state["state"] = "complete" if all(j["state"] == "complete" for j in jobs) else "finished_with_incomplete_jobs"
            atomic_json(run / "status.json", state)
            build(run, render=True)
            record_experiment(run, state["state"])
            print(json.dumps({"event": "coordinator_finished", "state": state["state"]}), flush=True)
            break
        atomic_json(run / "status.json", state)
        if now - last_report > 60:
            build(run)
            record_experiment(run, "running")
            last_report = now
        time.sleep(10)


def launch(args):
    os.umask(0o077)
    subprocess.run(["systemctl", "--user", "show-environment"], check=True, stdout=subprocess.DEVNULL)
    run = args.out.resolve()
    if run.exists():
        raise ValueError("Choose a new run directory; never overwrite an experiment")
    (run / "source").mkdir(parents=True)
    (run / "logs").mkdir()
    for name in ("medworld", "medworld_spatial"):
        shutil.copytree(PROJECT / "code" / name, run / "source" / name,
            ignore=shutil.ignore_patterns("runs", "__pycache__", ".pytest_cache"))
    source = {str(p.relative_to(run / "source")): _sha256(p) for p in (run / "source").rglob("*") if p.is_file()}
    atomic_json(run / "source_sha256.json", source)
    now = time.time()
    deadline = now + args.hours * 3600
    cfg = {"checkpoint": str(args.checkpoint.resolve()), "semantic_teacher": str(args.semantic_teacher.resolve()),
        "started_unix": now, "deadline": deadline, "budget_hours": args.hours,
        "deadline_local": datetime.fromtimestamp(deadline, ZoneInfo("Asia/Shanghai")).isoformat(),
        "allowed_gpus": list(ALLOWED_GPUS), "excluded_indices": [4, 5], "no_process_termination": True,
        "input_mode": "on_demand_no_disk_cache", "grouping": "Six matched variants share only the current batch",
        "finish_reserve": min(1800, args.hours * 3600 / 8),
        **{key: getattr(args, key) for key in ("train_n", "val_n", "test_n", "human_n", "steps", "batch_size", "semantic_batch", "save_every", "seeds")},
        "variants": list(VARIANTS),
        "tasks": args.tasks, "sr_scale": args.sr_scale, "sr_geometry": sr_geometry(args.sr_scale),
        "register_experiment": args.register,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "source_of_truth": "source_sha256.json includes current uncommitted implementation"}
    atomic_json(run / "plan.json", cfg)
    record_experiment(run, "starting")
    unit = "medworld-spatial-" + run.name.replace("_", "-")
    command = [sys.executable, "-u", "-m", "medworld_spatial.night", "--worker", "--out", str(run)]
    service = ["systemd-run", "--user", "--unit", unit, "--property=Restart=no",
               "--property=TimeoutStopSec=180", "--property=WorkingDirectory=" + str(PROJECT),
               "--property=StandardOutput=append:" + str(run / "launcher.log"),
               "--property=StandardError=append:" + str(run / "launcher.log")]
    for key, value in {"PYTHONPATH": str(run / "source"), "MEDWORLD_PROJECT_ROOT": str(PROJECT),
                       "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
                       "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}.items():
        service.append("--setenv=" + key + "=" + value)
    try:
        subprocess.run([*service, *command], check=True)
    except Exception:
        record_experiment(run, "failed")
        raise
    pid = int(subprocess.check_output(["systemctl", "--user", "show", unit, "--property=MainPID", "--value"], text=True).strip())
    record = {"run": str(run), "pid": pid, "service": unit + ".service", "command": command,
              "deadline_local": cfg["deadline_local"], "supervisor": "systemd user service"}
    atomic_json(run / "launch.json", record)
    atomic_json(PROJECT / "code/medworld_spatial/runs/active.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--semantic-teacher", type=Path)
    p.add_argument("--hours", type=float, default=8)
    for name, value in (("train-n", 4096), ("val-n", 128), ("test-n", 447), ("human-n", 138),
                        ("steps", 4000), ("batch-size", 8), ("semantic-batch", 8), ("save-every", 100)):
        p.add_argument("--" + name, type=int, default=value)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    p.add_argument("--sr-scale", type=int, choices=(4, 8), default=4, help="SR enlargement per axis; 4 means 16x pixels, 8 means 64x pixels")
    p.add_argument("--register", action="store_true", help="Maintain the brief experiment registry and completion history")
    args = p.parse_args()
    if args.worker:
        try:
            worker(args.out)
        except BaseException:
            atomic_json(args.out / "coordinator_failure.json", {"traceback": traceback.format_exc(), "time": time.time()})
            record_experiment(args.out, "failed")
            raise
    else:
        if not args.checkpoint or not args.semantic_teacher:
            p.error("checkpoint and semantic-teacher are required")
        if not 0 < args.hours <= 8:
            p.error("Budget must be positive and at most eight hours")
        if min(args.train_n, args.val_n, args.steps, args.batch_size, args.semantic_batch, args.save_every) <= 0 or len(set(args.seeds)) != len(args.seeds) or len(set(args.tasks)) != len(args.tasks):
            p.error("Counts must be positive; seeds and tasks must be distinct")
        launch(args)
