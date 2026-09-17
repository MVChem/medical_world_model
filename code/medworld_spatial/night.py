"""Eight-hour isolated queue; idle GPUs only, no process termination."""
import argparse
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from medworld.config import PROJECT
from medworld.datasets.current import _sha256
from medworld.runtime import atomic_json

ALLOWED_GPUS = ("0", "1", "2", "6", "7")
VARIANTS = ("image_only", "visual_slots", "slots", "featup", "featup_semantic", "image_only_featup")


def idle(gpu):
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
    reserve = min(3600, (cfg["deadline"] - cfg["started_unix"]) / 4)
    for task in ("segmentation", "sr"):
        jobs.append({"id": f"prepare_{task}", "kind": "prepare", "state": "pending", "command": [
            "-m", "medworld_spatial.prepare", "--checkpoint", cfg["checkpoint"],
            "--semantic-teacher", cfg["semantic_teacher"], "--out", str(run / "cache" / task),
            "--task", task, "--train-n", str(cfg["train_n"]), "--val-n", str(cfg["val_n"]),
            "--test-n", "447", "--human-n", "138", "--views", "2", "--batch-size", "8",
            "--semantic-batch", "8", "--deadline", str(cfg["deadline"] - reserve),
            "--source-commit", cfg["source_commit"]]})
    for seed in cfg["seeds"]:
        for variant in VARIANTS:
            jobs.append({"id": f"{variant}_seed{seed}", "kind": "train", "state": "pending", "command": [
                "-m", "medworld_spatial.train", "--cache", str(run / "cache"),
                "--out", str(run / "jobs" / f"{variant}_seed{seed}"), "--variant", variant,
                "--steps", str(cfg["steps"]), "--batch-size", str(cfg["batch_size"]), "--seed", str(seed),
                "--deadline", str(cfg["deadline"]), "--finish-reserve",
                str(min(1200, (cfg["deadline"] - cfg["started_unix"]) / 20)), "--resume"]})
    return jobs


def worker(run):
    from .report import build
    run = Path(run).resolve()
    cfg = json.loads((run / "plan.json").read_text())
    jobs = plan(run, cfg)
    active = {}
    last_report = 0
    while True:
        now = time.time()
        for identity, (process, handle, gpu) in list(active.items()):
            rc = process.poll()
            if rc is None:
                continue
            handle.close()
            job = next(j for j in jobs if j["id"] == identity)
            # A GPU can become busy between checking and locking. Retry only
            # this acquisition failure; never interfere with its owner.
            log = (run / "logs" / f"{identity}.log").read_text(errors="replace")
            if rc and "No idle unlocked GPU" in log and job.get("attempts", 0) < 10:
                job.update(state="pending", returncode=rc)
            else:
                job.update(state="complete" if rc == 0 else "failed", returncode=rc, ended_unix=now)
            del active[identity]
        prepared = all(j["state"] == "complete" for j in jobs if j["kind"] == "prepare")
        prepare_failed = any(j["state"] == "failed" for j in jobs if j["kind"] == "prepare")
        if prepare_failed:
            for job in jobs:
                if job["kind"] == "train" and job["state"] == "pending":
                    job.update(state="blocked", reason="Feature/semantic cache preparation failed; see logs")
        if now >= cfg["deadline"] - min(1800, (cfg["deadline"] - cfg["started_unix"]) / 16):
            for job in jobs:
                if job["state"] == "pending":
                    job.update(state="not_started_budget", reason="Preserve time for evaluation and export")
        occupied = {item[2] for item in active.values()}
        for gpu in ALLOWED_GPUS:
            if gpu in occupied or not idle(gpu):
                continue
            candidates = [j for j in jobs if j["state"] == "pending" and (j["kind"] == "prepare" or prepared)]
            if not candidates:
                break
            job = candidates[0]
            command = [sys.executable, *job["command"], "--gpu", gpu]
            handle = (run / "logs" / f"{job['id']}.log").open("a", buffering=1)
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=os.environ.copy(),
                                       cwd=PROJECT, start_new_session=True)
            active[job["id"]] = (process, handle, gpu)
            job.update(state="running", gpu=gpu, pid=process.pid, started_unix=now, attempts=job.get("attempts", 0) + 1)
        state = {"state": "running", "heartbeat_unix": now, "deadline_unix": cfg["deadline"],
                 "allowed_gpus": list(ALLOWED_GPUS), "jobs": jobs}
        if not active and not any(j["state"] == "pending" for j in jobs):
            state["state"] = "complete" if all(j["state"] == "complete" for j in jobs) else "finished_with_incomplete_jobs"
            atomic_json(run / "status.json", state)
            build(run, render=True)
            break
        atomic_json(run / "status.json", state)
        if now - last_report > 60:
            build(run)
            last_report = now
        time.sleep(15)


def launch(args):
    os.umask(0o077)
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
    local = datetime.now(ZoneInfo("Asia/Shanghai"))
    morning = local.replace(hour=8, minute=0, second=0, microsecond=0)
    if morning <= local:
        morning += timedelta(days=1)
    deadline = min(now + args.hours * 3600, morning.timestamp() - 300)
    cfg = {"checkpoint": str(args.checkpoint.resolve()), "semantic_teacher": str(args.semantic_teacher.resolve()),
        "started_unix": now, "deadline": deadline, "budget_hours": (deadline - now) / 3600,
        "deadline_local": datetime.fromtimestamp(deadline, ZoneInfo("Asia/Shanghai")).isoformat(),
        "allowed_gpus": list(ALLOWED_GPUS), "excluded_indices": [3, 4, 5], "no_process_termination": True,
        "train_n": args.train_n, "val_n": args.val_n, "steps": args.steps, "batch_size": args.batch_size,
        "seeds": args.seeds, "variants": list(VARIANTS),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()}
    atomic_json(run / "plan.json", cfg)
    env = dict(os.environ, PYTHONPATH=str(run / "source"), MEDWORLD_PROJECT_ROOT=str(PROJECT),
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    handle = (run / "launcher.log").open("a")
    command = [sys.executable, "-m", "medworld_spatial.night", "--worker", "--out", str(run)]
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env,
                               cwd=PROJECT, start_new_session=True)
    handle.close()
    atomic_json(run / "launch.json", {"pid": process.pid, "command": command})
    atomic_json(PROJECT / "code/medworld_spatial/runs/active.json", {"run": str(run), "pid": process.pid,
                                                                  "deadline_local": cfg["deadline_local"]})
    print(json.dumps({"run": str(run), "pid": process.pid, "deadline_local": cfg["deadline_local"]}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--semantic-teacher", type=Path)
    p.add_argument("--hours", type=float, default=8)
    p.add_argument("--train-n", type=int, default=4096)
    p.add_argument("--val-n", type=int, default=128)
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = p.parse_args()
    if args.worker:
        worker(args.out)
    else:
        if not args.checkpoint or not args.semantic_teacher:
            p.error("checkpoint and semantic-teacher are required")
        if not 0 < args.hours <= 8:
            p.error("Budget must be positive and at most eight hours")
        if min(args.train_n, args.val_n, args.steps, args.batch_size) <= 0 or len(set(args.seeds)) != len(args.seeds):
            p.error("Counts must be positive and seeds must be distinct")
        launch(args)
