"""Durable, free-GPU-only queue for the frozen encoder-slot experiment.

The coordinator may exit/restart without terminating its workers. A worker
rechecks GPU occupancy, holds a shared per-card advisory lock, and records its
exit status. Only successful workers create run-specific completion records.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time
import uuid

MODELS = ["qwen08b", "qwen4b", "medgemma4b", "qwen9b", "qwen27b_fp8", "medgemma27b"]
CONTROLS = ["qwen4b", "medgemma4b"]
DEFAULT_RUN = Path(__file__).resolve().parent / "runs/frozen_slots_20260913"


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def gpu_inventory():
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True, timeout=15)
    active = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits"], text=True, timeout=15)
    busy = {line.split(",")[0].strip() for line in active.splitlines() if "," in line}
    result = []
    for line in raw.splitlines():
        idx, gpu_uuid, mem, util = [v.strip() for v in line.split(",")]
        result.append(dict(index=int(idx), uuid=gpu_uuid, memory_mib=int(mem),
                           utilization=int(util), has_compute_process=gpu_uuid in busy))
    return result


def idle(gpu):
    return not gpu["has_compute_process"] and gpu["memory_mib"] < 512 and gpu["utilization"] < 5


def process_start(pid):
    try:
        # Fields after the process name begin with state (field 3).
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def make_jobs(run, models, controls, epochs, seed, batch_size, microbatch):
    jobs = []

    def add(job_id, script, args, artifact, deps=()):
        jobs.append(dict(id=job_id, script=script, args=args, artifact=str(artifact),
                         deps=list(deps), status="queued", attempts=0))

    def train(model, task, condition, deps=()):
        args = ["--data-run", str(run), "--run", str(run), "--model", model,
                "--task", task, "--condition", condition, "--epochs", str(epochs),
                "--seed", str(seed), "--batch-size", str(batch_size),
                "--microbatch", str(microbatch)]
        add(f"{model}_{task}_{condition}", "frozen_slots_train.py", args,
            run / model / f"{task}_{condition}" / "metrics.json", deps)

    for task in ("segmentation", "sr"):
        train("image_only", task, "image_only")
    for model in models:
        feature_id = f"extract_{model}"
        add(feature_id, "frozen_slots_extract.py", ["--run", str(run), "--model", model],
            run / model / "slots_complete.json")
        for task in ("segmentation", "sr"):
            train(model, task, "slots", [feature_id])
    # Controls begin after every main comparison has completed successfully.
    main_jobs = [job["id"] for job in jobs]
    for model in controls:
        for task in ("segmentation", "sr"):
            train(model, task, "shuffled_slots", main_jobs)
    return jobs


def valid_completion(run, job, run_id):
    marker = read_json(run / "completion" / f'{job["id"]}.json', {})
    artifact = Path(job["artifact"])
    return bool(marker.get("run_id") == run_id and marker.get("job_signature") == job["signature"]
                and artifact.exists() and marker.get("artifact_sha256") == digest(artifact))


def worker(args):
    """Execute one script in this process so the visible GPU claim persists."""
    started = time.time()
    rc, reason = 1, None
    lock = None
    try:
        gpu = next((g for g in gpu_inventory() if g["index"] == args.gpu), None)
        if gpu is None:
            raise RuntimeError(f"GPU {args.gpu} is absent")
        lock_path = Path("/tmp") / f'medworld-frozen-slots-{gpu["uuid"]}.lock'
        # Do not truncate a preexisting shared lockfile.
        lock = lock_path.open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("GPU_BUSY_AT_START: another queue holds this card", flush=True)
            rc, reason = 75, "GPU lock busy"
            return rc
        gpu = next(g for g in gpu_inventory() if g["index"] == args.gpu)
        if not idle(gpu):
            print("GPU_BUSY_AT_START: yielding to an existing process", flush=True)
            rc, reason = 75, "GPU no longer idle"
            return rc
        import torch
        reservation = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
        command = list(args.command)
        if command and command[0] == "--":
            command.pop(0)
        script = Path(command[0]).resolve()
        sys.argv = command
        sys.path.insert(0, str(script.parent))
        try:
            runpy.run_path(str(script), run_name="__main__")
            rc = 0
        except SystemExit as exc:
            rc = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        del reservation
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        reason = f"{type(exc).__name__}: {exc}"
        rc = 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        atomic(args.result, dict(run_id=args.run_id, job_id=args.job_id,
                                attempt_id=args.attempt_id, returncode=rc,
                                started=started, finished=time.time(), reason=reason))
        if lock is not None:
            lock.close()
    return rc


def initialize(args):
    run, source = args.run.resolve(), args.source.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if args.data:
        data = args.data.resolve()
        if not (data / "manifest.json").exists() and (data / "data/manifest.json").exists():
            data = data / "data"
        if not (run / "data").exists():
            (run / "data").symlink_to(data, target_is_directory=True)
        elif (run / "data").resolve() != data:
            raise RuntimeError("Existing run/data differs from --data; choose a new run")
    for path in [run / "data/manifest.json", run / "models.json"]:
        if not path.is_file():
            raise FileNotFoundError(f"Required experiment input: {path}")
    for name in ["frozen_slots_queue.py", "frozen_slots_report.py", "frozen_slots_train.py", "frozen_slots_extract.py"]:
        if not (source / name).is_file():
            raise FileNotFoundError(f"Required source snapshot file: {source / name}")
    models = args.models.split(",")
    controls = [m for m in args.controls.split(",") if m]
    if len(models) != len(set(models)) or not set(models) <= set(MODELS):
        raise ValueError("--models must contain unique supported model IDs")
    if not set(controls) <= set(models):
        raise ValueError("Control models must be in --models")
    jobs = make_jobs(run, models, controls, args.epochs, args.seed, args.batch_size, args.microbatch)
    protocol = read_json(run / "protocol.json", {})
    project = protocol.get("project") or os.environ.get("MEDWORLD_PROJECT")
    if not project:
        project = next((str(parent) for parent in run.parents
                        if (parent / "code/medworld_dense_baselines").is_dir()), str(Path.cwd()))
    settings = dict(project=str(Path(project).resolve()), source=str(source),
                    source_sha256={p.name: digest(p) for p in sorted(source.glob("*.py"))},
                    data=str((run / "data").resolve()), data_manifest_sha256=digest(run / "data/manifest.json"),
                    models_sha256=digest(run / "models.json"), models=models, controls=controls,
                    epochs=args.epochs, seed=args.seed, batch_size=args.batch_size, microbatch=args.microbatch,
                    delivery_target=args.delivery_target,
                    protocol_sha256=digest(run / "protocol.json") if (run / "protocol.json").exists() else None)
    config_path = run / "queue_config.json"
    config = read_json(config_path)
    if config:
        if config["settings"] != settings:
            raise RuntimeError("Run settings or source snapshot changed; use a new run directory")
    else:
        config = dict(run_id=str(uuid.uuid4()), created=time.time(), settings=settings,
                      policy="at most 4 idle GPUs; no foreign process preemption; native vision tower only")
        atomic(config_path, config)
    for job in jobs:
        job["signature"] = signature(dict(run_id=config["run_id"], job=job))
    old_jobs = read_json(run / "queue.json")
    if old_jobs is not None:
        if {j["id"]: j["signature"] for j in old_jobs} != {j["id"]: j["signature"] for j in jobs}:
            raise RuntimeError("Persisted queue differs from requested immutable job plan")
        jobs = old_jobs
    for folder in ["logs", "worker_results", "completion", "preview"]:
        (run / folder).mkdir(exist_ok=True)
    for job in jobs:
        if valid_completion(run, job, config["run_id"]):
            job["status"] = "complete"
        elif job["status"] == "complete":
            raise RuntimeError(f'Completion provenance mismatch for {job["id"]}')
        elif args.retry_failed and job["status"] in ("failed", "blocked"):
            job.update(status="queued", attempts=0, retry_after=0)
            job.pop("reason", None)
    atomic(run / "queue.json", jobs)
    return run, source, config, jobs


def write_report(run, source):
    try:
        completed = subprocess.run([sys.executable, str(source / "frozen_slots_report.py"),
                                    "--run", str(run)], timeout=45, capture_output=True, text=True)
        if completed.returncode:
            print("Report failed:", completed.stderr[-2000:], flush=True)
    except subprocess.TimeoutExpired:
        print("Report exceeded 45 seconds; queue continues", flush=True)


def run_queue(args):
    os.umask(0o077)
    args.run.mkdir(parents=True, exist_ok=True)
    coordinator_lock = (args.run / "frozen_slots_coordinator.lock").open("a")
    fcntl.flock(coordinator_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run, source, config, jobs = initialize(args)
    if args.init_only:
        write_report(run, source)
        print(json.dumps(dict(run=str(run), run_id=config["run_id"], jobs=len(jobs),
                              status="initialized; no workers launched")), flush=True)
        return 0
    children, stop = {}, False
    started, last_report = time.time(), 0

    def request_stop(signum, frame):
        nonlocal stop
        stop = True
        print("Coordinator stopping; workers continue and can be adopted on restart", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    gpu_order = [int(x) for x in args.gpu_order.split(",")]
    inventory, inventory_error = [], None
    while not stop:
        now = time.time()
        for job in jobs:
            if job["status"] != "running":
                continue
            child = children.get(job["id"])
            if child:
                child.poll()
            if process_start(job["pid"]) == job.get("pid_start") and job.get("pid_start") is not None:
                continue
            result = read_json(job["worker_result"], {})
            matched = result.get("attempt_id") == job["attempt_id"] and result.get("run_id") == config["run_id"]
            rc = result.get("returncode") if matched else None
            job.update(finished=now, returncode=rc)
            if rc == 0 and Path(job["artifact"]).is_file():
                # JSON must be readable before a completed record is committed.
                artifact = read_json(job["artifact"])
                if not isinstance(artifact, dict):
                    rc = 1
                    job["reason"] = "Worker exited 0 but artifact JSON is invalid"
                else:
                    atomic(run / "completion" / f'{job["id"]}.json',
                           dict(run_id=config["run_id"], job_signature=job["signature"],
                                attempt_id=job["attempt_id"], finished=now,
                                artifact=job["artifact"], artifact_sha256=digest(job["artifact"])))
                    job["status"] = "complete"
            if job["status"] != "complete":
                if rc == 75:
                    job.update(status="queued", attempts=max(0, job["attempts"] - 1), retry_after=now + 30)
                else:
                    job.update(status="failed" if job["attempts"] >= args.max_attempts else "queued",
                               retry_after=now + 90)
                    job["reason"] = job.get("reason") or result.get("reason") or "Worker failed or exit record missing"
            print("finished", job["id"], job["status"], rc, flush=True)
        failed = {j["id"] for j in jobs if j["status"] in ("failed", "blocked")}
        for job in jobs:
            if job["status"] == "queued" and set(job["deps"]) & failed:
                job.update(status="blocked", reason="Dependency failed: " + ", ".join(sorted(set(job["deps"]) & failed)))
        try:
            inventory, inventory_error = gpu_inventory(), None
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            inventory_error = f"{type(exc).__name__}: {exc}"
            print("GPU inventory unavailable; no launches:", inventory_error, flush=True)
        held = {j["gpu"] for j in jobs if j["status"] == "running"}
        capacity = max(0, args.max_gpus - len(held))
        free = [] if inventory_error else [i for i in gpu_order if i not in held and any(
            g["index"] == i and idle(g) for g in inventory)]
        complete = {j["id"] for j in jobs if j["status"] == "complete"}
        for job in jobs:
            if not capacity or not free:
                break
            if job["status"] != "queued" or job.get("retry_after", 0) > now or not set(job["deps"]) <= complete:
                continue
            gpu = free.pop(0)
            capacity -= 1
            job["attempts"] += 1
            attempt = str(uuid.uuid4())
            log_path = run / "logs" / f'{job["id"]}_attempt{job["attempts"]}_{attempt[:8]}.log'
            result_path = run / "worker_results" / f"{attempt}.json"
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MEDWORLD_MODEL_FILE=str(run / "models.json"),
                       MEDWORLD_PROJECT=config["settings"]["project"],
                       OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
                       HF_HUB_DISABLE_PROGRESS_BARS="1", TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR="1",
                       MEDWORLD_FROZEN_SLOTS_RUN_ID=config["run_id"])
            command = [sys.executable, "-u", str(source / "frozen_slots_queue.py"), "worker",
                       "--gpu", str(gpu), "--result", str(result_path), "--run-id", config["run_id"],
                       "--job-id", job["id"], "--attempt-id", attempt, "--",
                       str(source / job["script"])] + job["args"]
            with log_path.open("a") as log:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                         env=env, start_new_session=True)
            children[job["id"]] = child
            job.update(status="running", pid=child.pid, pid_start=process_start(child.pid), gpu=gpu,
                       started=now, log=str(log_path), worker_result=str(result_path), attempt_id=attempt,
                       command=command)
            job.pop("reason", None)
            atomic(run / "queue.json", jobs)
            print("started", job["id"], "GPU", gpu, "pid", child.pid, flush=True)
        counts = {state: sum(j["status"] == state for j in jobs)
                  for state in ["queued", "running", "complete", "failed", "blocked"]}
        pending = counts["queued"] + counts["running"]
        phase = "running" if pending else "finished_with_errors" if counts["failed"] + counts["blocked"] else "finished"
        atomic(run / "queue.json", jobs)
        atomic(run / "status.json", dict(run_id=config["run_id"], pid=os.getpid(), started=started, updated=now,
                                        counts=counts, gpus=inventory, inventory_error=inventory_error,
                                        status=phase, phase=phase, max_gpus=args.max_gpus, delivery_target=args.delivery_target,
                                        policy=config["policy"]))
        if now - last_report >= args.report_seconds or not pending:
            write_report(run, source)
            last_report = now
        if not pending:
            return 1 if counts["failed"] + counts["blocked"] else 0
        time.sleep(args.poll_seconds)
    atomic(run / "queue.json", jobs)
    status = read_json(run / "status.json", {})
    status.update(status="coordinator_stopped_workers_continue", phase="coordinator_stopped_workers_continue", updated=time.time())
    atomic(run / "status.json", status)
    write_report(run, source)
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        parser = argparse.ArgumentParser()
        parser.add_argument("--gpu", type=int, required=True)
        parser.add_argument("--result", type=Path, required=True)
        parser.add_argument("--run-id", required=True)
        parser.add_argument("--job-id", required=True)
        parser.add_argument("--attempt-id", required=True)
        parser.add_argument("command", nargs=argparse.REMAINDER)
        return worker(parser.parse_args(sys.argv[2:]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--data", type=Path, help="Prepared data directory or its parent run")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--controls", default=",".join(CONTROLS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--max-gpus", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--gpu-order", default="2,3,7,6,5,1,0,4")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--report-seconds", type=float, default=120)
    parser.add_argument("--delivery-target", default="2026-09-13T20:00:00+08:00")
    parser.add_argument("--init-only", action="store_true", help="Write reviewable queue/report without launching GPU jobs")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < 1 or args.report_seconds < 1 or args.max_attempts < 1:
        parser.error("Poll/report intervals and max attempts must be positive")
    return run_queue(args)


if __name__ == "__main__":
    sys.exit(main())
