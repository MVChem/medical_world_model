"""Reserve physical GPUs, freeze source, launch torchrun and monitor the run."""
import argparse
from datetime import datetime
import fcntl
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .config import PROJECT, load_config


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def gpu_status(selector):
    fields = ("index", "uuid", "memory.used", "memory.total", "utilization.gpu", "power.draw")
    line = subprocess.check_output(["nvidia-smi", "-i", selector, "--query-gpu=" + ",".join(fields),
                                    "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
    values = [value.strip() for value in line.split(",")]
    if len(values) != len(fields):
        raise ValueError(f"Unexpected GPU status for {selector}")
    return dict(zip(fields, values))


def reserve(selectors):
    locks, devices = [], []
    try:
        for selector in selectors:
            initial = gpu_status(selector)
            if initial["uuid"] in {d["uuid"] for d in devices}:
                raise ValueError("Select distinct GPUs")
            lock = open(f"/tmp/medworld-frozen-slots-{initial['uuid']}.lock", "a")
            locks.append(lock)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            active = subprocess.check_output(["nvidia-smi", "-i", selector, "--query-compute-apps=pid",
                                               "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
            status = gpu_status(selector)
            if active or int(status["memory.used"]) > 1024 or int(status["utilization.gpu"]) > 5:
                raise RuntimeError(f"GPU {selector} is occupied; no jobs were modified")
            devices.append(status)
        return locks, devices
    except BaseException:
        for lock in locks:
            lock.close()
        raise


def snapshot(out):
    source = Path(__file__).resolve().parent
    manifest = {}
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if not path.is_file() or any(p in ("runs", "__pycache__") for p in relative.parts):
            continue
        destination = out / "source/medworld" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        manifest[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic(out / "source_manifest.json", manifest)


def worker_pids(out, torchrun_pid):
    # torchrun starts workers in separate sessions/process groups. Verify their
    # actual parent instead of assuming they share torchrun's process group.
    result = []
    for path in Path(out).glob("rank*.pid"):
        try:
            pid = int(path.read_text())
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[1]) == torchrun_pid:
                result.append(pid)
        except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
            pass
    return result


def signal_workers(out, torchrun_pid, sig):
    for pid in worker_pids(out, torchrun_pid):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", required=True, help="Physical GPU IDs, e.g. 2,3")
    parser.add_argument("--config")
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", help="Checkpoint from this same distributed run")
    parser.add_argument("--monitor-seconds", type=float, default=10)
    parser.add_argument("--cpu-base", type=int, help="First physical CPU core for rank affinity, e.g. 32 for GPUs 4-7")
    parser.add_argument("--skip-evaluation", action="store_true",
                        help="Explicitly skip full held-out tests (e.g. a short preflight)")
    args = parser.parse_args()
    selectors = args.gpus.split(",")
    if len(selectors) < 2 or len(set(selectors)) != len(selectors):
        parser.error("At least two distinct GPUs are required")
    if args.monitor_seconds <= 0:
        parser.error("monitor-seconds must be positive")
    if args.resume and args.config:
        parser.error("Resume uses its original configuration and frozen source")
    out = Path(args.out).resolve()
    if args.resume:
        if Path(args.resume).resolve().parent != out or not (out / "source_manifest.json").exists():
            parser.error("Resume requires a checkpoint and frozen source in the existing run folder")
        for name, digest in json.loads((out / "source_manifest.json").read_text()).items():
            if hashlib.sha256((out / "source/medworld" / name).read_bytes()).hexdigest() != digest:
                parser.error("Frozen source was modified")
    elif out.exists() and any(out.iterdir()):
        parser.error("Choose an empty output folder")
    cfg = None if args.resume else load_config(args.config)
    timed = bool((cfg or json.loads((out / "requested_config.json").read_text())).get("total_hours", 0))
    locks, devices = reserve(selectors)
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        snapshot(out)
        atomic(out / "requested_config.json", cfg)
    launch = {"launcher_pid": os.getpid(), "started_unix": time.time(), "started_local": datetime.now().isoformat(),
              "gpus": devices, "world_size": len(devices), "resume": args.resume, "project_root": str(PROJECT)}
    packages = {}
    for name in ("torch", "transformers", "triton", "flash-linear-attention", "fla-core", "causal-conv1d"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    launch["packages"] = packages
    atomic(out / "launch.json", launch)
    runtime_cfg = cfg or json.loads((out / "requested_config.json").read_text())
    cpu_threads = str(runtime_cfg.get("cpu_threads", 8))
    cpu_cores = runtime_cfg.get("cpu_cores_per_rank", 8)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(d["uuid"] for d in devices),
                       MEDWORLD_PROJECT_ROOT=str(PROJECT), PYTHONPATH=str(out / "source"),
                       OMP_NUM_THREADS=cpu_threads, MKL_NUM_THREADS=cpu_threads, TOKENIZERS_PARALLELISM="false",
                       MEDWORLD_CPU_CORES_PER_RANK=str(cpu_cores),
                       HF_HUB_OFFLINE="1", TORCH_NCCL_ASYNC_ERROR_HANDLING="1", NCCL_DEBUG="WARN",
                       PYTHONUNBUFFERED="1")
    if args.cpu_base is not None:
        allowed = set(os.sched_getaffinity(0))
        selected = allowed & set(range(args.cpu_base, args.cpu_base + len(devices) * cpu_cores))
        if len(selected) != len(devices) * cpu_cores:
            raise ValueError("Requested CPU affinity is unavailable")
        os.sched_setaffinity(0, selected)
        launch["cpu_affinity"] = sorted(selected)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(devices)}",
               "--module", "medworld.distributed_train", "--out", str(out)]
    command += ["--resume", str(Path(args.resume).resolve())] if args.resume else ["--config", str(out / "requested_config.json")]
    stop_requested = False
    def request_stop(*_args):
        nonlocal stop_requested
        stop_requested = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    log_path = out / ("resume.log" if args.resume else "train.log")
    with log_path.open("a") as log:
        child = subprocess.Popen(command, cwd=PROJECT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        launch.update(torchrun_pid=child.pid, command=command, log=str(log_path))
        atomic(out / "launch.json", launch)
        print(json.dumps(launch), flush=True)
        signalled = False
        stop_at = None
        try:
            while child.poll() is None:
                now = time.time()
                status_path = out / "status.json"
                status = json.loads(status_path.read_text()) if status_path.exists() else {}
                deadline = status.get("deadline_unix") if timed else None
                if deadline and now > deadline + 300:
                    stop_requested = True
                if stop_requested and not signalled:
                    # Wait until initialization has produced a fresh heartbeat;
                    # before that, worker stop handlers might not be installed.
                    if status.get("heartbeat_unix", 0) >= launch["started_unix"]:
                        signal_workers(out, child.pid, signal.SIGUSR1)
                        signalled = True
                    if stop_at is None:
                        stop_at = now
                if stop_at and now - stop_at > 300:
                    signal_workers(out, child.pid, signal.SIGKILL)
                    os.killpg(child.pid, signal.SIGKILL)
                    break
                telemetry = {"wall_unix": now, "gpus": []}
                for selector in selectors:
                    try:
                        telemetry["gpus"].append(gpu_status(selector))
                    except (subprocess.SubprocessError, ValueError):
                        telemetry["gpus"].append({"index": selector, "query_failed": True})
                with (out / "gpu_telemetry.jsonl").open("a") as handle:
                    handle.write(json.dumps(telemetry) + "\n")
                time.sleep(args.monitor_seconds)
            code = child.wait()
            atomic(out / "launcher_status.json", {"exit_code": code, "finished_unix": time.time(),
                                                  "stop_requested": stop_requested, "torchrun_pid": child.pid})
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    signal_workers(out, child.pid, signal.SIGKILL)
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            for lock in locks:
                lock.close()
    if code == 0 and not stop_requested:
        state = json.loads((out / "status.json").read_text())
        if state.get("complete") and not state.get("stopped"):
            if args.skip_evaluation:
                atomic(out / "pipeline_status.json", {"phase": "training_complete", "evaluation_complete": False,
                       "evaluation_skipped": True, "heartbeat_unix": time.time()})
            else:
                from .evaluate_run import evaluate_run
                def interrupt_evaluation(*_):
                    raise KeyboardInterrupt
                for sig in (signal.SIGINT, signal.SIGTERM):
                    signal.signal(sig, interrupt_evaluation)
                evaluate_run(out, selectors)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
