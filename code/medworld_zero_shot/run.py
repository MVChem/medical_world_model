"""Persistent single-GPU queue. Resumes only the identical frozen protocol."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback

from .common import PROJECT, atomic, digest, read, verify_contract
from .report import render


def main(args):
    os.umask(0o077)
    run = args.run.resolve()
    coordinator = (run / "queue.lock").open("a")
    fcntl.flock(coordinator, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_contract(run)
    source = Path(__file__).resolve().parents[1]
    if not source.is_relative_to(run):
        raise ValueError("Run the frozen source inside this experiment directory")
    manifest_path = run / "source_manifest.json" if source == run / "source" else source / "execution_manifest.json"
    for name, expected in read(manifest_path).items():
        if digest(source / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    from medworld.gpu import acquire_gpu
    gpu_lock, _ = acquire_gpu(str(args.gpu))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(source / "medworld_zero_shot/startup"), str(source), str(source / "medworld_baselines")]),
        MEDWORLD_PROJECT=str(PROJECT), MEDWORLD_PROJECT_ROOT=str(PROJECT), OMP_NUM_THREADS="4",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_USE_FLASHINFER_SAMPLER="0", NCCL_P2P_DISABLE="1",
        MEDWORLD_VLLM_COMPAT="1")
    if args.cpu_cores:
        os.sched_setaffinity(0, {int(x) for x in args.cpu_cores.split(",")})
    state = read(run / "status.json") if (run / "status.json").exists() else dict(models={})
    state.update(state="running", phase="starting", pid=os.getpid(), gpu=args.gpu,
                 gpu_uuid=os.environ["CUDA_VISIBLE_DEVICES"], started_unix=time.time(), protocol_sha256=digest(run / "protocol.json"))
    state.update(execution_source=str(source), execution_manifest_sha256=digest(manifest_path))
    children = set()
    def stop(process):
        # Also signal surviving workers if the parent has already exited.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if process.poll() is None:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        children.discard(process)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    def refresh():
        while state["state"] == "running":
            try:
                state["updated_unix"] = time.time()
                atomic(run / "status.json", state)
                render(run)
            except Exception:
                traceback.print_exc()
            time.sleep(15)
    thread = threading.Thread(target=refresh, daemon=True)
    thread.start()
    def start(command, name):
        with (run / (name + ".log")).open("a") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=PROJECT)
        children.add(process)
        atomic(run / (name + "_process.json"), dict(pid=process.pid, command=command, gpu=args.gpu, started_unix=time.time()))
        return process
    def execute(module, name, extra=(), timeout=21600):
        process = start([sys.executable, "-m", "medworld_zero_shot."+module, "--run", str(run), *extra], name)
        try:
            code = process.wait(timeout=timeout)
            if code:
                raise RuntimeError(f"{name} returned {code}; see {name}.log")
        finally:
            stop(process)
    failures = []
    try:
        from infer import request
        for model in read(run / "models.json"):
            name = model["id"]
            status = state["models"].setdefault(name, {})
            server = None
            try:
                state["phase"] = name + ": verifying checkpoint"
                model_root = Path(model["path"])
                for weight in model["weight_files"]:
                    if digest(model_root / weight["name"]) != weight["sha256"]:
                        raise ValueError("Checkpoint weights changed since protocol preparation")
                for filename, expected in model["processor_sha256"].items():
                    if digest(model_root / filename) != expected:
                        raise ValueError("Checkpoint configuration or tokenizer changed")
                inference = run / name / "test/status.json"
                if not inference.exists() or read(inference)["status"] != "complete":
                    state["phase"] = name + ": loading"
                    with socket.socket() as sock:
                        sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
                    endpoint = f"http://127.0.0.1:{port}"
                    command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", model["path"],
                        "--served-model-name", "matched-"+name, "--host", "127.0.0.1", "--port", str(port),
                        "--tensor-parallel-size", "1", "--dtype", "bfloat16", "--max-model-len", "8192",
                        "--max-num-seqs", "16", "--max-num-batched-tokens", "4096", "--gpu-memory-utilization", "0.90",
                        "--limit-mm-per-prompt", '{"image":1,"video":0}', "--enforce-eager", "--disable-custom-all-reduce",
                        "--no-enable-log-requests", "--disable-uvicorn-access-log", "--logprobs-mode", "raw_logprobs"]
                    if model["family"] == "qwen":
                        pixels = read(run / "training_config.json")["vision_pixels"] ** 2
                        import json
                        command += ["--reasoning-parser", "qwen3", "--mm-processor-kwargs", json.dumps(dict(min_pixels=pixels, max_pixels=pixels))]
                    server = start(command, name + "_server")
                    deadline = time.monotonic() + 1200
                    while True:
                        if server.poll() is not None:
                            raise RuntimeError(f"{name} server exited: {server.returncode}")
                        try:
                            request(endpoint, route="/v1/models", timeout=5)
                            break
                        except Exception:
                            if time.monotonic() > deadline:
                                raise TimeoutError("Server startup timeout")
                            time.sleep(5)
                    extra = ["--model", name, "--endpoint", endpoint]
                    state["phase"] = name + ": smoke"
                    execute("infer", name + "_smoke", [*extra, "--smoke"], timeout=1800)
                    state["phase"] = name + ": inference"
                    execute("infer", name + "_infer", extra)
                    stop(server); server = None
                    time.sleep(5)
                status["inference"] = "complete"
                if status.get("clinical") != "complete":
                    state["phase"] = name + ": clinical scoring"
                    execute("score", name + "_score", ["--model", name], timeout=7200)
                    status["clinical"] = "complete"
            except Exception:
                status["error"] = traceback.format_exc()
                failures.append(name)
                traceback.print_exc()
            finally:
                if server is not None:
                    stop(server)
        state["phase"] = "GREEN scoring"
        try:
            execute("green", "green", ["--gpu", str(args.gpu)], timeout=21600)
            state["green"] = "complete"
        except Exception:
            state["green"] = "failed"
            state["green_error"] = traceback.format_exc()
            failures.append("green")
        state.update(state="complete" if not failures else "failed", phase="complete" if not failures else "completed with errors", failures=failures)
    except BaseException:
        state.update(state="interrupted", phase="interrupted", error=traceback.format_exc())
        raise
    finally:
        for process in list(children):
            stop(process)
        state["finished_unix"] = time.time()
        thread.join(timeout=16)
        atomic(run / "status.json", state)
        render(run)
        gpu_lock.close()
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--cpu-cores", default="")
    main(p.parse_args())
