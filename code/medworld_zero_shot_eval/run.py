"""Own one GPU for a resumable serial four-model pilot, then release it."""

import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .common import atomic, digest, read, request, verify


def main(args):
    os.umask(0o077)
    run = args.run.resolve()
    lock = (run / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify(run)
    if read(run / "protocol.json").get("prompt_style") in {
        "question_only",
        "llava_short_answer",
    }:
        from .literature_score import render, score
    else:
        from .score import render, score
    source = Path(__file__).resolve().parents[1]
    if source != run / "source":
        raise ValueError("Launch the frozen source inside the run directory")
    from .gpu import acquire_gpu

    gpu_lock, _ = acquire_gpu(str(args.gpu))
    if args.cpu_cores:
        os.sched_setaffinity(0, {int(x) for x in args.cpu_cores.split(",")})
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(source / "medworld_vqa/startup"), str(source)]),
        MEDWORLD_VLLM_COMPAT="1",
        OMP_NUM_THREADS="4",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_USE_FLASHINFER_SAMPLER="0",
        NCCL_P2P_DISABLE="1",
    )
    state = {
        "status": "running",
        "pid": os.getpid(),
        "gpu": args.gpu,
        "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"],
        "started_unix": time.time(),
        "models": {},
        "source_manifest_sha256": digest(run / "source_manifest.json"),
    }
    children = set()

    def save():
        state["updated_unix"] = time.time()
        atomic(run / "status.json", state)
        render(run)

    def start(command, name):
        with (run / (name + ".log")).open("a") as log:
            process = subprocess.Popen(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        children.add(process)
        atomic(
            run / (name + "_process.json"),
            {"pid": process.pid, "command": command, "started_unix": time.time()},
        )
        return process

    def stop(process):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        children.discard(process)

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    save()
    try:
        for model in read(run / "models.json"):
            name = model["id"]
            server = worker = None
            state.update(phase=name + ": load")
            state["models"][name] = "running"
            save()
            try:
                for filename, expected in model["file_sha256"].items():
                    if digest(Path(model["path"]) / filename) != expected:
                        raise ValueError("Model weights/tokenizer/config changed")
                path = run / name / "status.json"
                if not path.exists() or read(path)["status"] != "complete":
                    with socket.socket() as sock:
                        sock.bind(("127.0.0.1", 0))
                        port = sock.getsockname()[1]
                    endpoint = f"http://127.0.0.1:{port}"
                    command = [
                        sys.executable,
                        "-m",
                        "vllm.entrypoints.cli.main",
                        "serve",
                        model["path"],
                        "--served-model-name",
                        "cxrvqa-" + name,
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(port),
                        "--tensor-parallel-size",
                        "1",
                        "--dtype",
                        "bfloat16",
                        "--max-model-len",
                        "4096",
                        "--max-num-seqs",
                        "8",
                        "--max-num-batched-tokens",
                        "2048",
                        "--gpu-memory-utilization",
                        "0.90",
                        "--limit-mm-per-prompt",
                        '{"image":1,"video":0}',
                        "--enforce-eager",
                        "--disable-custom-all-reduce",
                        "--no-enable-log-requests",
                        "--disable-uvicorn-access-log",
                    ]
                    if model["family"] == "qwen":
                        command += [
                            "--reasoning-parser",
                            "qwen3",
                            "--mm-processor-kwargs",
                            json.dumps({"min_pixels": 512**2, "max_pixels": 512**2}),
                        ]
                    server = start(command, name + "_server")
                    deadline = time.monotonic() + 1200
                    while True:
                        if server.poll() is not None:
                            raise RuntimeError(
                                f"Server exited {server.returncode}; inspect {name}_server.log"
                            )
                        try:
                            request(endpoint, route="/v1/models", timeout=3)
                            break
                        except Exception:  # noqa: BLE001 -- report failures and clean up owned workers
                            if time.monotonic() > deadline:
                                raise TimeoutError("Server startup exceeded 20 minutes")
                            time.sleep(3)
                    state.update(phase=name + ": inference")
                    save()
                    worker = start(
                        [
                            sys.executable,
                            "-m",
                            "medworld_vqa.infer",
                            "--run",
                            str(run),
                            "--model",
                            name,
                            "--endpoint",
                            endpoint,
                        ],
                        name + "_infer",
                    )
                    deadline = time.monotonic() + 3600
                    while worker.poll() is None:
                        if server.poll() is not None:
                            raise RuntimeError("Server died during inference")
                        if time.monotonic() > deadline:
                            raise TimeoutError(
                                "Per-model one-hour inference budget exceeded"
                            )
                        save()
                        time.sleep(15)
                    if worker.returncode:
                        raise RuntimeError(f"Inference exited {worker.returncode}")
                    stop(worker)
                    worker = None
                    stop(server)
                    server = None
                    time.sleep(5)
                score(run, name)
                state["models"][name] = "complete"
            except Exception:  # noqa: BLE001 -- report failures and clean up owned workers
                state["models"][name] = "failed"
                state.setdefault("errors", {})[name] = traceback.format_exc()
                traceback.print_exc()
            finally:
                for process in (worker, server):
                    if process is not None:
                        stop(process)
                save()
        state.update(
            status="complete"
            if all(s == "complete" for s in state["models"].values())
            else "failed",
            phase="finished",
        )
    except BaseException:
        state.update(status="interrupted", error=traceback.format_exc())
        raise
    finally:
        for process in list(children):
            stop(process)
        state["finished_unix"] = time.time()
        save()
        gpu_lock.close()
    if state["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--cpu-cores", default="0,1,2,3,4,5,6,7")
    main(p.parse_args())
