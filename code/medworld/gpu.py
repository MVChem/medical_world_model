"""Cooperate with the machine's existing per-GPU experiment locks."""
import fcntl
import os
import subprocess

ALLOWED_GPUS = ("1", "2", "3", "6", "7", "0")


def acquire_gpu(selector):
    if selector == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return None, "cpu"
    if selector == "auto":
        candidates = ALLOWED_GPUS
    else:
        if selector not in ALLOWED_GPUS:
            raise ValueError(f"GPU selector must be auto, cpu, or one of {ALLOWED_GPUS}; GPUs 4/5 are forbidden")
        candidates = [selector]
    for candidate in candidates:
        try:
            def status():
                line = subprocess.check_output(["nvidia-smi", "-i", candidate,
                    "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL, text=True, timeout=10).strip()
                uuid, memory, utilization = [x.strip() for x in line.split(",")]
                active = subprocess.check_output(["nvidia-smi", "-i", candidate,
                    "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL, text=True, timeout=10).strip()
                return uuid, int(memory), int(utilization), bool(active)
            uuid, memory, utilization, active = status()
            if active or memory > 1024 or utilization > 5:
                continue
            lock = open(f"/tmp/medworld-frozen-slots-{uuid}.lock", "a")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                _, memory, utilization, active = status()
                if active or memory > 1024 or utilization > 5:
                    lock.close()
                    continue
            except BlockingIOError:
                lock.close()
                continue
            os.environ["CUDA_VISIBLE_DEVICES"] = uuid
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            print(f"GPU acquired: {uuid}", flush=True)
            return lock, "cuda"
        except (subprocess.SubprocessError, ValueError, OSError):
            continue
    raise RuntimeError(f"No idle unlocked GPU for selector {selector}")
