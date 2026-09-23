"""Offline CPU transport for official RadGraph under its isolated dependency pins.

The parent uses the Qwen training environment. Only this child process imports
the asset-local Transformers 4 overlay; the training environment stays intact.
Stdout contains JSON protocol messages only. Reports and annotations stay in RAM.
"""
from contextlib import redirect_stdout
import importlib.metadata
import json
import sys
import traceback


def emit(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()


def main():
    try:
        request = json.loads(sys.stdin.readline())
        actual = {name: importlib.metadata.version(name) for name in request["versions"]}
        if actual != request["versions"]:
            raise ValueError(f"Worker dependency versions differ: {actual}")
        with redirect_stdout(sys.stderr):
            import torch
            from .future_metrics import OfficialRadGraphScorer
            torch.set_num_threads(4)
            scorer = OfficialRadGraphScorer(request["provenance"],
                model_cache_dir=request["model_cache_dir"],
                tokenizer_cache_dir=request["tokenizer_cache_dir"], cuda=-1)
        emit({"state": "ready", "versions": actual})
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        emit({"state": "failed", "error": f"{type(exc).__name__}: {exc}"})
        return
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if (set(request) != {"refs", "hyps"} or not isinstance(request["refs"], list)
                    or not isinstance(request["hyps"], list) or not 1 <= len(request["refs"]) <= 8
                    or len(request["refs"]) != len(request["hyps"])
                    or any(not isinstance(text, str) for text in request["refs"] + request["hyps"])):
                raise ValueError("Worker expects at most eight matched report strings")
            with redirect_stdout(sys.stderr), torch.inference_mode():
                scores = scorer.score(request["refs"], request["hyps"])
            emit({"state": "ok", "scores": scores})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            emit({"state": "failed", "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
