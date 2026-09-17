"""Evaluate a finished unified checkpoint under the same frozen baseline contract."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from .common import atomic, digest, read, verify_contract


def main(args):
    run, out = args.run.resolve(), args.out.resolve()
    protocol = verify_contract(run)
    training = Path(protocol["training_run"])
    env = dict(os.environ, PYTHONPATH=str(training / "source"))
    command = [sys.executable, "-m", "medworld.evaluate", "--checkpoint", str(args.checkpoint.resolve()),
               "--out", str(out), "--task", "all", "--split", "test", "--max-new-tokens", "384", "--gpu", args.gpu]
    subprocess.run(command, env=env, check=True)
    summary = read(out / "summary.json")
    if summary["data_fingerprint"] != protocol["data_fingerprint"]:
        raise ValueError("Checkpoint data protocol differs from the baseline test contract")
    atomic(out / "zero_shot_comparison.json", dict(protocol_sha256=digest(run / "protocol.json"),
        max_new_tokens=384, checkpoint_sha256=summary["checkpoint_sha256"], command=command))
    from medworld.gpu import acquire_gpu
    lock, _ = acquire_gpu(args.gpu)
    try:
        subprocess.run([sys.executable, "-m", "medworld_zero_shot.score", "--run", str(run),
                        "--model", args.name, "--ours", str(out)], check=True)
        # The adapted GREEN runner preserves this wrapper's reserved UUID.
        subprocess.run([sys.executable, "-m", "medworld_zero_shot.green", "--run", str(run),
                        "--models", args.name, "--gpu", "0"], check=True)
    finally:
        lock.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--name", default="ours")
    p.add_argument("--gpu", default="auto")
    main(p.parse_args())
