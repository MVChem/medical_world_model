"""Start a current Stage 1 run with configuration-selected readout and frozen sources."""

import argparse
import json
from pathlib import Path
from bootstrap import atomic_json
from training_variants import resolve_variant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--train-gpu", type=int, required=True)
    parser.add_argument("--eval-gpu", type=int, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    variant = resolve_variant(cfg)
    if variant == "legacy":
        parser.error(
            "Use train.py for legacy report-generation experiments; see README.md."
        )
    if args.train_gpu == args.eval_gpu:
        parser.error("Training and concurrent preview require different GPUs.")
    if args.run.exists():
        parser.error("Choose a new run directory; existing runs are never overwritten.")
    cfg["preview_gpu"] = args.eval_gpu
    args.run.mkdir(parents=True)
    atomic_json(args.run / "config.json", cfg)
    args.config = args.run / "config.json"
    if variant == "slot44":
        from slot44_runner import main as run
    else:
        from noslots_runner import main as run
    run(args)


if __name__ == "__main__":
    main()
