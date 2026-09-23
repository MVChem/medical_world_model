"""Run from project root: PYTHONPATH=code python -m medworld.train --help."""
import argparse
from pathlib import Path

from . import WEIGHTS_ONLY_RESUME_ERROR
from .config import load_config
from .architecture import require_reviewed_data
from .gpu import acquire_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", help="Unavailable: checkpoints contain trained weights only")
    args = parser.parse_args()
    if args.resume:
        parser.error(WEIGHTS_ONLY_RESUME_ERROR)
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Choose a new output folder; existing runs are never overwritten")
    lock, device = acquire_gpu(args.gpu)
    from .datasets import UnifiedData
    from .model import MedWorld
    from .runtime import (Trainer, atomic_json, seed_all, source_fingerprint)
    overrides = {"steps": 3, "accumulation": 1, "total_hours": 0,
                 "answer_tokens": 64, "context_tokens": 96, "batch_size": 1,
                 "task_batch_sizes": {}, "generation_tokens": 24,
                 "validation_samples": 1, "save_every": 2} if args.smoke else None
    cfg = load_config(args.config, overrides)
    if cfg.get("total_hours", 0):
        parser.error("Timed multi-GPU runs use medworld.launch_distributed")
    seed_all(cfg["seed"])
    data = UnifiedData(cfg)
    require_reviewed_data(cfg, data)
    weights = source_fingerprint(cfg)
    model = MedWorld(cfg, device)
    model.pos_weight.copy_(data.current.pos_weight.to(device))
    trainer = Trainer(model, data, out, weights)
    out.mkdir(parents=True, exist_ok=True)
    from .launch_distributed import snapshot
    snapshot(out)
    atomic_json(out / "config.json", cfg)
    atomic_json(out / "data_protocol.json", data.metadata)
    atomic_json(out / "model.json", model.metadata)
    trainer.install_signals()
    trainer.run()
    if args.smoke and trainer.progress["complete"]:
        from .smoke import audit_model
        atomic_json(out / "smoke_audit.json", audit_model(model, data, out))
    atomic_json(out / "status.json", {**trainer.progress, "stopped": trainer.stop})
    # Keep the file descriptor alive throughout training and smoke checks.
    if lock is not None:
        lock.close()
    if not args.smoke and trainer.progress["complete"] and not trainer.stop:
        import gc
        import torch
        del trainer, model, data
        gc.collect()
        torch.cuda.empty_cache()
        from .evaluate_run import evaluate_run
        from .gpu import ALLOWED_GPUS
        evaluate_run(out, list(ALLOWED_GPUS) if args.gpu == "auto" else [args.gpu])


if __name__ == "__main__":
    main()
