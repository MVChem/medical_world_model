"""Run from project root: PYTHONPATH=code python -m medworld.train --help."""
import argparse
from pathlib import Path

from .config import load_config
from .gpu import acquire_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", help="Resume the same run at an optimizer boundary")
    args = parser.parse_args()
    if args.resume and (args.config or args.smoke):
        parser.error("Resume uses its saved config; omit --config and --smoke")
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        parser.error("Choose a new output folder; existing runs are never overwritten")
    if args.resume and Path(args.resume).resolve().parent != out:
        parser.error("Resume --out must be the checkpoint's existing run folder")
    lock, device = acquire_gpu(args.gpu)
    from .datasets import UnifiedData
    from .model import MedWorld
    from .runtime import (Trainer, atomic_json, read_checkpoint, seed_all, source_fingerprint)
    saved = read_checkpoint(args.resume) if args.resume else None
    overrides = {"steps": 4, "accumulation": 1, "total_hours": 0,
                 "report_tokens": 64, "context_tokens": 96, "batch_size": 1,
                 "task_batch_sizes": {}, "generation_tokens": 24,
                 "validation_samples": 1, "save_every": 2} if args.smoke else None
    cfg = saved["config"] if saved else load_config(args.config, overrides)
    if cfg.get("total_hours", 0):
        parser.error("Timed multi-GPU runs use medworld.launch_distributed")
    seed_all(cfg["seed"])
    data = UnifiedData(cfg)
    weights = source_fingerprint(cfg)
    model = MedWorld(cfg, device)
    model.pos_weight.copy_(data.current.pos_weight.to(device))
    trainer = Trainer(model, data, out, weights)
    if args.resume:
        trainer.resume(saved)
    out.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    main()
