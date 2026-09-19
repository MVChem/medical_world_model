"""Run from project root: PYTHONPATH=code python -m medworld.train --help."""
import argparse
from pathlib import Path

from .config import DEFAULTS, load_config
from .gpu import acquire_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--out", required=True)
    parser.add_argument("--stage", choices=("stage1", "stage2", "both"), default="both")
    parser.add_argument("--gpu", default="auto")
    parser.add_argument("--smoke", action="store_true")
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--resume", help="Resume the same run at an optimizer boundary")
    checkpoint.add_argument("--init-checkpoint", help="Completed unified Stage 1 checkpoint for Stage 2")
    args = parser.parse_args()
    if args.resume and (args.config or args.smoke):
        parser.error("Resume uses its saved config; omit --config and --smoke")
    if args.init_checkpoint and args.stage != "stage2":
        parser.error("--init-checkpoint requires --stage stage2")
    if args.stage == "stage2" and not (args.resume or args.init_checkpoint):
        parser.error("Stage 2 requires a completed Stage 1 checkpoint")
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        parser.error("Choose a new output folder; existing runs are never overwritten")
    if args.resume and Path(args.resume).resolve().parent != out:
        parser.error("Resume --out must be the checkpoint's existing run folder")
    lock, device = acquire_gpu(args.gpu)
    from .datasets import UnifiedData
    from .model import MedWorld
    from .runtime import (Trainer, atomic_json, read_checkpoint, seed_all, source_fingerprint)
    saved = read_checkpoint(args.resume or args.init_checkpoint) if args.resume or args.init_checkpoint else None
    overrides = {"stage1_steps": 4, "stage2_steps": 4, "stage1_accumulation": 1,
                 "stage2_accumulation": 1, "report_tokens": 64, "context_tokens": 96,
                 "generation_tokens": 24, "validation_samples": 1, "save_every": 2} if args.smoke else None
    cfg = (saved["config"] if args.resume or args.init_checkpoint and not args.config and not args.smoke
           else load_config(args.config, overrides))
    if args.smoke and cfg.get("spatial_decoder") == "featup":
        # Exercise classification, report, segmentation and SR replay in Stage 2.
        cfg["stage2_steps"] = 16
    if cfg.get("total_hours", 0):
        parser.error("Timed multi-GPU runs use medworld.launch_distributed")
    if args.init_checkpoint and cfg != saved["config"]:
        # Preserve the architecture and protocol; only Stage 2 run budgets may change.
        allowed = {"stage2_steps", "stage2_accumulation", "save_every", "validate_every", "validation_samples"}
        if any(cfg[k] != saved["config"].get(k, DEFAULTS[k]) for k in cfg if k not in allowed):
            parser.error("Stage 1 transfer requires matching config except Stage 2/checkpoint budgets")
    seed_all(cfg["seed"])
    data = UnifiedData(cfg)
    weights = source_fingerprint(cfg)
    model = MedWorld(cfg, device)
    model.pos_weight.copy_(data.current.pos_weight.to(device))
    trainer = Trainer(model, data, out, weights)
    if args.resume:
        trainer.resume(saved)
        if args.stage == "stage1" and trainer.progress["stage"] != "stage1":
            parser.error("Cannot resume Stage 2 as Stage 1")
    elif args.init_checkpoint:
        if saved["data_fingerprint"] != data.fingerprint or saved["weights_fingerprint"] != weights:
            parser.error("Stage 1 transfer data/weights differ")
        if saved["progress"]["stage"] != "stage1" or not saved["progress"]["stage_complete"]:
            parser.error("Expected a completed Stage 1 checkpoint")
        model.restore(saved["model"])
        trainer.progress = saved["progress"]
        trainer.enter_stage2()
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "config.json", cfg)
    atomic_json(out / "data_protocol.json", data.metadata)
    atomic_json(out / "model.json", model.metadata)
    trainer.install_signals()
    if trainer.progress["stage"] == "stage1":
        complete = trainer.run_stage()
        if complete and args.stage in ("both", "stage2") and not trainer.stop:
            trainer.enter_stage2()
    if trainer.progress["stage"] == "stage2" and not trainer.stop:
        trainer.run_stage()
    if args.smoke and trainer.progress["stage"] == "stage2" and trainer.progress["stage_complete"]:
        from .smoke import audit_model
        atomic_json(out / "smoke_audit.json", audit_model(model, data, out))
    atomic_json(out / "status.json", {**trainer.progress, "stopped": trainer.stop})
    # Keep the file descriptor alive throughout training and smoke checks.
    if lock is not None:
        lock.close()


if __name__ == "__main__":
    main()
