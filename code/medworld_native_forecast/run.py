"""Matched native-image forecasting, resumable at complete optimizer updates.

Stage 1 is trained once with slots. Stage 2 branches load that exact checkpoint,
reset their optimizer, and consume the same deterministic (step, micro) stream.
Smoke checkpoints are execution checks and cannot initialize formal training.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import torch

FORMAT_VERSION = 1
STOP = False
HERE = Path(__file__).resolve().parent


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_torch(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def tensor_digest(values):
    hasher = hashlib.sha256()
    for name, value in sorted(values.items()):
        tensor = value.detach().cpu().contiguous()
        hasher.update(name.encode())
        hasher.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
        hasher.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def decoder_digest(model):
    state = model.compact_state()
    selected = {name: value for name, value in state.items()
                if name.startswith(("decoder.", "finding."))}
    if not selected:
        raise ValueError("No shared decoder/finding checkpoint parameters found")
    return tensor_digest(selected)


def target_digest(model):
    target = getattr(model, "target_encoder", None)
    if target is None:
        return None
    if any(parameter.requires_grad for parameter in target.parameters()):
        raise RuntimeError("Target encoder is not frozen")
    state = {name: value for name, value in model.compact_state().items()
             if name.startswith("target_encoder.")}
    if not state:
        raise RuntimeError("Frozen target trainables are missing from compact checkpoint")
    return tensor_digest(state)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if torch.device(device).type == "cuda" else nullcontext()


def memory(device):
    if torch.device(device).type != "cuda":
        return dict(device=device)
    return dict(device=device, allocated_gib=torch.cuda.memory_allocated(device) / 2**30,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30)


def optimizer_for(model, cfg):
    groups = {"lora": [], "other": []}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            groups["lora" if "lora_" in name else "other"].append(parameter)
    rates = {"lora": cfg["lora_learning_rate"], "other": cfg["learning_rate"]}
    return torch.optim.AdamW([dict(params=values, name=name, lr=rates[name], base_lr=rates[name])
                             for name, values in groups.items() if values],
                            betas=(0.9, 0.95), weight_decay=0.01)


def accumulation_for(cfg, stage):
    return cfg.get("stage1_gradient_accumulation", 8 // cfg["batch_size"]) if stage == 1 else cfg["gradient_accumulation"]


def stop_handler(signum, frame):
    global STOP
    STOP = True
    print(json.dumps(dict(event="stop_requested", signal=signum)), flush=True)


def prepare_output(args):
    out = Path(args.out).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        if not args.resume or Path(args.resume).resolve().parent != out:
            raise FileExistsError(f"Populated output requires its own --resume checkpoint: {out}")
    if args.resume and not Path(args.resume).is_file():
        raise FileNotFoundError(args.resume)
    out.mkdir(parents=True, exist_ok=True)
    return out


def effective_config(args):
    cfg = json.loads(Path(args.config).read_text())
    for name in ("batch_size", "gradient_accumulation", "seed"):
        value = getattr(args, name)
        if value is not None:
            cfg[name] = value
    cfg["state_condition"] = args.condition
    cfg.setdefault("sampling", "permutation")
    if cfg["sampling"] != "permutation":
        raise ValueError("Matched forecasting requires sampling=permutation")
    if args.smoke:
        cfg["generation_tokens"] = min(cfg.get("generation_tokens", 384), 8)
        cfg["batch_size"] = 1
        cfg["gradient_accumulation"] = 1
        cfg["stage1_gradient_accumulation"] = 1
    if type(cfg.get("batch_size")) is not int or cfg["batch_size"] not in (1, 2):
        raise ValueError("Validated microbatch sizes are integer 1 or 2")
    cfg.setdefault("stage1_gradient_accumulation", 8 // cfg["batch_size"])
    for key in ("gradient_accumulation", "stage1_gradient_accumulation", "learning_rate", "lora_learning_rate"):
        if not isinstance(cfg.get(key), (int, float)) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    for key in ("gradient_accumulation", "stage1_gradient_accumulation"):
        if type(cfg[key]) is not int:
            raise ValueError(f"{key} must be a whole number of microbatches")
    return cfg


def transfer_compatible(checkpoint, cfg, smoke):
    if checkpoint.get("format_version") != FORMAT_VERSION or checkpoint["stage"] != 1:
        raise ValueError("Expected this runner's shared Stage-1 checkpoint")
    if not checkpoint.get("completed_stage"):
        raise ValueError("Shared Stage-1 checkpoint has not finished its configured budget")
    if bool(checkpoint.get("smoke_only")) != bool(smoke):
        raise ValueError("Smoke and formal checkpoints cannot be mixed")
    # Representation and optimizer branch may differ; input/model settings may not.
    ignored = {"state_condition", "max_stage1_steps", "max_stage2_steps", "stage1_only", "initialize_stage1"}
    previous = {key: value for key, value in checkpoint["config"].items() if key not in ignored}
    current = {key: value for key, value in cfg.items() if key not in ignored}
    if previous != current:
        changed = sorted(key for key in previous.keys() | current.keys() if previous.get(key) != current.get(key))
        raise ValueError(f"Shared Stage-1 configuration differs: {changed}")


def data_signature(corpus, cfg):
    if hasattr(corpus, "metadata"):
        return corpus.metadata
    root = Path(cfg["cache"])
    return {name: digest(root / name) for name in
            ("manifest.json", "observations.jsonl", "train.jsonl", "validate.jsonl", "test.jsonl")}


def runtime_signature(cfg):
    """Pin executable sources and frozen weight files without rereading 18 GB."""
    weights = {}
    if cfg.get("qwen"):
        root = Path(cfg["qwen"]).resolve()
        index = root / "model.safetensors.index.json"
        files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else [p.name for p in sorted(root.glob("*.safetensors"))]
        if not files:
            raise FileNotFoundError(f"No declared Qwen weight shards in {root}")
        weights["qwen"] = dict(path=str(root), config_sha256=digest(root / "config.json"),
            index_sha256=digest(index) if index.exists() else None,
            shards={name: dict(bytes=(root / name).stat().st_size, mtime_ns=(root / name).stat().st_mtime_ns)
                    for name in files}, weight_content_sha256_recomputed=False,
            identity_note="Local immutable snapshot path, config/index SHA256, and every shard size/mtime; no new whole-weight content hash")
    if cfg.get("vjepa_checkpoint"):
        path = Path(cfg["vjepa_checkpoint"]).resolve()
        weights["vjepa"] = dict(path=str(path), bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns,
                                role="Frozen cached feature provenance; not loaded by native decoder")
    return dict(source_sha256={name: digest(HERE / name) for name in ("model.py", "data.py", "run.py")},
                pretrained=weights)


def check_runtime(checkpoint, current, *, smoke=False):
    previous = checkpoint.get("runtime_signature")
    if previous is None:
        if smoke:
            return "legacy smoke checkpoint has no executable/pretrained signature"
        raise ValueError("Checkpoint lacks executable/pretrained signatures; use a fresh formal run")
    if previous != current:
        raise ValueError("Checkpoint source code or pretrained file identity changed; refusing silent transfer/resume")
    return None


@torch.no_grad()
def validate(model, corpus, cfg, stage, device):
    saved_rng, was_training = rng_state(), model.training
    model.eval()
    totals, count = {}, 0
    try:
        rows = corpus.pairs["validate"][:cfg.get("validation_pairs", 32)]
        for row in rows:
            if STOP:
                break
            batch = corpus.batch([row], device=device)
            with autocast(device):
                loss, parts = model.losses(batch, stage)
            for name, value in dict(loss=loss, **parts).items():
                scalar = float(value.detach())
                if not math.isfinite(scalar):
                    raise FloatingPointError(f"Nonfinite validation {name}")
                totals[name] = totals.get(name, 0.0) + scalar
            count += 1
        return dict(samples=count, losses={key: value / count for key, value in totals.items()} if count else {})
    finally:
        model.train(was_training)
        restore_rng(saved_rng)


@torch.no_grad()
def smoke_checks(model, corpus, cfg, device, checkpoint_path, target_initial):
    from data import source_view
    model.eval()
    rows = corpus.pairs["validate"][:1]
    # Inference loads only source pixels and source evidence; future labels/reports
    # are absent. Poisoning all training-only fields must not alter this view.
    source_rows = [{key: value for key, value in row.items() if key != "target"} for row in rows]
    source = corpus.batch(source_rows, device=device, source_only=True)
    batch = corpus.batch(rows, device=device)
    if any(key.startswith("target") or key == "_target_images" for key in source):
        raise RuntimeError("Future fields leaked into inference input")
    prediction_rng = rng_state()
    with autocast(device):
        reports, scores = model.predict(source)
    before_scores = scores.detach().float().cpu().clone()
    if not torch.isfinite(before_scores).all() or bool(((before_scores < 0) | (before_scores > 1)).any()):
        raise FloatingPointError("Finding outputs must be finite probabilities")
    poisoned = dict(batch)
    poisoned_fields = []
    for key, value in batch.items():
        if key not in source:
            poisoned_fields.append(key)
            poisoned[key] = torch.full_like(value, -999) if isinstance(value, torch.Tensor) else "FUTURE_SENTINEL"
    restore_rng(prediction_rng)
    with autocast(device):
        other_reports, other_scores = model.predict(source_view(poisoned))
    if reports != other_reports:
        raise RuntimeError("Reports changed after discarded future targets were perturbed")
    torch.testing.assert_close(other_scores.float().cpu(), before_scores, rtol=1e-5, atol=1e-6)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected = checkpoint["model"]
    # Corrupt a real trainable so this exercises restoration rather than a no-op.
    first = next(parameter for parameter in model.parameters() if parameter.requires_grad)
    first.add_(0.125)
    model.load_compact(expected)
    actual = model.compact_state()
    if actual.keys() != expected.keys() or not all(torch.equal(actual[key].cpu(), value.cpu()) for key, value in expected.items()):
        raise RuntimeError("Compact checkpoint did not restore every saved tensor exactly")
    restore_rng(prediction_rng)
    with autocast(device):
        reloaded_reports, reloaded_scores = model.predict(source)
    if reports != reloaded_reports:
        raise RuntimeError("Report generation changed after checkpoint reload")
    torch.testing.assert_close(reloaded_scores.float().cpu(), before_scores, rtol=1e-5, atol=1e-6)
    target_final = target_digest(model)
    if target_initial != target_final:
        raise RuntimeError("Frozen target changed across Stage-2 training/checkpoint reload")
    return dict(smoke_only=True, table1_ready=False, checkpoint_reload_exact=True,
                reports_equal_after_reload=True, probabilities_close_after_reload=True,
                future_perturbation_equal=True, perturbation_scope="discarded training-only fields",
                poisoned_fields=poisoned_fields, prediction_fields=sorted(source),
                count=len(reports), generated_character_counts=[len(text) for text in reports],
                generation_tokens=cfg["generation_tokens"], target_frozen_hash=target_final,
                target_unchanged=True, memory=memory(device))


def run(args):
    global STOP
    STOP = False
    os.umask(0o077)
    out, cfg = prepare_output(args), effective_config(args)
    stage_mode = args.stage
    if stage_mode == "both" and args.condition != "slots":
        stage_mode = "stage2"
    if stage_mode == "stage1" and args.condition != "slots":
        raise ValueError("Stage 1 is shared and trained once with condition=slots")
    if stage_mode == "stage2" and not (args.init_checkpoint or args.resume):
        raise ValueError("Stage 2 requires an explicit shared --init-checkpoint or --resume")
    budgets = {1: 1 if args.smoke else cfg.get("max_stage1_steps", 1694),
               2: 1 if args.smoke else cfg.get("max_stage2_steps", 2400)}
    if args.steps is not None:
        if args.smoke or stage_mode == "both":
            raise ValueError("--steps requires one explicit stage and cannot be used with --smoke")
        budgets[1 if stage_mode == "stage1" else 2] = args.steps
    for budget in budgets.values():
        if not isinstance(budget, int) or budget <= 0:
            raise ValueError("Stage budgets must be positive integer optimizer updates")
    stage, stage_step, elapsed = (2 if stage_mode == "stage2" else 1), 0, 0.0
    started = time.monotonic()
    model = corpus = optimizer = None
    target_initial, decoder_initial, last_validation = None, None, {}
    execution_signature = runtime_signature(cfg)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop_handler)

    def status(state, **extra):
        value = dict(state=state, pid=os.getpid(), stage=stage, stage_step=stage_step,
                     stage_budget=budgets[stage], condition=args.condition, smoke_only=args.smoke,
                     table1_ready=False, elapsed_seconds=elapsed + time.monotonic() - started,
                     updated_unix=time.time(), **extra)
        atomic_json(out / "status.json", value)
        print(json.dumps(value, allow_nan=False), flush=True)

    def save(name="checkpoint_latest.pt", completed=False):
        atomic_torch(out / name, dict(format_version=FORMAT_VERSION, model=model.compact_state(),
            model_metadata=model.metadata, data_metadata=data_signature(corpus, cfg),
            optimizer=optimizer.state_dict(), config=cfg, condition=args.condition,
            stage=stage, stage_step=stage_step, stage_mode=stage_mode, budgets=budgets,
            sampler=dict(kind="stateless_step_micro_permutation", next_step=stage_step,
                         stage=stage, seed=cfg["seed"], microbatch=cfg["batch_size"],
                         accumulation=accumulation_for(cfg, stage)), rng=rng_state(),
            elapsed_seconds=elapsed + time.monotonic() - started, smoke_only=args.smoke,
            completed_stage=completed, target_initial_sha256=target_initial,
            decoder_initial_sha256=decoder_initial, last_validation=last_validation,
            runtime_signature=execution_signature))

    try:
        status("initializing")
        seed_all(cfg["seed"])
        torch.set_num_threads(args.threads)
        from data import Corpus
        from model import NativeForecast
        model = NativeForecast(cfg, condition=args.condition, device=args.device)
        corpus = Corpus(cfg, model.tokenizer)
        if args.init_checkpoint:
            checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
            transfer_compatible(checkpoint, cfg, args.smoke)
            legacy_note = check_runtime(checkpoint, execution_signature, smoke=args.smoke)
            if checkpoint["data_metadata"] != data_signature(corpus, cfg):
                raise ValueError("Shared Stage-1 data provenance differs")
            model.load_compact(checkpoint["model"])
            decoder_initial = decoder_digest(model)
            expected_decoder = tensor_digest({key: value for key, value in checkpoint["model"].items()
                                               if key.startswith(("decoder.", "finding."))})
            if decoder_initial != expected_decoder:
                raise RuntimeError("Shared decoder initialization differs after Stage-1 transfer")
            model.begin_stage2()
            target_initial = target_digest(model)
            atomic_json(out / "initialization.json", dict(checkpoint=str(Path(args.init_checkpoint).resolve()),
                sha256=digest(args.init_checkpoint), shared_stage1_updates=checkpoint["stage_step"],
                decoder_sha256=decoder_initial, optimizer_reset=True, signature_note=legacy_note))
            restore_rng(checkpoint["rng"])
            del checkpoint
        elif stage == 2:
            model.begin_stage2()
        optimizer = optimizer_for(model, cfg)
        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
            if checkpoint["format_version"] != FORMAT_VERSION or checkpoint["config"] != cfg:
                raise ValueError("Resume configuration differs")
            check_runtime(checkpoint, execution_signature)
            if checkpoint["smoke_only"] != args.smoke or checkpoint["stage_mode"] != stage_mode:
                raise ValueError("Resume stage/mode differs; smoke cannot become formal training")
            if checkpoint["data_metadata"] != data_signature(corpus, cfg):
                raise ValueError("Resume data provenance differs")
            stage = checkpoint["stage"]
            if stage == 2 and getattr(model, "target_encoder", None) is None:
                model.begin_stage2()
            model.load_compact(checkpoint["model"])
            optimizer = optimizer_for(model, cfg)
            optimizer.load_state_dict(checkpoint["optimizer"])
            stage_step, elapsed = checkpoint["stage_step"], checkpoint["elapsed_seconds"]
            if checkpoint["sampler"]["next_step"] != stage_step or checkpoint["sampler"]["stage"] != stage:
                raise ValueError("Resume sampler cursor does not match optimizer state")
            target_initial, decoder_initial = checkpoint["target_initial_sha256"], checkpoint["decoder_initial_sha256"]
            last_validation = checkpoint["last_validation"]
            restore_rng(checkpoint["rng"])
            del checkpoint
            if stage_step > budgets[stage]:
                raise ValueError("Checkpoint exceeds requested total stage budget")
        if decoder_initial is None:
            decoder_initial = decoder_digest(model)
        atomic_json(out / "config.json", cfg)
        atomic_json(out / "provenance.json", dict(model=model.metadata, data=data_signature(corpus, cfg),
            source_hashes={path.name: digest(path) for path in sorted(HERE.glob("*.py"))},
            effective_batch_size={str(s): cfg["batch_size"] * accumulation_for(cfg, s) for s in (1, 2)},
            budgets=budgets, stage_mode=stage_mode, decoder_initial_sha256=decoder_initial,
            target_initial_sha256=target_initial, torch=str(torch.__version__), python=sys.version,
            cuda=torch.version.cuda, smoke_only=args.smoke,
            checkpoint_scope="trainable parameters plus frozen target adapters/readouts; pretrained frozen base by reference"))
        atomic_json(out / "execution_signature.json", execution_signature)
        if torch.device(args.device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(args.device)
        model.train()
        save()
        with (out / "metrics.jsonl").open("a", buffering=1) as history:
            while not STOP:
                if stage_step >= budgets[stage]:
                    save(f"checkpoint_stage{stage}.pt", completed=True)
                    if stage == 1 and stage_mode == "both":
                        decoder_initial = decoder_digest(model)
                        model.begin_stage2()
                        target_initial = target_digest(model)
                        stage, stage_step = 2, 0
                        optimizer = optimizer_for(model, cfg)
                        save()
                        continue
                    break
                if args.max_seconds and time.monotonic() - started >= args.max_seconds:
                    STOP = True
                    break
                tick = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                accumulation = accumulation_for(cfg, stage)
                corpus.cfg["gradient_accumulation"] = accumulation
                multiplier = min(1.0, (stage_step + 1) / max(1, cfg.get("warmup_steps", 100)))
                for group in optimizer.param_groups:
                    group["lr"] = group["base_lr"] * multiplier
                totals = {}
                for micro in range(accumulation):
                    batch = corpus.training_batch(stage, stage_step, micro, device=args.device)
                    with autocast(args.device):
                        loss, parts = model.losses(batch, stage)
                    if loss.numel() != 1 or not torch.isfinite(loss):
                        raise FloatingPointError("Training loss must be a finite scalar")
                    for name, value in dict(loss=loss, **parts).items():
                        scalar = float(value.detach())
                        if not math.isfinite(scalar):
                            raise FloatingPointError(f"Nonfinite training component {name}")
                        totals[name] = totals.get(name, 0.0) + scalar / accumulation
                    (loss / accumulation).backward()
                gradients = model.gradient_audit(stage)
                trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
                norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                if not float(norm) > 0:
                    raise RuntimeError("No nonzero training gradients")
                optimizer.step()
                stage_step += 1
                if torch.device(args.device).type == "cuda":
                    torch.cuda.synchronize(args.device)
                event = dict(event="train_step", stage=stage, stage_step=stage_step,
                    losses=totals, grad_norm=float(norm), gradient_audit=gradients,
                    seconds=time.monotonic() - tick, effective_batch_size=cfg["batch_size"] * accumulation,
                    lr_multiplier=multiplier, smoke_only=args.smoke, memory=memory(args.device))
                history.write(json.dumps(event, allow_nan=False) + "\n")
                print(json.dumps(event, allow_nan=False), flush=True)
                every = cfg.get("validation_every_steps", 800)
                if not args.smoke and not STOP and every and stage_step % every == 0:
                    last_validation = validate(model, corpus, cfg, stage, args.device)
                    history.write(json.dumps(dict(event="validation", stage=stage, stage_step=stage_step,
                                                  metrics=last_validation)) + "\n")
                    atomic_json(out / "validation_latest.json", dict(stage=stage, stage_step=stage_step, **last_validation))
                if stage_step % args.save_every == 0 or STOP:
                    save()
                status("running", last_loss=totals["loss"], memory=memory(args.device))
        save(completed=stage_step >= budgets[stage])
        if STOP:
            status("interrupted", checkpoint=str(out / "checkpoint_latest.pt"))
            return 130
        save("checkpoint_final.pt", completed=True)
        ending_signature = runtime_signature(cfg)
        atomic_json(out / "execution_signature_end.json", dict(signature=ending_signature,
                    files_unchanged_during_run=ending_signature == execution_signature,
                    initial_signature=execution_signature))
        if args.smoke and stage == 2:
            checks = smoke_checks(model, corpus, cfg, args.device, out / "checkpoint_final.pt", target_initial)
            checks["decoder_initial_sha256"] = decoder_initial
            atomic_json(out / "smoke_checks.json", checks)
        status("smoke_complete" if args.smoke else "budget_complete", checkpoint=str(out / "checkpoint_final.pt"),
               memory=memory(args.device))
        return 0
    except BaseException as error:
        status("failed", error_type=type(error).__name__, error=str(error),
               resume_checkpoint=str(out / "checkpoint_latest.pt") if (out / "checkpoint_latest.pt").exists() else None)
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", required=True)
    result.add_argument("--out", required=True)
    result.add_argument("--condition", choices=("slots", "native", "shuffled"), default="slots")
    result.add_argument("--stage", choices=("stage1", "stage2", "both"), default="both")
    result.add_argument("--smoke", action="store_true")
    result.add_argument("--steps", type=int, help="Total optimizer updates for one stage, including resumed updates")
    source = result.add_mutually_exclusive_group()
    source.add_argument("--init-checkpoint")
    source.add_argument("--resume")
    result.add_argument("--batch-size", type=int)
    result.add_argument("--gradient-accumulation", type=int)
    result.add_argument("--seed", type=int)
    result.add_argument("--device", default="cuda")
    result.add_argument("--threads", type=int, default=4)
    result.add_argument("--save-every", type=int, default=40)
    result.add_argument("--max-seconds", type=float, default=0)
    return result


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if args.save_every <= 0 or args.threads <= 0 or args.max_seconds < 0 or not math.isfinite(args.max_seconds):
        argument_parser.error("Save interval/threads must be positive; max seconds must be nonnegative and finite")
    if args.init_checkpoint and args.stage == "stage1":
        argument_parser.error("Stage-1 transfer initializes Stage 2 only")
    if args.init_checkpoint:
        args.stage = "stage2"
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
