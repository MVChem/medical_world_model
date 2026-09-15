"""Train/evaluate online VLM downstream pilots with resumable adapter checkpoints."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
from PIL import Image
import torch

try:
    from .model import (CONDITIONS, JointModel, adapter_state, gradient_audit,
                        load_adapter_state, parameter_group, update_audit)
except ImportError:
    from model import (CONDITIONS, JointModel, adapter_state, gradient_audit,
                       load_adapter_state, parameter_group, update_audit)
from common import atomic, digest, write_rows
from frozen_slots_train import (SlotCorpus, autocast, capture_rng, evaluate, objective,
                                patient_derangement, restore_rng, target_fingerprint)


class OnlineCorpus(SlotCorpus):
    def __init__(self, args):
        super().__init__(args.data_run, args.out, args.model, args.task, "image_only",
                         args.device, args.train_limit, args.seed, args.pseudo_path)
        self.condition = args.condition
        if args.eval_limit:
            for split in ("validate", "test", "human_test"):
                self.pool[split] = self.pool[split][:args.eval_limit]
        if self.condition == "shuffled_slots":
            for index, ids in enumerate(self.pool.values()):
                self.donors.update(patient_derangement(ids, self.rows, args.seed + index))

    def batch(self, ids):
        visual, _unused, target, mask = super().batch(ids)
        if self.condition == "image_only":
            return visual, [], target, mask
        # The task branch chooses its source before loading pixels. HR is never
        # passed to the SR VLM, including for shuffled donors and evaluations.
        source = self.lr_images if self.task == "sr" else self.images
        donors = [self.donors.get(index, index) for index in ids]
        images = [Image.fromarray(np.array(source[index], copy=True)).convert("RGB") for index in donors]
        return visual, images, target, mask


@torch.inference_mode()
def validate(model, corpus, count):
    model.eval()
    ids = corpus.pool["validate"][:count]
    total = 0.
    for index in ids:
        image, sources, target, mask = corpus.batch([index])
        with autocast(corpus.device):
            prediction = model(image, sources)
        loss = objective(corpus.task, prediction, target, mask)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite validation loss")
        total += float(loss)
    return total / len(ids)


def save_checkpoint(path, model, optimizer, contract, state, started, prior_seconds):
    payload = dict(trainable=adapter_state(model), optimizer=optimizer.state_dict(), contract=contract,
                   rng=capture_rng(), seconds=prior_seconds + time.monotonic() - started, **state)
    temporary = path.with_suffix(".tmp.pt")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def experiment_contract(args, corpus):
    import transformers
    return dict(version=1, model=args.model, task=args.task, condition=args.condition,
        protocol="online native VLM downstream adaptation pilot; original pretrained initialization",
        world_model_checkpoint_used=False, frozen_vjepa_features_used=False,
        slots=0 if args.condition in ("image_only", "joint_full_tokens") else 8,
        slot_branches="4 language depths + 4 native vision depths", dense_routing="all eight",
        input="LR only (128x128)" if args.task == "sr" else "HR (512x512), decoder image resized to256",
        patient_text_or_report_input=False,
        image_only="same seeded decoder and zero context; encoder absent",
        joint_full_tokens="all last native vision tokens and 64 final-language visual positions; no learned slot queries",
        frozen_slots="matched frozen-VLM control: queries, projections, decoder trainable; both backbone LoRA branches frozen",
        shuffled_slots="online donor pixels from a different patient within same split, including training",
        data_run=str(args.data_run.resolve()), cohort_sha256=corpus.cohort_hash,
        manifest_sha256=digest(corpus.data / "manifest.json"), image_sha256=corpus.manifest["image_sha256"],
        lr_image_sha256=corpus.manifest["lr_image_sha256"],
        pseudo_path=str(corpus.pseudo_path.resolve()) if args.task == "segmentation" else None,
        pseudo_sha256=target_fingerprint(corpus.pseudo_path, args.out.parent) if args.task == "segmentation" else None,
        human_mask_sha256=corpus.manifest.get("human_mask_sha256"),
        split_indices_sha256={key: hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                              for key, ids in corpus.pool.items()},
        split_counts={key: len(ids) for key, ids in corpus.pool.items()},
        donor_sha256=hashlib.sha256(json.dumps(corpus.donors, sort_keys=True).encode()).hexdigest(),
        seed=args.seed, epochs=args.epochs, max_steps=args.max_steps, effective_batch=args.batch_size,
        microbatch=args.microbatch, learning_rate=args.learning_rate, lora_learning_rate=args.lora_learning_rate,
        weight_decay=.01, lora_rank=args.rank, vision_pixels=args.vision_pixels, visual_tokens=args.visual_tokens,
        gradient_checkpointing=not args.no_checkpointing, optimizer="AdamW", precision="BF16 autocast, FP32 trainables",
        selection="final configured optimization step; validation monitored but no test-based selection",
        validation_every=args.validate_every, validation_count=args.validation_count,
        eval_limit=args.eval_limit, torch_version=str(torch.__version__), transformers_version=transformers.__version__,
        model_source_sha256=digest(Path(__file__).with_name("model.py")),
        trainer_source_sha256=digest(Path(__file__)),
        reused_decoder_source_sha256=digest(Path(__file__).parents[1] / "medworld_dense_baselines/frozen_slots_train.py"),
        reused_loss_source_sha256=digest(Path(__file__).parents[1] / "medworld_dense_baselines/heads.py"))


def train(args):
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _train(args)


def _train(args):
    os.umask(0o077)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    corpus = OnlineCorpus(args)
    contract = experiment_contract(args, corpus)
    contract_path = args.out / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("existing experiment contract differs; use a fresh output directory")
    atomic(contract_path, contract)
    if (args.out / "complete.json").exists():
        print("already complete", args.out, flush=True)
        return
    model = JointModel(args.model, args.task, args.condition, device=args.device, rank=args.rank,
                       vision_pixels=args.vision_pixels, visual_tokens=args.visual_tokens,
                       checkpointing=not args.no_checkpointing)
    if model.encoder is not None:
        atomic(args.out / "encoder.json", model.encoder.metadata)
    groups = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            groups.setdefault(parameter_group(name), []).append(param)
    optimizer = torch.optim.AdamW([dict(params=params,
        lr=args.lora_learning_rate if name.endswith("lora") else args.learning_rate,
        initial_lr=args.lora_learning_rate if name.endswith("lora") else args.learning_rate, group_name=name)
        for name, params in groups.items()], weight_decay=.01)
    head_hash = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes()
                                       for p in model.head.state_dict().values())).hexdigest()
    atomic(args.out / "initialization.json", dict(decoder_sha256=head_hash,
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        trainable_by_group={name: sum(p.numel() for p in params) for name, params in groups.items()},
        physical_gpus=os.environ.get("CUDA_VISIBLE_DEVICES")))
    state = dict(epoch=0, next_group=0, step=0, epoch_loss_sum=0., epoch_samples=0, history=[])
    checkpoint = args.out / "checkpoint.pt"
    prior_seconds = 0.
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["contract"] != contract:
            raise ValueError("checkpoint contract differs")
        load_adapter_state(model, saved["trainable"])
        optimizer.load_state_dict(saved["optimizer"])
        state = {key: saved[key] for key in state}
        prior_seconds = saved["seconds"]
        restore_rng(saved["rng"])
    stopped = {"requested": False, "signal": None}
    def handle_signal(signum, _frame):
        stopped.update(requested=True, signal=signum)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    deadline = datetime.fromisoformat(args.deadline).timestamp() if args.deadline else float("inf")
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats() if corpus.device.type == "cuda" else None
    training_done = False
    while state["epoch"] < args.epochs and not training_done:
        epoch = state["epoch"]
        ids = np.random.default_rng(args.seed + epoch).permutation(corpus.pool["train"]).tolist()
        model.train()
        for start in range(state["next_group"], len(ids), args.batch_size):
            if args.max_steps and state["step"] >= args.max_steps:
                training_done = True
                break
            if stopped["requested"] or time.time() >= deadline:
                save_checkpoint(checkpoint, model, optimizer, contract, state, started, prior_seconds)
                atomic(args.out / "progress.json", dict(status="paused", signal=stopped["signal"],
                                                        epoch=epoch, step=state["step"]))
                print("paused after resumable checkpoint", flush=True)
                return
            chunk = ids[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            progress = state["step"] / max(1, args.max_steps or math.ceil(len(ids) / args.batch_size) * args.epochs)
            schedule = .1 + .9 * .5 * (1 + math.cos(math.pi * min(progress, 1.)))
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * schedule
            first_before = adapter_state(model) if state["step"] == 0 else None
            audit = None
            for position in range(0, len(chunk), args.microbatch):
                ii = chunk[position:position + args.microbatch]
                image, sources, target, mask = corpus.batch(ii)
                with autocast(corpus.device):
                    prediction = model(image, sources)
                    loss = objective(corpus.task, prediction, target, mask)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite training loss")
                (loss * len(ii) / len(chunk)).backward()
                if state["step"] == 0:
                    audit = gradient_audit(model)
                state["epoch_loss_sum"] += float(loss.detach()) * len(ii)
                state["epoch_samples"] += len(ii)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                           1., error_if_nonfinite=True)
            optimizer.step()
            state["step"] += 1
            state["next_group"] = start + len(chunk)
            if audit is not None:
                audit.update(step=state["step"], update_l2=update_audit(model, first_before),
                             all_pretrained_base_weights_frozen=True,
                             effective_backbone_updated_via_lora=args.condition not in ("image_only", "frozen_slots"))
                for key in ("language_lora", "vision_lora"):
                    if args.condition not in ("image_only", "frozen_slots") and audit["update_l2"].get(key, 0.) <= 0:
                        raise AssertionError(f"optimizer did not update {key}")
                atomic(args.out / "gradient_audit.json", audit)
            if state["step"] == 1 or state["step"] % args.log_every == 0:
                record = dict(status="training", epoch=epoch + 1, step=state["step"],
                    epoch_samples=state["epoch_samples"], train_loss=state["epoch_loss_sum"] / state["epoch_samples"],
                    seconds=prior_seconds + time.monotonic() - started,
                    max_cuda_memory_gib=torch.cuda.max_memory_allocated() / 2**30 if corpus.device.type == "cuda" else 0.)
                atomic(args.out / "progress.json", record)
                print(json.dumps(record), flush=True)
            if args.validate_every and state["step"] % args.validate_every == 0:
                val = validate(model, corpus, args.validation_count)
                state["history"].append(dict(step=state["step"], epoch=epoch + 1, validate_loss=val))
                write_rows(args.out / "validation.jsonl", state["history"])
                model.train()
            if state["step"] % args.checkpoint_every == 0:
                save_checkpoint(checkpoint, model, optimizer, contract, state, started, prior_seconds)
        if state["next_group"] >= len(ids):
            state.update(epoch=epoch + 1, next_group=0, epoch_loss_sum=0., epoch_samples=0)
            save_checkpoint(checkpoint, model, optimizer, contract, state, started, prior_seconds)
    save_checkpoint(checkpoint, model, optimizer, contract, state, started, prior_seconds)
    completed_epochs = state["epoch"] + state["next_group"] / len(corpus.pool["train"])
    atomic(args.out / "progress.json", dict(status="evaluating", step=state["step"], epochs_completed=completed_epochs))
    evaluate(model, corpus, args.out, args.microbatch, completed_epochs, args.seed, contract)
    metrics = json.loads((args.out / "metrics.json").read_text())
    metrics.update(training_steps=state["step"], training_epoch_fraction=completed_epochs,
        checkpoint="final configured optimization step; no test-based selection",
        joint_online=args.condition in ("joint_slots", "shuffled_slots", "joint_full_tokens"),
        world_model_pretraining=False, pilot=True, evaluated_at=datetime.now().astimezone().isoformat())
    atomic(args.out / "metrics.json", metrics)
    complete = dict(complete=True, step=state["step"], epochs_completed=completed_epochs,
        seconds=prior_seconds + time.monotonic() - started, contract_sha256=digest(contract_path),
        metrics_sha256=digest(args.out / "metrics.json"), checkpoint_sha256=digest(checkpoint))
    atomic(args.out / "complete.json", complete)
    atomic(args.out / "progress.json", dict(status="complete", **complete))
    print(json.dumps(complete), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", choices=("segmentation", "sr"), required=True)
    parser.add_argument("--condition", choices=CONDITIONS, default="joint_slots")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lora-learning-rate", type=float, default=5e-5)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--vision-pixels", type=int, default=256)
    parser.add_argument("--visual-tokens", type=int, default=64)
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--eval-limit", type=int, help="development smoke only; limits all evaluation splits")
    parser.add_argument("--pseudo-path", type=Path)
    parser.add_argument("--validate-every", type=int, default=200)
    parser.add_argument("--validation-count", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--deadline", help="timezone-aware ISO time; save and pause before next optimizer step")
    args = parser.parse_args()
    for key in ("epochs", "batch_size", "microbatch", "rank", "vision_pixels", "visual_tokens",
                "validation_count", "checkpoint_every", "log_every"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.max_steps < 0 or args.validate_every < 0:
        parser.error("max-steps and validate-every must be nonnegative")
    if args.deadline and datetime.fromisoformat(args.deadline).tzinfo is None:
        parser.error("deadline must include a timezone")
    return args


if __name__ == "__main__":
    train(parse_args())
