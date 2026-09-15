"""Official SwinIR-M classical x4, adapted to the fixed CXR SR protocol.

The official RGB network receives repeated LR grayscale channels. Its RGB output
is averaged into a grayscale prediction. HR is read only as a supervised target.
All native SwinIR parameters are adapted; no random crop, augmentation, HR input,
test selection, tiling, or extra reconstruction bypass is added.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get("MEDWORLD_PROJECT", HERE.parent.parent)).resolve()
BASE = PROJECT / "code/medworld_open_baselines"
WEIGHT_NAME = "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
WEIGHT_SHA256 = "4e78e33f22c1aa8a773db0cf4a7381bae97c2362c717f155439ebc690cbd9215"
OFFICIAL_COMMIT = "6545850fbf8df298df73d81f3e8cba638787c8bd"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["train", "smoke", "inspect"])
    p.add_argument("--data-run", type=Path, default=PROJECT / "code/medworld_dense_baselines/runs/dense_20260912")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=BASE / "third_party/SwinIR")
    p.add_argument("--weights", type=Path, default=BASE / "swinir_weights" / WEIGHT_NAME)
    p.add_argument("--protocol-source", type=Path, default=BASE / "swinir_protocol")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--microbatch", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=20260913)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--checkpoint-seconds", type=float, default=120.)
    p.add_argument("--stop-at", help="Optional timezone-aware deadline; none is imposed by default")
    p.add_argument("--stop-after-steps", type=int, help="Operational checkpoint/pause, not a final result")
    p.add_argument("--activation-checkpoint", action="store_true")
    return p.parse_args()


def load_protocol(args):
    os.environ["MEDWORLD_PROJECT"] = str(PROJECT)
    sys.path.insert(0, str(args.protocol_source.resolve()))
    import frozen_slots_train as protocol
    if Path(protocol.__file__).resolve().parent != args.protocol_source.resolve():
        raise ValueError("another module shadowed the frozen SR protocol")
    return protocol


class SwinIRGray(nn.Module):
    def __init__(self, args, protocol):
        super().__init__()
        path = args.repo / "models/network_swinir.py"
        spec = importlib.util.spec_from_file_location("medworld_official_network_swinir", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.network = module.SwinIR(
            upscale=4, in_chans=3, img_size=64, window_size=8,
            img_range=1., depths=[6] * 6, embed_dim=180, num_heads=[6] * 6,
            mlp_ratio=2, upsampler="pixelshuffle", resi_connection="1conv",
            use_checkpoint=args.activation_checkpoint)
        if protocol.digest(args.weights) != WEIGHT_SHA256:
            raise ValueError("official SwinIR-M DF2K x4 weight fingerprint differs")
        state = torch.load(args.weights, map_location="cpu", weights_only=True)
        self.network.load_state_dict(state["params"], strict=True)

    def forward(self, lr, unused_slots=None):
        if lr.ndim != 4 or lr.shape[1] != 1:
            raise ValueError("SwinIR CXR adapter requires B,1,H,W LR")
        return self.network(lr.repeat(1, 3, 1, 1)).mean(dim=1, keepdim=True)


def setup(args):
    if min(args.epochs, args.batch_size, args.microbatch, args.threads) <= 0:
        raise ValueError("epochs, effective/micro batch sizes, threads must be positive")
    os.umask(0o077)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    protocol = load_protocol(args)
    corpus = protocol.SlotCorpus(args.data_run, args.out, "swinir_m_df2k_x4", "sr", "image_only", args.device)
    corpus.condition = "cxr_adapted_official_swinir"
    # Hash actual arrays, not just a self-reported manifest, before training.
    for filename, key in [("images.npy", "image_sha256"), ("lr_images.npy", "lr_image_sha256")]:
        if protocol.digest(corpus.data / filename) != corpus.manifest[key]:
            raise ValueError(f"SR input array changed: {filename}")
    counts = {s: len(v) for s, v in corpus.pool.items()}
    if counts["validate"] != 249 or counts["test"] != 447 or counts["train"] not in [4096, 18708]:
        raise ValueError(f"unrecognized matched dense cohort: {counts}")
    model = SwinIRGray(args, protocol).to(corpus.device)
    commit = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != OFFICIAL_COMMIT:
        raise ValueError(f"official source checkout changed: {commit}")
    changed = subprocess.check_output(["git", "-C", str(args.repo), "status", "--porcelain", "--untracked-files=no"], text=True)
    if changed.strip():
        raise ValueError("official SwinIR checkout has tracked modifications")
    contract = dict(
        version=1, model="swinir_m_df2k_x4", task="sr", condition=corpus.condition,
        architecture="official SwinIR-M classical x4, full-parameter CXR adaptation",
        official_repo="https://github.com/JingyunLiang/SwinIR", official_commit=commit,
        official_pretraining="DF2K (DIV2K + Flickr2K), RGB natural-image classical x4 SR",
        weights=str(args.weights.resolve()), weights_sha256=WEIGHT_SHA256,
        adapter="repeat grayscale LR to RGB; mean RGB output channels; full LR128 to HR512 canvas",
        target_use="HR is a reconstruction target only; forward takes only LR",
        data_run=str(args.data_run.resolve()), seed=args.seed, epochs=args.epochs,
        batch_size=args.batch_size, microbatch=args.microbatch,
        learning_rate=args.learning_rate, learning_rate_schedule="epoch cosine decay to 10% floor",
        optimizer="AdamW", weight_decay=.01, gradient_clip_norm=1., loss="valid-pixel MSE, per-image normalization",
        activation_checkpoint=args.activation_checkpoint,
        train_n=counts["train"], split_counts=counts,
        planned_optimizer_updates=args.epochs * math.ceil(counts["train"] / args.batch_size),
        planned_train_samples=args.epochs * counts["train"],
        compute_matching="same cohort, epochs, effective batch, order, optimizer and single GPU class; actual compute differs by architecture",
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        split_indices_sha256={s: hashlib.sha256(json.dumps(ids).encode()).hexdigest() for s, ids in corpus.pool.items()},
        cohort_sha256=corpus.cohort_hash, manifest_sha256=protocol.digest(corpus.data / "manifest.json"),
        image_sha256=corpus.manifest["image_sha256"], lr_image_sha256=corpus.manifest["lr_image_sha256"],
        source_sha256=protocol.digest(Path(__file__)),
        protocol_source_sha256={name: protocol.digest(args.protocol_source / name) for name in ["common.py", "heads.py", "frozen_slots_train.py"]},
        official_network_sha256=protocol.digest(args.repo / "models/network_swinir.py"),
        device_type=corpus.device.type, amp="bfloat16" if corpus.device.type == "cuda" else "float32",
        torch_version=torch.__version__, numpy_version=np.__version__)
    return protocol, corpus, model, contract


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def smoke(args, protocol, corpus, model, contract):
    """Real LR full forward, LR isolation, tiny real-data gradient/roundtrip check."""
    begun = time.monotonic()
    ids = corpus.pool["train"][:1]
    lr, _, target, mask = corpus.batch(ids)
    model.eval()
    with torch.inference_mode(), protocol.autocast(corpus.device):
        pred = model(lr)
    assert pred.shape == target.shape == (1, 1, 512, 512)
    assert torch.isfinite(pred).all()
    # Mutating only an in-memory target cannot affect the next LR/model input.
    target.zero_()
    lr_again, _, target, mask = corpus.batch(ids)
    assert torch.equal(lr, lr_again)
    y, x, h, w = corpus.rows[ids[0]]["box"]
    yy, xx = y // 4, x // 4
    lr_crop = lr[:, :, yy:yy+16, xx:xx+16]
    truth_crop = target[:, :, y:y+64, x:x+64]
    mask_crop = mask[:, :, y:y+64, x:x+64]
    assert mask_crop.sum() == 4096
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    before = model.network.conv_first.weight.detach().clone()
    with protocol.autocast(corpus.device):
        prediction = model(lr_crop)
    loss = protocol.objective("sr", prediction, truth_crop, mask_crop)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    assert not torch.equal(before, model.network.conv_first.weight)
    smoke_checkpoint = args.out / "smoke_checkpoint.pt"
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rng=protocol.capture_rng()), smoke_checkpoint)
    model.eval()
    with torch.inference_mode(), protocol.autocast(corpus.device):
        expected = model(lr_crop)
    saved = torch.load(smoke_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    protocol.restore_rng(saved["rng"])
    with torch.inference_mode(), protocol.autocast(corpus.device):
        actual = model(lr_crop)
    assert torch.equal(actual, expected)
    result = dict(status="passed", test_scope="implementation smoke only, not an effect estimate", device=str(corpus.device),
                  full_lr_shape=list(lr.shape), full_prediction_shape=list(pred.shape),
                  tiny_crop_training_loss=float(loss.detach()), gradient_update=True,
                  checkpoint_reload_max_abs_error=float((actual-expected).abs().max()),
                  actual_array_hashes_verified=True, patient_splits_verified=True,
                  split_counts=contract["split_counts"], parameters=contract["parameters"],
                  source_sha256=contract["source_sha256"], weight_sha256=WEIGHT_SHA256,
                  seconds=time.monotonic()-begun)
    protocol.atomic(args.out / "smoke.json", result)
    protocol.atomic(args.out / "smoke_contract.json", contract)
    print(json.dumps(result), flush=True)


def train(args, protocol, corpus, model, contract):
    out = args.out
    path = out / "contract.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("existing run contract differs; use a fresh output directory")
    protocol.atomic(path, contract)
    if (out / "metrics.json").exists():
        print(f"already complete: {out}", flush=True)
        return
    gpu_info = dict(device=str(corpus.device), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    if corpus.device.type == "cuda":
        props = torch.cuda.get_device_properties(corpus.device)
        gpu_info.update(name=props.name, total_memory=props.total_memory, uuid=str(props.uuid))
    protocol.atomic(out / "hardware.json", gpu_info)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    first = step = cursor = partial_n = samples = 0
    partial_total = prior_seconds = training_seconds = 0.
    history = []
    checkpoint = out / "checkpoint.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["contract"] != contract:
            raise ValueError("checkpoint contract mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        first, step, cursor = saved["epoch"], saved["step"], saved["resume_batch_start"]
        partial_total, partial_n = saved["partial_total"], saved["partial_n"]
        history, prior_seconds = saved["history"], saved["seconds"]
        samples, training_seconds = saved["train_samples_seen"], saved["training_step_seconds"]
        protocol.restore_rng(saved["rng"])
        protocol.write_rows(out / "epochs.jsonl", history)
    started = last_checkpoint = time.monotonic()

    def progress(epoch, status, **kwargs):
        return dict(epoch=epoch, epochs=args.epochs, step=step, status=status,
                    train_samples_seen=samples, training_step_seconds=training_seconds,
                    seconds=prior_seconds+time.monotonic()-started,
                    gpu_count=1 if corpus.device.type == "cuda" else 0, **kwargs)

    def save(epoch, next_cursor=0, total=0., n=0):
        nonlocal last_checkpoint
        payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch, step=step,
                       resume_batch_start=next_cursor, partial_total=total, partial_n=n,
                       rng=protocol.capture_rng(), history=history, contract=contract,
                       seconds=prior_seconds+time.monotonic()-started,
                       train_samples_seen=samples, training_step_seconds=training_seconds)
        torch.save(payload, out / "checkpoint.tmp.pt")
        os.replace(out / "checkpoint.tmp.pt", checkpoint)
        last_checkpoint = time.monotonic()

    with protocol.StopRequest(args.stop_at) as stop:
        for epoch in range(first, args.epochs):
            model.train()
            total, n = (partial_total, partial_n) if epoch == first else (0., 0)
            next_cursor = cursor if epoch == first else 0
            ids = np.random.default_rng(args.seed + epoch).permutation(corpus.pool["train"]).tolist()
            lr = args.learning_rate * (.1 + .9 * .5 * (1 + np.cos(np.pi * epoch / args.epochs)))
            for group in optimizer.param_groups:
                group["lr"] = lr
            for start in range(next_cursor, len(ids), args.batch_size):
                if stop.requested() or (args.stop_after_steps is not None and step >= args.stop_after_steps):
                    save(epoch, start, total, n)
                    protocol.atomic(out / "progress.json", progress(epoch, "paused", samples_in_epoch=n,
                                    reason=stop.reason or "operational step pause; resumable"))
                    return 124
                chunk = ids[start:start+args.batch_size]
                synchronize(corpus.device)
                step_start = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                for position in range(0, len(chunk), args.microbatch):
                    ii = chunk[position:position+args.microbatch]
                    low, _, target, mask = corpus.batch(ii)
                    with protocol.autocast(corpus.device):
                        prediction = model(low)
                    loss = protocol.objective("sr", prediction, target, mask)
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite training loss")
                    (loss * len(ii) / len(chunk)).backward()
                    total += float(loss.detach()) * len(ii)
                    n += len(ii)
                    samples += len(ii)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                synchronize(corpus.device)
                training_seconds += time.monotonic() - step_start
                step += 1
                if time.monotonic() - last_checkpoint >= args.checkpoint_seconds:
                    save(epoch, start+len(chunk), total, n)
                if step % 10 == 0 or step == 1:
                    info = progress(epoch+1, "training", samples_in_epoch=n, training_loss=total/n)
                    protocol.atomic(out / "progress.json", info)
                    print(json.dumps(info), flush=True)
            val = protocol.validation_loss(model, corpus, args.microbatch)
            record = progress(epoch+1, "epoch_complete", train_loss=total/n, validate_loss=val, learning_rate=lr,
                              order_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
            history.append(record)
            save(epoch+1)
            protocol.write_rows(out / "epochs.jsonl", history)
            if epoch+1 in [5, 10, 20]:
                torch.save(dict(model=model.state_dict(), epoch=epoch+1, contract=contract), out / f"epoch_{epoch+1}.pt")
            protocol.atomic(out / "progress.json", record)
            print(json.dumps(record), flush=True)
        if step != contract["planned_optimizer_updates"] or samples != contract["planned_train_samples"]:
            raise ValueError("final budget accounting differs from declared epoch/sample/update budget")
        protocol.evaluate(model, corpus, out, args.microbatch, args.epochs, args.seed, contract)
        metrics = json.loads((out / "metrics.json").read_text())
        metrics.pop("segmentation_scope", None)
        metrics["compute"] = progress(args.epochs, "complete")
        metrics["pretraining"] = contract["official_pretraining"]
        metrics["checkpoint_sha256"] = protocol.digest(checkpoint)
        protocol.atomic(out / "metrics.json", metrics)
        protocol.atomic(out / "progress.json", progress(args.epochs, "complete"))
        print(json.dumps(metrics), flush=True)
    return 0


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol, corpus, model, contract = setup(args)
        if args.command == "inspect":
            protocol.atomic(args.out / "inspection.json", contract)
            print(json.dumps(contract), flush=True)
        elif args.command == "smoke":
            smoke(args, protocol, corpus, model, contract)
        else:
            sys.exit(train(args, protocol, corpus, model, contract) or 0)


if __name__ == "__main__":
    main()
