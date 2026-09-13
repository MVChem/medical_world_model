"""Matched image decoders conditioned on four frozen vision-depth slots.

The four slots are a set of depth summaries, never a 2D feature grid.  An
image-only control uses this exact decoder with zero slot inputs.  Only decoder
parameters are optimized; no VLM or learned slot vectors are instantiated here.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity

from common import DEFAULT_RUN, OLD, atomic, digest, read_rows, write_rows
from heads import block, segmentation_loss

DEFAULT_SEED = 20260913
CONDITIONS = ("image_only", "slots", "shuffled_slots")


def positional_encoding(height, width, channels=64):
    """Fixed sine/cosine image coordinates; no trainable position/slot vectors."""
    if channels % 4:
        raise ValueError("position channels must be divisible by four")
    frequency = torch.exp(-math.log(10000) * torch.arange(channels // 4) / (channels // 4))
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    xx, yy = x.flatten()[:, None] * frequency, y.flatten()[:, None] * frequency
    return torch.cat([xx.sin(), xx.cos(), yy.sin(), yy.cos()], dim=-1)[None]


class FrozenSlotHead(nn.Module):
    def __init__(self, task):
        super().__init__()
        if task not in ("segmentation", "sr"):
            raise ValueError(task)
        self.task = task
        self.stem = nn.Sequential(block(1, 32, 2), block(32, 64, 2))
        # Affine-free normalization prevents native backbone scale differences
        # from dominating the common projection.  Inputs always remain frozen.
        self.slot_normalize = nn.LayerNorm(1024, elementwise_affine=False)
        self.slot_project = nn.Sequential(nn.Linear(1024, 64), nn.GELU())
        self.query_normalize = nn.LayerNorm(64)
        self.attention = nn.MultiheadAttention(64, 4, batch_first=True, dropout=0)
        self.register_buffer("image_position", positional_encoding(32, 32))
        self.register_buffer("depth_position", positional_encoding(1, 4))
        self.fuse = block(128, 64)
        layers = []
        channels = 64
        for width in [48, 32, 16] + ([8] if task == "sr" else []):
            layers.extend([nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                           block(channels, width)])
            channels = width
        layers.append(nn.Conv2d(channels, 1 if task == "sr" else 3, 1))
        self.decode = nn.Sequential(*layers)

    def forward(self, image, slots):
        if slots.ndim != 3 or tuple(slots.shape[1:]) != (4, 1024):
            raise ValueError(f"expected frozen slots [B,4,1024], got {tuple(slots.shape)}")
        z = F.interpolate(self.stem(image), (32, 32), mode="bilinear", align_corners=False)
        query = self.query_normalize(z.flatten(2).transpose(1, 2) + self.image_position)
        values = self.slot_project(self.slot_normalize(slots.detach().float())) + self.depth_position
        attended, _ = self.attention(query, values, values, need_weights=False)
        attended = attended.transpose(1, 2).reshape(image.shape[0], 64, 32, 32)
        result = self.decode(self.fuse(torch.cat([z, attended], dim=1)))
        if self.task == "sr":
            return F.interpolate(image, scale_factor=4, mode="bicubic", align_corners=False) + .1 * result
        return result


def patient_derangement(ids, rows, seed):
    """Deterministic within-split donor mapping with a different patient."""
    groups = {}
    for i in ids:
        groups.setdefault(str(rows[i]["subject_id"]), []).append(i)
    if not ids:
        return {}
    rng = np.random.default_rng(seed)
    keys = list(groups)
    rng.shuffle(keys)
    ordered = [i for key in keys for i in groups[key]]
    largest = max(map(len, groups.values()))
    if largest * 2 > len(ordered):
        raise ValueError("cannot construct shuffled control with different-patient donors in this split")
    donors = ordered[largest:] + ordered[:largest]
    result = dict(zip(ordered, donors))
    assert all(str(rows[i]["subject_id"]) != str(rows[j]["subject_id"]) for i, j in result.items())
    return result


class SlotCorpus:
    def __init__(self, data_run, run, mid, task, condition, device="cuda", train_limit=None,
                 seed=DEFAULT_SEED, pseudo_path=None):
        self.data = Path(data_run) / "data"
        self.cache = Path(run) / mid
        self.task, self.condition = task, condition
        if condition not in CONDITIONS:
            raise ValueError(condition)
        self.device = torch.device(device)
        self.rows = read_rows(self.data / "observations.jsonl")
        if any(row["index"] != i for i, row in enumerate(self.rows)):
            raise ValueError("observation indices do not match cache row order")
        self.cohort_hash = digest(self.data / "observations.jsonl")
        manifest = json.loads((self.data / "manifest.json").read_text())
        self.manifest = manifest
        if manifest["cohort_sha256"] != self.cohort_hash:
            raise ValueError("prepared cohort manifest no longer matches observations")
        self.images = np.load(self.data / "images.npy", mmap_mode="r")
        self.lr_images = np.load(self.data / "lr_images.npy", mmap_mode="r")
        self.human = np.load(self.data / "human_masks.npy", mmap_mode="r") if task == "segmentation" else None
        self.pseudo_path = Path(pseudo_path) if pseudo_path else OLD / "seg_probs.npy"
        self.pseudo = np.load(self.pseudo_path, mmap_mode="r") if task == "segmentation" else None
        self.pool = {split: [i for i, r in enumerate(self.rows)
                             if r["split"] == split and task in r["tasks"]]
                     for split in ("train", "validate", "test", "human_test")}
        if train_limit is not None:
            if train_limit <= 0:
                raise ValueError("train-limit must be positive")
            self.pool["train"] = self.pool["train"][:train_limit]
        if any(not self.pool[s] for s in ("train", "validate", "test")):
            raise ValueError("train, validate, and test cohorts must all be nonempty")
        split_patients = [{str(self.rows[i]["subject_id"]) for i in ids} for ids in self.pool.values()]
        for a in range(len(split_patients)):
            for b in range(a):
                if split_patients[a] & split_patients[b]:
                    raise ValueError("patient leakage across task splits")
        self.slots = None
        self.slot_contract_hash = None
        self.donors = {}
        if condition != "image_only":
            modality = "lr" if task == "sr" else "hr"
            self.slots = np.load(self.cache / f"{modality}_slots.npy", mmap_mode="r")
            if self.slots.shape != (len(self.rows), 4, 1024):
                raise ValueError(f"invalid slot cache shape: {self.slots.shape}")
            done = np.load(self.cache / "slots_done.npy")
            required = sorted({i for ids in self.pool.values() for i in ids})
            if done.shape != (len(self.rows), 2) or not done[required, 1 if task == "sr" else 0].all():
                raise ValueError("incomplete frozen slot cache for selected cohort")
            self.slot_contract_hash = digest(self.cache / "slot_contract.json")
            slot_contract = json.loads((self.cache / "slot_contract.json").read_text())
            if slot_contract["model_id"] != mid or slot_contract["shape"] != [len(self.rows), 4, 1024]:
                raise ValueError("frozen slot cache model or shape contract mismatch")
            if slot_contract["slot_ids"] != [5, 6, 7, 8] or slot_contract["branch_columns"] != ["hr", "lr"]:
                raise ValueError("frozen slot cache identity or branch mapping mismatch")
            if slot_contract["slot_training"] is not False or slot_contract["language_model_loaded"] is not False:
                raise ValueError("slot cache must come from an untrained native vision encoder")
            if slot_contract["cohort_sha256"] != self.cohort_hash:
                raise ValueError("frozen slot cache belongs to a different cohort")
            for key in ("image_sha256", "lr_image_sha256"):
                if slot_contract[key] != manifest[key]:
                    raise ValueError(f"frozen slot cache input mismatch: {key}")
            if condition == "shuffled_slots":
                for index, ids in enumerate(self.pool.values()):
                    self.donors.update(patient_derangement(ids, self.rows, seed + index))

    def batch(self, ids):
        def tensor(array):
            return torch.from_numpy(np.array(array, copy=True)).to(self.device, dtype=torch.float32)

        # SR's visual input and slots both originate exclusively from the LR cache.
        # HR is loaded below solely as the reconstruction target.
        target_image = tensor(self.images[ids])[:, None] / 255
        if self.task == "sr":
            visual = tensor(self.lr_images[ids])[:, None] / 255
            target = target_image
            size = 512
        else:
            visual = F.interpolate(target_image, (256, 256), mode="area")
            target = tensor(np.stack([self.human[self.rows[i]["human_index"]]
                                     if self.rows[i]["kind"] == "montgomery"
                                     else self.pseudo[self.rows[i]["old_index"]] for i in ids]))
            size = 256
        mask = torch.zeros((len(ids), 1, size, size), device=self.device)
        scale = 512 // size
        for b, i in enumerate(ids):
            y, x, h, w = [v // scale for v in self.rows[i]["box"]]
            mask[b, :, y:y+h, x:x+w] = 1
        if self.slots is None:
            slots = torch.zeros((len(ids), 4, 1024), device=self.device)
        else:
            donors = [self.donors.get(i, i) for i in ids]
            slots = tensor(self.slots[donors])
        if not all(torch.isfinite(value).all() for value in (visual, slots, target)):
            raise ValueError("nonfinite input or target")
        return visual, slots, target, mask


def objective(task, prediction, target, mask):
    if task == "segmentation":
        return segmentation_loss(prediction[:, :target.shape[1]], target, mask)
    return (((prediction.float() - target).square() * mask).sum((1, 2, 3)) /
            mask.sum((1, 2, 3)).clamp_min(1)).mean()


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()


@torch.inference_mode()
def validation_loss(model, corpus, batch_size):
    model.eval()
    total, n = 0., 0
    ids = corpus.pool["validate"]
    for start in range(0, len(ids), batch_size):
        ii = ids[start:start+batch_size]
        x, slots, target, mask = corpus.batch(ii)
        with autocast(corpus.device):
            prediction = model(x, slots)
        loss = objective(corpus.task, prediction, target, mask)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite validation loss")
        total += float(loss) * len(ii)
        n += len(ii)
    return total / n


def bootstrap(records, keys, seed=DEFAULT_SEED, draws=1000):
    by_patient = {}
    for record in records:
        by_patient.setdefault(str(record["subject_id"]), []).append(record)
    groups = list(by_patient.values())
    counts = np.array([len(group) for group in groups])
    sums = np.array([[sum(record[key] for record in group) for key in keys] for group in groups])
    samples = np.random.default_rng(seed).integers(0, len(groups), (draws, len(groups)))
    means = sums[samples].sum(axis=1) / counts[samples].sum(axis=1)[:, None]
    return {key: np.quantile(means[:, index], [.025, .975]).tolist() for index, key in enumerate(keys)}


@torch.inference_mode()
def evaluate(model, corpus, out, batch_size, epochs, seed, contract):
    model.eval()
    results = {}
    splits = ["validate", "test"] + (["human_test"] if corpus.task == "segmentation" else [])
    for split in splits:
        ids, records = corpus.pool[split], []
        if not ids:
            raise ValueError(f"missing required evaluation split {split}")
        for start in range(0, len(ids), batch_size):
            ii = ids[start:start+batch_size]
            x, slots, target, mask = corpus.batch(ii)
            with autocast(corpus.device):
                prediction = model(x, slots).float()
            if not torch.isfinite(prediction).all():
                raise ValueError("nonfinite evaluation prediction")
            if corpus.task == "segmentation":
                predicted = (prediction[:, :target.shape[1]].sigmoid() > .5).float() * mask
                truth = (target > .5).float() * mask
                dices = (2 * (predicted * truth).sum((2, 3)) + 1e-8) / (
                    predicted.sum((2, 3)) + truth.sum((2, 3)) + 1e-8)
            for j, i in enumerate(ii):
                row = corpus.rows[i]
                record = dict(id=row["id"], subject_id=row["subject_id"], index=i)
                if corpus.condition == "shuffled_slots":
                    record["slot_donor_id"] = corpus.rows[corpus.donors[i]]["id"]
                if corpus.task == "segmentation":
                    record.update(dice=float(dices[j].mean()), per_organ=dices[j].cpu().tolist())
                else:
                    y, x0, h, w = row["box"]
                    restored = prediction[j, 0, y:y+h, x0:x0+w].clamp(0, 1).cpu().numpy()
                    truth_image = target[j, 0, y:y+h, x0:x0+w].cpu().numpy()
                    mse = float(np.mean((restored - truth_image) ** 2))
                    record.update(psnr=float(-10 * np.log10(max(mse, 1e-12))),
                                  ssim=float(structural_similarity(truth_image, restored, data_range=1.)))
                records.append(record)
        if len(records) != len(ids):
            raise ValueError("evaluation cohort lost samples")
        keys = ["dice"] if corpus.task == "segmentation" else ["psnr", "ssim"]
        if not all(np.isfinite(record[key]) for record in records for key in keys):
            raise ValueError("nonfinite evaluation metric")
        write_rows(out / f"{split}_per_sample.jsonl", records)
        results[split] = dict(n=len(records), patients=len({r["subject_id"] for r in records}),
                              **{key: float(np.mean([r[key] for r in records])) for key in keys},
                              ci95=bootstrap(records, keys, seed))
        if corpus.task == "segmentation":
            results[split]["per_organ"] = np.mean([r["per_organ"] for r in records], axis=0).tolist()
            results[split]["organs"] = ["right lung", "left lung"] + ([] if split == "human_test" else ["heart"])
    atomic(out / "metrics.json", dict(model=contract["model"], task=corpus.task, condition=corpus.condition,
           epochs=epochs, seed=seed, train_n=len(corpus.pool["train"]), metrics=results,
           checkpoint="final epoch; no test-based checkpoint selection",
           cohort_sha256=contract["cohort_sha256"], contract_sha256=digest(out / "contract.json"),
           segmentation_scope="CXAS teacher agreement; external Montgomery human lungs",
           sr_scope="x4; LR-only encoder inputs; valid ROI; clamp [0,1]; no border shave; skimage 7x7 SSIM"))
    return results


def state_hash(model):
    return hashlib.sha256(b"".join(value.detach().cpu().numpy().tobytes()
                                  for value in model.state_dict().values())).hexdigest()


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def target_fingerprint(path, run):
    """Hash the large shared pseudo-label file once, reusing only unchanged stat identity."""
    path = Path(path).resolve()
    fingerprint_path = Path(run) / "target_fingerprint.json"
    Path(run).mkdir(parents=True, exist_ok=True)
    with (Path(run) / "target_fingerprint.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        stat = path.stat()
        identity = dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                        ctime_ns=stat.st_ctime_ns, inode=stat.st_ino, device=stat.st_dev)
        cached = json.loads(fingerprint_path.read_text()) if fingerprint_path.exists() else {}
        if cached.get("identity") == identity:
            return cached["sha256"]
        result = digest(path)
        after = path.stat()
        if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns):
            raise ValueError("pseudo targets changed while hashing")
        atomic(fingerprint_path, dict(identity=identity, sha256=result))
        return result


def train(args):
    out = args.out or args.run / args.model / f"{args.task}_{args.condition}"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another trainer owns output directory {out}") from error
        return _train(args)


def _train(args):
    if args.epochs <= 0 or args.batch_size <= 0 or args.microbatch <= 0:
        raise ValueError("epochs, batch-size, and microbatch must be positive")
    os.umask(0o077)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    corpus = SlotCorpus(args.data_run, args.run, args.model, args.task, args.condition,
                        args.device, args.train_limit, args.seed, args.pseudo_path)
    out = args.out or args.run / args.model / f"{args.task}_{args.condition}"
    out.mkdir(parents=True, exist_ok=True)
    contract = dict(version=1, model=args.model, task=args.task, condition=args.condition,
                    architecture="image_cnn_plus_fixed_position_query_cross_attention_to_4_depth_slots",
                    slots_shape=[4, 1024], slots_trained=False, vlm_loaded=False,
                    slot_norm="affine-free LayerNorm over 1024 aligned channels",
                    image_only="identical parameters and initialization; zero slot input",
                    data_run=str(args.data_run.resolve()), seed=args.seed, epochs=args.epochs,
                    batch_size=args.batch_size, microbatch=args.microbatch, learning_rate=args.learning_rate,
                    learning_rate_schedule="epoch cosine decay to 10% floor",
                    loss="valid-pixel MSE" if args.task == "sr" else "valid-pixel BCE plus soft Dice",
                    weight_decay=.01, optimizer="AdamW", device_type=corpus.device.type,
                    amp="bfloat16" if corpus.device.type == "cuda" else "float32",
                    train_n=len(corpus.pool["train"]),
                    split_indices_sha256={s: hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                                          for s, ids in corpus.pool.items()},
                    cohort_sha256=digest(corpus.data / "observations.jsonl"),
                    manifest_sha256=digest(corpus.data / "manifest.json"),
                    image_sha256=corpus.manifest["image_sha256"],
                    lr_image_sha256=corpus.manifest["lr_image_sha256"],
                    pseudo_sha256=target_fingerprint(corpus.pseudo_path, args.run) if args.task == "segmentation" else None,
                    human_mask_sha256=corpus.manifest.get("human_mask_sha256") if args.task == "segmentation" else None,
                    source_sha256=digest(Path(__file__)),
                    reused_loss_sha256=digest(Path(__file__).with_name("heads.py")),
                    slot_contract_sha256=corpus.slot_contract_hash,
                    pseudo_path=str(corpus.pseudo_path.resolve()) if args.task == "segmentation" else None,
                    donor_sha256=hashlib.sha256(json.dumps(corpus.donors, sort_keys=True).encode()).hexdigest()
                    if corpus.donors else None)
    contract_path = out / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError(f"existing experiment contract differs: {contract_path}; use a fresh output")
    atomic(contract_path, contract)
    if (out / "metrics.json").exists():
        print(f"already complete: {out}", flush=True)
        return
    model = FrozenSlotHead(args.task).to(corpus.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    initialization = dict(state_sha256=state_hash(model), parameters=sum(p.numel() for p in model.parameters()),
                          trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                          slot_parameters=0, microbatch=args.microbatch, effective_batch=args.batch_size)
    initial_path = out / "initialization.json"
    if initial_path.exists() and json.loads(initial_path.read_text()) != initialization:
        raise ValueError("decoder initialization mismatch")
    atomic(initial_path, initialization)
    first, step, history, prior_seconds = 0, 0, [], 0.
    checkpoint = out / "checkpoint.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["contract"] != contract:
            raise ValueError("checkpoint contract mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        first, step = saved["epoch"], saved["step"]
        history, prior_seconds = saved["history"], saved["seconds"]
        restore_rng(saved["rng"])
        write_rows(out / "epochs.jsonl", history)
    started = time.time()
    for epoch in range(first, args.epochs):
        model.train()
        total, n = 0., 0
        ids = np.random.default_rng(args.seed + epoch).permutation(corpus.pool["train"]).tolist()
        lr = args.learning_rate * (.1 + .9 * .5 * (1 + np.cos(np.pi * epoch / args.epochs)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        for start in range(0, len(ids), args.batch_size):
            chunk = ids[start:start+args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            for position in range(0, len(chunk), args.microbatch):
                ii = chunk[position:position+args.microbatch]
                image, slots, target, mask = corpus.batch(ii)
                with autocast(corpus.device):
                    prediction = model(image, slots)
                loss = objective(args.task, prediction, target, mask)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite training loss")
                (loss * len(ii) / len(chunk)).backward()
                total += float(loss.detach()) * len(ii)
                n += len(ii)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if step % 50 == 0:
                atomic(out / "progress.json", dict(epoch=epoch+1, epochs=args.epochs, step=step,
                       samples_in_epoch=n, training_loss=total/n,
                       seconds=prior_seconds+time.time()-started, status="training"))
        val = validation_loss(model, corpus, args.microbatch)
        record = dict(epoch=epoch+1, step=step, train_loss=total/n, validate_loss=val, learning_rate=lr,
                      order_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                      seconds=prior_seconds+time.time()-started)
        history.append(record)
        payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch+1, step=step,
                       rng=capture_rng(), history=history, seconds=record["seconds"], contract=contract)
        torch.save(payload, out / "checkpoint.tmp.pt")
        os.replace(out / "checkpoint.tmp.pt", checkpoint)
        write_rows(out / "epochs.jsonl", history)
        if epoch+1 in (5, 10, 20):
            torch.save(dict(model=model.state_dict(), epoch=epoch+1, contract=contract), out / f"epoch_{epoch+1}.pt")
        atomic(out / "progress.json", dict(**record, epochs=args.epochs, status="epoch_complete"))
        print(args.model, args.task, args.condition, json.dumps(record), flush=True)
    results = evaluate(model, corpus, out, args.microbatch, args.epochs, args.seed, contract)
    atomic(out / "progress.json", dict(epoch=args.epochs, epochs=args.epochs, step=step, status="complete",
           seconds=prior_seconds+time.time()-started))
    print(json.dumps(results), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", choices=("segmentation", "sr"), required=True)
    parser.add_argument("--condition", choices=CONDITIONS, default="slots")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--pseudo-path", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
