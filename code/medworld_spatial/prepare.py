"""Cache frozen, checkpoint-specific features and weak native-VLM crop scores.

Only image inputs are given to either teacher. Targets are stored for the
existing dense tasks, never passed to feature or semantic extraction.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from medworld.adaptation import capture_depths
from medworld.datasets import UnifiedData
from medworld.datasets.current import _sha256
from medworld.gpu import acquire_gpu
from medworld.runtime import atomic_json, load_model
from . import CONCEPTS
from .geometry import jitter, patch_grid, view
from .model import SlotReader


def pil_images(pixels):
    values = (pixels.detach().cpu().clamp(0, 1)[:, 0] * 255).round().byte().numpy()
    return [Image.fromarray(row).convert("RGB") for row in values]


def select_indices(rows, count, seed):
    ranked = sorted(range(len(rows)), key=lambda i: hashlib.sha256(f"{seed}:{rows[i]['id']}".encode()).hexdigest())
    return ranked[:min(count, len(rows))] if count else ranked


@torch.no_grad()
def extract(model, images, full):
    with capture_depths(model.encoder.vision.blocks) as captures:
        state = model.encode(images, spatial=not full)
    raw = torch.stack([captures[j].reshape(len(images), -1, captures[j].shape[-1]) for j in range(4)], 1)
    grid = model.processor.image_processor(images=images[:1], return_tensors="pt",
        min_pixels=model.cfg["vision_pixels"] ** 2, max_pixels=model.cfg["vision_pixels"] ** 2)["image_grid_thw"][0]
    if int(grid[0]) != 1:
        raise ValueError("Expected a single-frame spatial grid")
    return raw.float(), state.float(), tuple(int(n) for n in grid[1:])


class NativeSemanticTeacher:
    def __init__(self, path, device, batch_size=8):
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(path, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval().requires_grad_(False)
        self.device, self.batch_size = device, batch_size
        self.ids = [self.processor.tokenizer.encode(word, add_special_tokens=False) for word in ("No", "Yes")]
        if any(len(tokens) != 1 for tokens in self.ids):
            raise ValueError("Native semantic scoring requires single-token No/Yes")
        self.ids = [tokens[0] for tokens in self.ids]

    @torch.no_grad()
    def text_embeddings(self):
        # Text-only language states, then a learned query projection. No assumption
        # that raw visual and text embeddings share a cosine-similarity space.
        tokenizer = self.processor.tokenizer
        inputs = tokenizer([f"A chest radiograph containing {concept}." for concept in CONCEPTS],
                           padding=True, return_tensors="pt").to(self.device)
        hidden = self.model.model.language_model(**inputs, use_cache=False, return_dict=True).last_hidden_state
        return hidden[:, -1].float().cpu()

    @torch.no_grad()
    def scores(self, pixels):
        crops, texts = [], []
        for image in pil_images(pixels):
            width, height = image.size
            for concept in CONCEPTS:
                for row in range(2):
                    for col in range(2):
                        crop = image.crop((col * width // 2, row * height // 2,
                                           (col + 1) * width // 2, (row + 1) * height // 2))
                        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text",
                            "text": f"Does this chest radiograph crop visibly contain {concept}? Answer only Yes or No."}]}]
                        texts.append(self.processor.apply_chat_template(messages, tokenize=False,
                            add_generation_prompt=True, enable_thinking=False))
                        crops.append(crop)
        probabilities = []
        for start in range(0, len(crops), self.batch_size):
            inputs = self.processor(text=texts[start:start + self.batch_size],
                images=crops[start:start + self.batch_size], padding=True, return_tensors="pt",
                min_pixels=256**2, max_pixels=256**2).to(self.device)
            result = self.model(**inputs, use_cache=False, logits_to_keep=1)
            logits = result.logits[:, -1, self.ids].float()
            probabilities.append(logits.softmax(-1)[:, 1].cpu())
        return torch.cat(probabilities).reshape(len(pixels), len(CONCEPTS), 2, 2)


def main(args):
    os.umask(0o077)
    lock, device = acquire_gpu(args.gpu)
    torch.set_num_threads(args.threads)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        raise ValueError("Refusing to overwrite a completed cache")
    model, checkpoint = load_model(args.checkpoint, device)
    model.eval().requires_grad_(False)
    data = UnifiedData(model.cfg)
    if data.fingerprint != checkpoint["data_fingerprint"]:
        raise ValueError("Checkpoint and current data patient protocol differ")
    merge = model.encoder.vision.config.spatial_merge_size
    reader = SlotReader(model.encoder.visual_readouts, model.encoder.slot_queries[4:], model.encoder.visual_norm, merge)
    reader_state = {"raw_width": model.encoder.vision.config.hidden_size, "merge": merge,
                    "queries": reader.queries.detach().cpu(),
                    "state_dict": {k: v.detach().cpu() for k, v in reader.state_dict().items()}}
    torch.save(reader_state, out / "reader_init.pt")
    del reader
    generator = torch.Generator().manual_seed(9137)
    projection = torch.randn(reader_state["raw_width"], 64, generator=generator)
    projection = torch.linalg.qr(projection, mode="reduced").Q.to(device)
    torch.save(projection.cpu(), out / "teacher_projection.pt")
    protocol = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": _sha256(args.checkpoint),
        "data_fingerprint": data.fingerprint, "seed": args.seed, "task": args.task,
        "semantic_teacher": str(args.semantic_teacher.resolve()), "concepts": list(CONCEPTS),
        "views": args.views, "grid_order": "Qwen pre-merger block order, verified against image processor",
        "vision_depths_1based": checkpoint["metadata"]["vision_depths_1based"],
        "feature_target": "frozen checkpoint visual tower; fixed orthonormal 64-D projection",
        "inputs": "segmentation: 256-square canvas; SR: prepared 128-square LR only; reports absent",
        "semantic_targets": "native frozen VLM 2x2 crop No/Yes likelihoods; not segmentation labels or attention",
        "slot_training": "visual readouts and queries trainable; backbone and image-only JEPA/fusion slots frozen",
        "source_commit": args.source_commit, "started_unix": time.time()}
    atomic_json(out / "protocol.json", protocol)
    records = []
    splits = [("train", args.train_n), ("validate", args.val_n), ("test", args.test_n)]
    if args.task == "segmentation" and args.human_n:
        splits.append(("human_test", args.human_n))
    for split, count in splits:
        if count < 0:
            continue
        selected = select_indices(data.rows(args.task, split), count, args.seed)
        for start in range(0, len(selected), args.batch_size):
            if args.deadline and time.time() > args.deadline:
                raise TimeoutError("Feature preparation deadline reached; incomplete cache is not trainable")
            indices = selected[start:start + args.batch_size]
            batch = data.batch(args.task, split, indices)
            images = pil_images(batch["pixels"])
            raw, slots, grid = extract(model, images, full=True)
            # Cached-reader identity check protects the visual attention path.
            if not records:
                probe = SlotReader(model.encoder.visual_readouts, model.encoder.slot_queries[4:], model.encoder.visual_norm, merge).to(device)
                with torch.no_grad():
                    restored, maps = probe(raw, grid)
                torch.testing.assert_close(restored, slots[:, 4:], atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(maps.sum((-1, -2)), torch.ones_like(maps[:, :, 0, 0]))
                del probe
            thetas = [torch.tensor([[1., 0, 0], [0, 1., 0]]).repeat(len(images), 1, 1)]
            target0 = patch_grid(F.layer_norm(raw[:, -1], (raw.shape[-1],)) @ projection, *grid, merge)
            targets = [target0.cpu().half()]
            for j in range(args.views):
                theta = []
                for identity in batch["ids"]:
                    seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{identity}:{j}".encode()).digest()[:8], "little")
                    theta.append(jitter(torch.Generator().manual_seed(seed)))
                theta = torch.stack(theta)
                transformed = pil_images(view(batch["pixels"], theta))
                shifted, _, shifted_grid = extract(model, transformed, full=False)
                if shifted_grid != grid:
                    raise ValueError("Teacher view changed the token grid")
                target = patch_grid(F.layer_norm(shifted[:, -1], (shifted.shape[-1],)) @ projection, *grid, merge)
                targets.append(target.cpu().half())
                thetas.append(theta)
            targets, thetas = torch.stack(targets, 1), torch.stack(thetas, 1)
            for k, index in enumerate(indices):
                path = f"{split}_{index:06d}.pt"
                payload = {"raw": raw[k].cpu().half(), "fusion": slots[k, :4].cpu().half(), "grid": grid,
                    "pixels": batch["pixels"][k], "targets": batch["targets"][k], "valid": batch["mask"][k],
                    "teacher_features": targets[k], "thetas": thetas[k], "id": batch["ids"][k],
                    "patient": batch["subject_ids"][k]}
                torch.save(payload, out / path)
                records.append({"path": path, "split": split, "id": payload["id"], "patient": payload["patient"]})
            atomic_json(out / "status.json", {"phase": "features", "split": split, "completed": len(records), "heartbeat": time.time()})
            if start % 128 == 0:
                print(args.task, split, "features", start + len(indices), "/", len(selected), flush=True)
    atomic_json(out / "records.json", records)
    del model, data, raw, slots, shifted, projection
    gc.collect()
    torch.cuda.empty_cache()
    teacher = NativeSemanticTeacher(args.semantic_teacher, device, args.semantic_batch)
    torch.save(teacher.text_embeddings(), out / "text_embeddings.pt")
    diagnostics = []
    # Validation/test are exported only for diagnostics; never used to update.
    for index, record in enumerate(records):
        if args.deadline and time.time() > args.deadline:
            raise TimeoutError("Semantic preparation deadline reached")
        path = out / record["path"]
        payload = torch.load(path, weights_only=True)
        scores = teacher.scores(payload["pixels"][None])[0]
        if not torch.isfinite(scores).all():
            raise ValueError("Nonfinite semantic teacher scores")
        payload["semantic_probabilities"] = scores
        torch.save(payload, path)
        flat = scores.flatten(1)
        eligible = ((flat.amax(-1) - flat.amin(-1) >= .08) & (flat.amax(-1) >= .55))
        diagnostics.append({"id": record["id"], "split": record["split"],
                            "probabilities": scores.tolist(), "eligible_before_roi": eligible.tolist()})
        if index % 32 == 0:
            atomic_json(out / "status.json", {"phase": "semantic", "completed": index + 1,
                "total": len(records), "heartbeat": time.time()})
            print(args.task, "semantic", index + 1, "/", len(records), flush=True)
    atomic_json(out / "semantic_diagnostics.json", diagnostics)
    files = ["reader_init.pt", "teacher_projection.pt", "text_embeddings.pt", "records.json", "protocol.json"]
    atomic_json(out / "complete.json", {"completed_unix": time.time(), "records": len(records),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "sha256": {name: _sha256(out / name) for name in files},
        "eligible_fraction_before_roi": np.mean([d["eligible_before_roi"] for d in diagnostics], axis=0).tolist()})
    if lock:
        lock.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--semantic-teacher", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--task", choices=("segmentation", "sr"), required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--train-n", type=int, default=1024)
    p.add_argument("--val-n", type=int, default=128)
    p.add_argument("--test-n", type=int, default=447)
    p.add_argument("--human-n", type=int, default=138)
    p.add_argument("--views", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--semantic-batch", type=int, default=8)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--source-commit", default="cc72e88")
    p.add_argument("--deadline", type=float, default=0)
    args = p.parse_args()
    if args.views < 1 or min(args.batch_size, args.semantic_batch) < 1:
        p.error("Need at least one transformed view and positive batch sizes")
    main(args)
