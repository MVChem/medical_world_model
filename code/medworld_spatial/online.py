"""Per-batch source decoding and frozen teachers; no intermediate disk files."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from medworld.adaptation import capture_depths
from medworld.datasets import UnifiedData
from medworld.datasets.current import _sha256
from medworld.runtime import load_model
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


def source_canvas(record):
    """Reproduce the frozen cohort's uint8 canvas using its recorded box."""
    y, x, height, width = record["box"]
    with Image.open(record["image"]) as image:
        resized = np.asarray(image.convert("L").resize((width, height), Image.Resampling.BICUBIC))
    canvas = np.zeros((512, 512), dtype=np.uint8)
    canvas[y:y + height, x:x + width] = resized
    return torch.from_numpy(canvas)[None].float() / 255


def low_resolution(hr):
    # Preserve the previous uint8 rounding, including antialiasing and clamping.
    lr = F.interpolate(hr[None], scale_factor=.25, mode="bicubic", align_corners=False, antialias=True)[0]
    return (lr.clamp(0, 1) * 255).round().byte().float() / 255


def human_target(record):
    y, x, height, width = [n // 2 for n in record["box"]]
    target = torch.zeros(2, 256, 256)
    for organ, path in enumerate(record["masks"]):
        with Image.open(path) as image:
            values = np.array(image.convert("L").resize((width, height), Image.Resampling.NEAREST), copy=True)
        target[organ, y:y + height, x:x + width] = torch.from_numpy(values > 0)
    return target


class SourceData:
    """Metadata only in memory; decode original images for every requested batch."""
    def __init__(self, cfg, train_n=4096, val_n=128, test_n=447, human_n=138, selection_seed=42):
        data = UnifiedData(cfg)
        self.fingerprint = data.fingerprint
        self.rows, self.permutations = {}, {}
        dense = data.current.dense
        originals = [json.loads(line) for line in (dense / "observations.jsonl").read_text().splitlines()]
        by_id = {row["id"]: row for row in originals}
        for task in ("segmentation", "sr"):
            for split, count in (("train", train_n), ("validate", val_n), ("test", test_n), ("human_test", human_n)):
                if task == "sr" and split == "human_test":
                    continue
                rows = data.rows(task, split)
                indices = select_indices(rows, count, selection_seed) if count >= 0 else []
                self.rows[task, split] = [by_id[rows[i]["id"]] for i in indices]
        # These are the original fixed supervised labels, never model inputs or
        # newly produced features. Keep the established CXAS supervision intact.
        self.pseudo = np.load(data.current.old / "seg_probs.npy", mmap_mode="r", allow_pickle=False)
        if self.pseudo.shape != (data.current._old_count, 3, 256, 256) or self.pseudo.dtype != np.float16:
            raise ValueError("Original segmentation target shape/dtype changed")
        for rows in self.rows.values():
            for row in rows:
                for path in (row["image"], *row.get("masks", [])):
                    if not Path(path).is_file():
                        raise FileNotFoundError(path)

    def batch(self, task, split, indices):
        rows = [self.rows[task, split][i] for i in indices]
        pixels, targets, masks = [], [], []
        for row in rows:
            hr = source_canvas(row)
            if task == "sr":
                pixel, target, size = low_resolution(hr), hr, 512
            else:
                pixel = F.interpolate(hr[None], (256, 256), mode="area")[0]
                target = (human_target(row) if row["kind"] == "montgomery" else
                          torch.from_numpy(np.array(self.pseudo[row["old_index"]], copy=True)).float())
                size = 256
            valid = torch.zeros(1, size, size)
            y, x, height, width = [n // (512 // size) for n in row["box"]]
            valid[:, y:y + height, x:x + width] = 1
            pixels.append(pixel)
            targets.append(target)
            masks.append(valid)
        return {"pixels": torch.stack(pixels), "targets": torch.stack(targets), "valid": torch.stack(masks),
                "ids": [row["id"] for row in rows], "patients": [str(row["subject_id"]) for row in rows]}

    def training_indices(self, task, offset, batch_size, seed):
        size = len(self.rows[task, "train"])
        if not size:
            raise ValueError(f"Empty {task} training pool")
        indices = []
        for position in range(offset, offset + batch_size):
            epoch, within = divmod(position, size)
            key = (task, epoch, seed)
            if key not in self.permutations:
                entropy = int.from_bytes(hashlib.sha256(f"{seed}:{task}:{epoch}".encode()).digest()[:8], "little")
                self.permutations = {k: v for k, v in self.permutations.items() if k[0] != task}
                self.permutations[key] = np.random.default_rng(entropy).permutation(size)
            indices.append(int(self.permutations[key][within]))
        return indices


class OnlineInputs:
    """Only the current batch is retained; repeated images are recomputed."""
    def __init__(self, args, device):
        self.device, self.views, self.selection_seed = device, args.views, args.selection_seed
        self.model, saved = load_model(args.checkpoint, device)
        self.model.eval().requires_grad_(False)
        self.data = SourceData(self.model.cfg, args.train_n, args.val_n, args.test_n, args.human_n, args.selection_seed)
        if self.data.fingerprint != saved["data_fingerprint"]:
            raise ValueError("Checkpoint and source patient protocols differ")
        self.rows = self.data.rows
        merge = self.model.encoder.vision.config.spatial_merge_size
        reader = SlotReader(self.model.encoder.visual_readouts, self.model.encoder.slot_queries[4:],
                            self.model.encoder.visual_norm, merge)
        self.reader = {"raw_width": self.model.encoder.vision.config.hidden_size, "merge": merge,
                       "queries": reader.queries.detach().cpu(),
                       "state_dict": {k: v.detach().cpu() for k, v in reader.state_dict().items()}}
        del reader
        generator = torch.Generator().manual_seed(9137)
        projection = torch.randn(self.reader["raw_width"], 64, generator=generator)
        self.projection = torch.linalg.qr(projection, mode="reduced").Q.to(device)
        self.teacher = NativeSemanticTeacher(args.semantic_teacher, device, args.semantic_batch)
        self.text = self.teacher.text_embeddings()
        self.protocol = {"input_mode": "on_demand_no_disk_cache", "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": _sha256(args.checkpoint), "data_fingerprint": self.data.fingerprint,
            "semantic_teacher": str(args.semantic_teacher.resolve()),
            "semantic_teacher_sha256": {p.name: _sha256(p) for p in sorted(args.semantic_teacher.iterdir())
                if p.is_file() and (p.suffix in (".json", ".safetensors") or p.name == "merges.txt")},
            "vision_depths_1based": saved["metadata"]["vision_depths_1based"],
            "selection_seed": args.selection_seed, "views": args.views,
            "counts": {f"{task}/{split}": len(rows) for (task, split), rows in self.rows.items()},
            "inputs": "Original source -> recorded 512 canvas; segmentation area 256; SR uint8 antialiased bicubic 128",
            "supervision": "Original fixed CXAS targets and original Montgomery PNG masks",
            "teacher_inputs": "Prepared pixels only; never reports, task targets or SR HR",
            "intermediate_lifetime": "One batch shared by matched variants, discarded before the next batch"}
        self.verified_reader = False

    @torch.no_grad()
    def features(self, batch, *, alignment=True, semantic=True):
        pixels = batch["pixels"]
        raw, slots, grid = extract(self.model, pil_images(pixels), full=True)
        if not self.verified_reader:
            from .model import reader_from_state
            probe = reader_from_state(self.reader).to(self.device)
            restored, attention = probe(raw, grid)
            torch.testing.assert_close(restored, slots[:, 4:], atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(attention.sum((-1, -2)), torch.ones_like(attention[:, :, 0, 0]))
            self.verified_reader = True
            del probe
        result = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in batch.items()}
        result.update(raw=raw, fusion=slots[:, :4], grid=grid)
        if alignment:
            identity = torch.tensor([[1., 0, 0], [0, 1., 0]]).repeat(len(pixels), 1, 1)
            targets = [patch_grid(F.layer_norm(raw[:, -1], (raw.shape[-1],)) @ self.projection,
                                  *grid, self.reader["merge"])]
            thetas = [identity]
            for j in range(self.views):
                transforms = []
                for identity in batch["ids"]:
                    seed = int.from_bytes(hashlib.sha256(f"{self.selection_seed}:{identity}:{j}".encode()).digest()[:8], "little")
                    transforms.append(jitter(torch.Generator().manual_seed(seed)))
                theta = torch.stack(transforms)
                shifted, _, shifted_grid = extract(self.model, pil_images(view(pixels, theta)), full=False)
                if shifted_grid != grid:
                    raise ValueError("Teacher view changed the token grid")
                targets.append(patch_grid(F.layer_norm(shifted[:, -1], (shifted.shape[-1],)) @ self.projection,
                                          *grid, self.reader["merge"]))
                thetas.append(theta)
            result.update(teacher_features=torch.stack(targets, 1), thetas=torch.stack(thetas, 1).to(self.device))
        if semantic:
            probabilities = self.teacher.scores(pixels).to(self.device)
            if not torch.isfinite(probabilities).all():
                raise ValueError("Nonfinite semantic probabilities")
            result["semantic_probabilities"] = probabilities
        return result

    def batch(self, task, split, indices, *, alignment=True, semantic=True):
        return self.features(self.data.batch(task, split, indices), alignment=alignment, semantic=semantic)

    def training_batch(self, task, offset, batch_size, seed, **kwargs):
        indices = self.data.training_indices(task, offset, batch_size, seed)
        return self.batch(task, "train", indices, **kwargs)
