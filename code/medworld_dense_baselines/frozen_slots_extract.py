"""Frozen native-vision slots 5--8 for the September 13 dense-task probe.

Only the checkpoint's vision tower is instantiated. Four block outputs are
spatially averaged in float32 and aligned without learned parameters. The LR
branch reads only the prepared LR array; segmentation reads the HR array.
"""
import argparse
import collections
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import PROJECT, atomic, digest, load_model_spec, read_rows

SLOT_WIDTH = 1024
BRANCHES = ("hr", "lr")


def block_indices(depth):
    """Quarter, half, three-quarter, and final block outputs in increasing depth."""
    if depth < 4:
        raise ValueError("at least four blocks are required for four distinct slots")
    return [math.ceil((i + 1) * depth / 4) - 1 for i in range(4)]


def pool_slot(hidden, width=SLOT_WIDTH):
    """One image's native tokens -> one fixed-width vector, with no parameters."""
    if isinstance(hidden, (tuple, list)):
        hidden = hidden[0]
    if hidden.ndim == 3:
        if hidden.shape[0] != 1:
            raise ValueError("slot extraction requires exactly one image per forward")
        hidden = hidden[0]
    if hidden.ndim != 2 or min(hidden.shape) < 1 or not torch.isfinite(hidden).all():
        raise ValueError("invalid native vision block output")
    pooled = hidden.float().mean(dim=0)
    native_width = pooled.shape[0]
    if native_width < width:
        return F.pad(pooled, (0, width - native_width))
    indices = torch.floor((torch.arange(width, device=hidden.device) + .5) * native_width / width).long()
    return pooled.index_select(0, indices)


def required_branches(rows):
    return np.array([["segmentation" in r["tasks"], "sr" in r["tasks"]] for r in rows], dtype=bool)


def branch_image(branch, index, images, lr_images):
    # Deliberately select the LR source before indexing either array.
    if branch not in BRANCHES:
        raise ValueError(f"unknown image branch: {branch}")
    source = lr_images if branch == "lr" else images
    arr = np.array(source[index], copy=True)
    expected = (128, 128) if branch == "lr" else (512, 512)
    if arr.dtype != np.uint8 or arr.shape != expected:
        raise ValueError(f"{branch} input must be prepared uint8 {expected}; got {arr.dtype} {arr.shape}")
    return Image.fromarray(arr).convert("RGB")


def checkpoint_layout(spec):
    root = Path(spec["path"])
    config = json.loads((root / "config.json").read_text())
    actual_config_hash = digest(root / "config.json")
    if actual_config_hash != spec["config_sha256"]:
        raise ValueError(f"model config changed: {spec['id']}")
    index_files = sorted(root.glob("*.safetensors.index.json"))
    if len(index_files) != 1:
        raise ValueError("expected one local safetensors index")
    mapping = json.loads(index_files[0].read_text())["weight_map"]
    prefixes = ("model.visual.", "visual.") if spec["family"] == "qwen" else (
        "vision_tower.vision_model.", "model.vision_tower.vision_model.", "vision_tower.")
    prefix = next((p for p in prefixes if any(k.startswith(p) for k in mapping)), None)
    if prefix is None:
        raise ValueError("checkpoint contains no recognized native vision tower")
    selected = {k: v for k, v in mapping.items() if k.startswith(prefix)}
    shards = []
    for name in sorted(set(selected.values())):
        path = root / name
        stat = path.stat()
        shards.append(dict(name=name, resolved=str(path.resolve()), bytes=stat.st_size,
                           mtime_ns=stat.st_mtime_ns))
    return config["vision_config"], prefix, selected, dict(
        config_sha256=actual_config_hash, index_sha256=digest(index_files[0]),
        vision_prefix=prefix, selected_tensors=len(selected), shards=shards)


def tensor_hash(state, exclude_merger=False):
    """Hash sorted names, dtype, shape and exact stored bytes, independent of shards."""
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if exclude_merger and name.startswith("merger."):
            continue
        metadata = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":"))
        h.update(metadata.encode() + b"\0")
        h.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def load_vision(spec, device="cuda:0", dtype=torch.bfloat16):
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import AutoImageProcessor
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel

    vision_config, prefix, selected, checkpoint = checkpoint_layout(spec)
    is_qwen = spec["family"] == "qwen"
    config_cls = Qwen3_5VisionConfig if is_qwen else SiglipVisionConfig
    model_cls = Qwen3_5VisionModel if is_qwen else SiglipVisionModel
    config = config_cls(**vision_config)
    config._attn_implementation = "sdpa"
    # Parameters are allocated only when their selected vision tensors are read;
    # non-persistent rotary/position buffers are initialized normally on CPU.
    with init_empty_weights(include_buffers=False):
        model = model_cls(config)
    state = {}
    for shard in sorted(set(selected.values())):
        with safe_open(Path(spec["path"]) / shard, framework="pt", device="cpu") as f:
            for key in sorted(k for k, v in selected.items() if v == shard):
                state[key[len(prefix):]] = f.get_tensor(key)
    dtypes = sorted({str(t.dtype) for t in state.values()})
    # Native FP8 Qwen checkpoints retain the entire vision tower in BF16. Fail
    # closed if a future checkpoint actually quantizes vision weights.
    if any(t.dtype not in (torch.bfloat16, torch.float16, torch.float32) for t in state.values()):
        raise ValueError(f"unsupported native vision weight dtype: {dtypes}")
    weight_hash = tensor_hash(state)
    backbone_hash = tensor_hash(state, exclude_merger=is_qwen)
    # Cast checkpoint parameters only. Native non-persistent rotary-frequency
    # buffers must remain float32, as in the normal pretrained loader.
    model.load_state_dict({k: v.to(dtype=dtype) for k, v in state.items()}, strict=True, assign=True)
    model = model.to(device=device).eval().requires_grad_(False)
    del state
    if any(p.requires_grad for p in model.parameters()) or any(p.is_meta for p in model.parameters()):
        raise AssertionError("native vision model must be fully loaded and frozen")
    proc = AutoImageProcessor.from_pretrained(spec["path"], local_files_only=True)
    if is_qwen:
        proc.size = {"longest_edge": 512 * 512, "shortest_edge": 256 * 256}
        blocks = model.blocks
        module_prefix = "blocks"
    else:
        proc.do_pan_and_scan = False
        blocks = model.encoder.layers
        module_prefix = "encoder.layers"
    indices = block_indices(len(blocks))
    native_width = int(vision_config["hidden_size"])
    metadata = dict(
        checkpoint=checkpoint, vision_config=vision_config,
        vision_model_class=model.__class__.__name__, native_width=native_width,
        vision_parameters=sum(p.numel() for p in model.parameters()),
        vision_backbone_parameters=sum(p.numel() for name, p in model.named_parameters()
                                       if not (is_qwen and name.startswith("merger."))),
        vision_weights_sha256=weight_hash, vision_backbone_weights_sha256=backbone_hash,
        checkpoint_vision_dtypes=dtypes, inference_dtype=str(dtype), attention="sdpa",
        depth=len(blocks), block_indices_zero_based=indices,
        block_indices_one_based=[i + 1 for i in indices],
        hook_modules=[f"{module_prefix}.{i}" for i in indices],
        hook_stage="block output before final vision norm/projector; image tokens only",
        slot_ids=[5, 6, 7, 8], slot_shape=[4, SLOT_WIDTH],
        pooling="spatial mean of all native image tokens, accumulated in float32",
        channel_alignment=(f"native {native_width} channels then {SLOT_WIDTH - native_width} right zeros"
                           if native_width < SLOT_WIDTH else
                           "1024 equal-width channel-bin centers, order preserving"),
        processor=json.loads(proc.to_json_string()),
        frozen_vision=True, trainable_slot_parameters=0, language_model_loaded=False,
        interpretation="checkpoint vision-tower comparison; language-model parameter count is not exercised",
    )
    return FrozenVisionSlots(model, proc, spec["family"], blocks, indices), metadata


class FrozenVisionSlots:
    def __init__(self, model, processor, family, blocks, indices):
        self.model, self.processor, self.family = model, processor, family
        self.slots = {}
        self.native_shapes = {}
        self.handles = [blocks[index].register_forward_hook(self._capture(i)) for i, index in enumerate(indices)]

    def _capture(self, slot_index):
        def capture(_module, _inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            self.native_shapes[slot_index] = list(hidden.shape)
            self.slots[slot_index] = pool_slot(hidden)
        return capture

    @torch.inference_mode()
    def __call__(self, image):
        self.slots.clear()
        self.native_shapes.clear()
        reference = next(self.model.parameters())
        inputs = self.processor(images=image, return_tensors="pt")
        pixels = inputs["pixel_values"].to(device=reference.device, dtype=reference.dtype)
        if self.family == "qwen":
            self.model(hidden_states=pixels, grid_thw=inputs["image_grid_thw"].to(reference.device))
        else:
            if pixels.shape[0] != 1:
                raise ValueError("native Gemma processor unexpectedly generated multiple crops")
            self.model(pixel_values=pixels)
        if set(self.slots) != set(range(4)):
            raise AssertionError("not all four intermediate blocks were captured")
        result = torch.stack([self.slots[i] for i in range(4)])
        if result.requires_grad or not torch.isfinite(result).all():
            raise ValueError("slot values must be frozen and finite")
        array = result.cpu().numpy().astype(np.float16)
        if not np.isfinite(array).all():
            raise ValueError("slot values overflowed the float16 cache")
        return array


def cache_array(path, shape):
    if path.exists():
        arr = np.load(path, mmap_mode="r+")
        if arr.shape != shape or arr.dtype != np.float16:
            raise ValueError(f"incompatible slot cache: {path}")
        return arr
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
    arr[:] = np.nan  # Missing/unneeded entries must never silently become zero-slot controls.
    arr.flush()
    return arr


def extract(run, mid, limit=None, data=None, device="cuda:0"):
    import transformers
    torch.set_num_threads(4)
    os.umask(0o077)
    run = Path(run).resolve()
    data = Path(data).resolve() if data else run / "data"
    out = run / mid
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "slots_extract.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = read_rows(data / "observations.jsonl")
    if any(r["index"] != i for i, r in enumerate(rows)):
        raise ValueError("observation indices must match cache row order")
    manifest = json.loads((data / "manifest.json").read_text())
    cohort_hash = digest(data / "observations.jsonl")
    if manifest["cohort_sha256"] != cohort_hash:
        raise ValueError("prepared cohort manifest no longer matches observations")
    for filename, key in [("images.npy", "image_sha256"), ("lr_images.npy", "lr_image_sha256")]:
        if digest(data / filename) != manifest[key]:
            raise ValueError(f"prepared image array no longer matches manifest: {filename}")
    required = required_branches(rows)
    spec = load_model_spec(mid)
    _, _, _, checkpoint = checkpoint_layout(spec)
    contract = dict(
        version=1, model_id=mid, model_label=spec["label"],
        shape=[len(rows), 4, SLOT_WIDTH], dtype="float16", slot_ids=[5, 6, 7, 8],
        data_path=str(data), cohort_sha256=cohort_hash, manifest_sha256=digest(data / "manifest.json"),
        image_sha256=manifest["image_sha256"], lr_image_sha256=manifest["lr_image_sha256"],
        checkpoint=checkpoint, extractor_code_sha256=digest(Path(__file__)),
        common_code_sha256=digest(Path(__file__).with_name("common.py")),
        transformers_version=transformers.__version__, torch_version=torch.__version__,
        branch_columns=list(BRANCHES), required_counts=dict(zip(BRANCHES, required.sum(0).tolist())),
        segmentation_input="prepared HR uint8 512x512; Montgomery HR included",
        sr_input="only prepared LR uint8 128x128; native processor may resize this LR image",
        slot_training=False, language_model_loaded=False,
        alignment="mean-pool 4 intermediate vision blocks; bin-center channels to1024 or right zero-pad",
    )
    contract_path = out / "slot_contract.json"
    done_path = out / "slots_done.npy"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("slot cache contract changed; use a fresh output run")
    if done_path.exists() and not contract_path.exists():
        raise ValueError("slot completion cache exists without provenance")
    atomic(contract_path, contract)
    done = np.load(done_path) if done_path.exists() else np.zeros_like(required)
    if done.shape != required.shape or done.dtype != bool or np.any(done & ~required):
        raise ValueError("invalid slot completion mask")
    arrays = {b: cache_array(out / f"{b}_slots.npy", tuple(contract["shape"])) for b in BRANCHES}
    for j, branch in enumerate(BRANCHES):
        if not np.isfinite(arrays[branch][done[:, j]]).all():
            raise ValueError("completed slot cache contains missing or nonfinite data")
    complete_path = out / "slots_complete.json"
    if np.all(done[required]) and complete_path.exists():
        previous = json.loads(complete_path.read_text())
        if not previous.get("complete") or previous["contract_sha256"] != digest(contract_path):
            raise ValueError("slot completion marker disagrees with contract")
        if previous["model_metadata_sha256"] != digest(out / "slots_model.json"):
            raise ValueError("completed slot-model provenance changed")
        for branch in BRANCHES:
            if previous[f"{branch}_slots_sha256"] != digest(out / f"{branch}_slots.npy"):
                raise ValueError(f"completed {branch} slot cache changed")
        print("slots already complete", mid, flush=True)
        return
    images = np.load(data / "images.npy", mmap_mode="r")
    lr_images = np.load(data / "lr_images.npy", mmap_mode="r")
    if len(images) != len(rows) or len(lr_images) != len(rows):
        raise ValueError("prepared image arrays do not match cohort")
    extractor, metadata = load_vision(spec, device=device)
    metadata_path = out / "slots_model.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError("loaded native-vision provenance changed")
    atomic(metadata_path, metadata)
    started = time.time()
    completed = 0
    native_shapes = collections.defaultdict(set)
    with torch.inference_mode():
        for i, row in enumerate(rows):
            for j, branch in enumerate(BRANCHES):
                if done[i, j] or not required[i, j]:
                    continue
                image = branch_image(branch, i, images, lr_images)
                arrays[branch][i] = extractor(image)
                native_shapes[branch].add(tuple(tuple(extractor.native_shapes[k]) for k in range(4)))
                arrays[branch].flush()
                done[i, j] = True
                np.save(out / "slots_done.tmp.npy", done)
                os.replace(out / "slots_done.tmp.npy", done_path)
                completed += 1
                if completed == 1 or completed % 20 == 0:
                    progress = dict(done=int(done[required].sum()), total=int(required.sum()),
                                    done_by_branch=dict(zip(BRANCHES, done.sum(0).tolist())),
                                    seconds=time.time() - started, forward_calls_this_session=completed,
                                    native_shapes={k: sorted(v) for k, v in native_shapes.items()},
                                    physical_gpus=os.environ.get("CUDA_VISIBLE_DEVICES"))
                    atomic(out / "slots_progress.json", progress)
                    print("slots", mid, progress["done"], "/", progress["total"],
                          "seconds", round(progress["seconds"], 1), flush=True)
                if limit is not None and completed >= limit and not np.all(done[required]):
                    return
    if not np.all(done[required]):
        raise AssertionError("required slots are incomplete")
    atomic(complete_path, dict(
        complete=True, contract_sha256=digest(contract_path), model_metadata_sha256=digest(metadata_path),
        done=int(done[required].sum()), total=int(required.sum()),
        hr_slots_sha256=digest(out / "hr_slots.npy"), lr_slots_sha256=digest(out / "lr_slots.npy"),
        native_shapes={k: sorted(v) for k, v in native_shapes.items()},
        forward_calls_this_session=completed, seconds=time.time() - started))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, help="stop after this many new image forwards; cache resumes")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    extract(args.run, args.model, args.limit, args.data, args.device)
