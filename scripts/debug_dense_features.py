"""Trace real Qwen visual/LM features and the separate Stage 1 slot readout.

Writes shape metadata only; no training, cache replacement, or patient text export.
Run with the project's Python environment and an idle CUDA_VISIBLE_DEVICES.
"""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
STAGE = PROJECT / "code/medworld_stage1"
DENSE = PROJECT / "code/medworld_dense_baselines"
sys.path.insert(0, str(STAGE))
import bootstrap
from medworld_common.runtime import atomic_json, digest


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def describe(x):
    return {"shape": list(x.shape), "dtype": str(x.dtype),
            "finite": bool(torch.isfinite(x).all())}


def rows_at(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def inspect_dense(run):
    # Load the exact extractor/heads archived with the completed run.
    source = run / "source"
    os.environ["MEDWORLD_PROJECT"] = str(PROJECT)
    sys.path.insert(0, str(source))
    features = module_at("debug_dense_extractor", source / "features.py")
    heads = module_at("debug_dense_heads", source / "heads.py")

    specs = json.loads((run / "models.json").read_text())
    configurations = {}
    for spec in specs:
        config = json.loads((Path(spec["path"]) / "config.json").read_text())
        meta = json.loads((run / spec["id"] / "features_complete.json").read_text())
        configurations[spec["id"]] = {
            "vision_config": config["vision_config"],
            "text_width": config["text_config"]["hidden_size"],
            "cached_lm_native_shapes": meta["native_shapes"],
            "cached_aligned_shape": meta["shape"],
        }
    spec = next(s for s in specs if s["id"] == "qwen08b")
    model, proc = features.load_vlm(spec)
    rows = rows_at(run / "data/observations.jsonl")
    index = next(i for i, r in enumerate(rows) if r["split"] == "validate"
                 and "segmentation" in r["tasks"] and "sr" in r["tasks"])
    captured = {}

    def vision_hook(module, inputs, output):
        captured["vision_premerger"] = output.last_hidden_state.detach().clone()
        captured["vision_postmerger"] = output.pooler_output.detach().clone()

    handle = model.model.visual.register_forward_hook(vision_hook)
    records = {}
    with torch.inference_mode():
        for branch, task, image_file in [("hr", "segmentation", "images.npy"),
                                         ("lr", "sr", "lr_images.npy")]:
            arr = np.array(np.load(run / "data" / image_file, mmap_mode="r")[index], copy=True)
            inputs = features.model_inputs(proc, Image.fromarray(arr).convert("RGB"), "cuda")
            output = model.model(**inputs, use_cache=False, return_dict=True)
            native = output.last_hidden_state[0][inputs["input_ids"][0] == model.config.image_token_id]
            aligned = features.align_hidden(native)
            cached = torch.from_numpy(np.array(np.load(run / "qwen08b" / f"{branch}_hidden.npy",
                                                      mmap_mode="r")[index], copy=True)).cuda()
            grid = inputs["image_grid_thw"][0].tolist()
            pre, post = captured["vision_premerger"], captured["vision_postmerger"]
            assert pre.shape == (int(np.prod(grid)), 768)
            assert post.shape == native.shape == (int(np.prod(grid)) // 4, 1024)
            assert aligned.shape == cached.shape == (64, 1024)
            # This API stops at the vision tower/merger; it does not run the LM.
            direct = model.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
            assert torch.equal(direct.last_hidden_state, pre)
            assert torch.equal(direct.pooler_output[0], post)
            # Qwen premerger patches are ordered in 2x2 merge blocks, not raster order.
            t, h, w = grid
            raster = pre.reshape(t, h//2, w//2, 2, 2, 768).permute(0, 5, 1, 3, 2, 4).reshape(t, 768, h, w)
            # Matching width does not identify matching representation.
            assert not torch.equal(post, native)
            item = {"input_canvas": list(arr.shape), "grid_thw": grid,
                    "pixel_values": describe(inputs["pixel_values"]),
                    "vision_premerger": describe(pre), "vision_postmerger": describe(post),
                    "vision_premerger_raster_map": describe(raster),
                    "direct_vision_api_matches_hook": True,
                    "lm_image_tokens": describe(native), "aligned": describe(aligned),
                    "cache_exact_equal": torch.equal(aligned.float(), cached.float()),
                    "cache_max_abs_error": float((aligned.float()-cached.float()).abs().max()),
                    "cache_mean_abs_error": float((aligned.float()-cached.float()).abs().mean()),
                    "cache_cosine_similarity": float(F.cosine_similarity(aligned.float().flatten(), cached.float().flatten(), dim=0)),
                    "vision_postmerger_vs_lm_mean_abs": float((post.float()-native.float()).abs().mean()),
                    "heads": {}}
            vjepa = torch.from_numpy(np.array(np.load(run / "data" / f"vjepa_{branch}.npy",
                                                     mmap_mode="r")[index:index+1], copy=True)).cuda().float()
            item["vjepa_cache"] = describe(vjepa)
            for variant in ["image", "vjepa", "vjepa_adapter"]:
                # Shape probe only: these heads are newly initialized, not evaluated checkpoints.
                head = heads.DenseHead(task, variant).cuda().eval()
                x = torch.from_numpy(arr).cuda().float()[None, None] / 255
                if variant != "image":
                    x = vjepa
                elif task == "segmentation":
                    x = F.interpolate(x, (256, 256), mode="area")
                fused = {}

                def fuse_hook(module, args):
                    fused["fusion_input"] = describe(args[0])

                hook = head.fuse.register_forward_pre_hook(fuse_hook)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = head(x, cached[None])
                hook.remove()
                expected = (1, 3, 256, 256) if task == "segmentation" else (1, 1, 512, 512)
                assert pred.shape == expected and torch.isfinite(pred).all()
                item["heads"][variant] = {"visual_input": describe(x), **fused, "output": describe(pred)}
                del head
            records[branch] = item
            print("dense", branch, json.dumps(item), flush=True)
    handle.remove()
    return {"model": "Qwen3.5-0.8B", "validation_index": index,
            "source_sha256": {n: digest(source / n) for n in ["features.py", "heads.py"]},
            "all_model_configurations": configurations, "branches": records,
            "head_probe": "fresh heads for shape checks only; no performance evaluation"}


def inspect_slots(run):
    sys.path.insert(0, str(STAGE))
    from slot44_networks import Slot44Model
    from cache import downsample
    cfg = json.loads((run / "config.json").read_text())
    model = Slot44Model(cfg).cuda().eval()
    checkpoint = run / "joint/checkpoint_final.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_compact(saved["model"])
    step = saved.get("step")
    del saved
    root = Path(cfg["cache"])
    rows = rows_at(root / "observations.jsonl")
    overrides = json.loads(Path(cfg["selection"]).read_text())["patient_split_overrides"]
    valid = np.load(root / "seg_valid_qc.npy")
    index = next(i for i, r in enumerate(rows)
                 if overrides.get(str(r["subject_id"]), r["split"]) == "validate"
                 and r["tasks"]["segmentation"] and r["tasks"]["sr"] and valid[i])
    ids = model.tokenizer(rows[index]["report"], add_special_tokens=False)["input_ids"][:cfg["report_tokens"]]
    ids = torch.tensor([ids], dtype=torch.long, device="cuda")
    text_mask = torch.ones_like(ids)
    if ids.shape[1] == 0:
        ids = torch.tensor([[model.tokenizer.eos_token_id]], device="cuda")
        text_mask = torch.zeros_like(ids)
    b = {"ids": ids, "text_mask": text_mask}
    for branch in ["hr", "lr"]:
        b[f"{branch}_features"] = torch.from_numpy(np.array(np.load(root / f"{branch}_features.npy",
                                           mmap_mode="r")[index:index+1], copy=True)).cuda()
    hr = torch.from_numpy(np.array(np.load(root / "images.npy", mmap_mode="r")[index:index+1], copy=True))
    b["lr"] = downsample(hr.cuda().float()[:, None]/255, cfg["scale"])
    b["seg_image"] = F.interpolate(hr.cuda().float()[:, None]/255, (256, 256), mode="bilinear", align_corners=False)
    captured = {}

    def adapter_hook(module, inputs, output):
        captured["adapter"] = describe(output)

    def backbone_hook(module, args, kwargs, output):
        captured["sequence"] = describe(kwargs["inputs_embeds"])
        captured["last_hidden"] = output.last_hidden_state

    handles = [model.encoder.adapter.register_forward_hook(adapter_hook),
               model.encoder.backbone.register_forward_hook(backbone_hook, with_kwargs=True)]
    records = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for task in ["segmentation", "sr"]:
            s = model.encode(b, task)
            exact = torch.equal(s, captured["last_hidden"][:, -8:])
            routed = model.route(s, task)
            assert exact and routed.shape == (1, 4, 1024)
            pred = model.segmentation(b["seg_image"], routed) if task == "segmentation" else model.sr(b["lr"], routed)
            item = {"feature_source": "lr_features" if task == "sr" else "hr_features",
                    "adapter": captured["adapter"], "sequence": captured["sequence"],
                    "full_slots": describe(s), "routed_slots": describe(routed),
                    "equals_last_hidden_final_8_tokens": exact,
                    "output": [describe(p) for p in pred] if isinstance(pred, tuple) else describe(pred)}
            if task == "sr":
                changed = dict(b, hr_features=b["hr_features"] + 100)
                item["hr_features_do_not_affect_sr_slots"] = torch.equal(s, model.encode(changed, task))
                assert item["hr_features_do_not_affect_sr_slots"]
            records[task] = item
            print("slots", task, json.dumps(item), flush=True)
    for handle in handles:
        handle.remove()
    return {"checkpoint": str(checkpoint), "step": step, "validation_index": index,
            "language_layers": len(model.encoder.backbone.base_model.model.layers), "tasks": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-run", type=Path, default=DENSE / "runs/dense_20260912")
    parser.add_argument("--slot-run", type=Path, default=STAGE / "runs/slot44_20260911")
    parser.add_argument("--out", type=Path, default=PROJECT / "results/feature_debug_20260912/shapes.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20260912)
    result = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "dense": inspect_dense(args.dense_run)}
    atomic_json(args.out, result)
    gc.collect()
    torch.cuda.empty_cache()
    result["slots"] = inspect_slots(args.slot_run)
    atomic_json(args.out, result)
    print("Saved", args.out, flush=True)


if __name__ == "__main__":
    main()
