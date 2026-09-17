"""Freeze pixels, labels, patient holdouts and inputs from the live training protocol."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time

from .common import PROJECT, atomic, digest, models, read, rows, write_rows, temporal_input


def prepare(args):
    from medworld.config import load_config
    from medworld.datasets import UnifiedData
    from transformers import AutoTokenizer
    os.umask(0o077)
    run, training = args.run.resolve(), args.training_run.resolve()
    if run.exists() and any(run.iterdir()):
        raise FileExistsError("Use a fresh baseline run")
    cfg = load_config(training / "config.json")
    data = UnifiedData(cfg)
    if data.metadata != read(training / "data_protocol.json"):
        raise ValueError("Baseline data differs from the running model's frozen protocol")
    tokenizer = AutoTokenizer.from_pretrained(cfg["qwen"], local_files_only=True)
    (run / "images").mkdir(parents=True)
    image_paths = {}
    def save_image(namespace, identity, image):
        key = namespace + ":" + identity
        if key not in image_paths:
            filename = hashlib.sha256(key.encode()).hexdigest() + ".png"
            path = run / "images" / filename
            image.convert("RGB").save(path)
            image_paths[key] = str(path)
        return image_paths[key]
    inputs1, refs1 = [], []
    for pair in data.rows("temporal", "test"):
        source, target = [data.temporal.lookup[pair[side]] for side in ("source", "target")]
        item = temporal_input(pair, source, tokenizer, cfg["context_tokens"])
        image_key = "temporal:" + pair["source"]
        if image_key not in image_paths:
            image, _ = data.temporal._observation(pair["source"])
            save_image("temporal", pair["source"], image)
        item["image"] = image_paths[image_key]
        inputs1.append(item)
        refs1.append(dict(id=pair["id"], patient=str(pair["patient"]), direction=pair["direction"],
                          delta_hours=pair["delta_hours"], current_report=source["report"],
                          target_report=target["report"], labels=target["labels"], current_labels=source["labels"]))
    inputs2, refs2 = {}, {}
    for task in ("classification", "report"):
        for row in data.rows(task, "test"):
            example = data.current._example(task, "test", row)
            item = inputs2.setdefault(row["id"], dict(id=row["id"], patient=str(row["subject_id"]),
                image=save_image("current", row["id"], example["image"]), classification=False, report_generation=False))
            ref = refs2.setdefault(row["id"], dict(id=row["id"], patient=str(row["subject_id"])))
            item["classification" if task == "classification" else "report_generation"] = True
            ref["labels" if task == "classification" else "report"] = row["labels"] if task == "classification" else row["report_target"]
    original_inputs = rows(data.current.raw / "cohort/table2_inputs_test.jsonl")
    if set(inputs2) != {r["id"] for r in original_inputs}:
        raise ValueError("Current input membership differs from the unified cohort")
    inputs2 = {r["id"]:inputs2[r["id"]] for r in original_inputs}
    for name, values in (("table1_inputs", inputs1), ("table1_references", refs1),
                         ("table2_inputs", list(inputs2.values())), ("table2_references", list(refs2.values()))):
        write_rows(run / "cohort" / f"{name}_test.jsonl", values)
    inventory = []
    for name in args.models.split(","):
        model = dict(next(m for m in models() if m["id"] == name))
        if model["tp"] != 1:
            raise ValueError("This protocol only schedules unquantized single-GPU checkpoints")
        model.pop("endpoint", None)
        root = Path(model["path"])
        model["config_sha256"] = digest(root / "config.json")
        weights = sorted(root.glob("*.safetensors"))
        if not weights or any(not p.exists() for p in weights):
            raise FileNotFoundError(f"Missing model weights: {name}")
        model["weight_files"] = [dict(name=p.name, bytes=p.stat().st_size, resolved=str(p.resolve()), sha256=digest(p)) for p in weights]
        model["processor_sha256"] = {p.name:digest(p) for p in root.glob("*.json") if "model.safetensors" not in p.name}
        inventory.append(model)
    atomic(run / "models.json", inventory)
    atomic(run / "training_config.json", cfg)
    atomic(run / "data_protocol.json", data.metadata)
    # Freeze all executable dependencies used by the queue and the unified dataset.
    source = run / "source"
    for package in ("medworld_zero_shot", "medworld_baselines", "medworld", "medworld_common"):
        shutil.copytree(PROJECT / "code" / package, source / package,
                        ignore=shutil.ignore_patterns("runs", "vendor", "__pycache__", "*.pyc"))
    clinical = source / "medworld_table1"
    clinical.mkdir()
    for name in ("clinical.py", "metrics.py", "common.py"):
        shutil.copy2(PROJECT / "code/medworld_table1" / name, clinical / name)
    for name in ("weights", "metric_vendor", "vendor"):
        (clinical / name).symlink_to(PROJECT / "code/medworld_table1" / name, target_is_directory=True)
    official = source / "medworld_baselines/green_official"
    official.mkdir(exist_ok=True)
    for name in ("utils.py", "green.py"):
        shutil.copy2(PROJECT / "code/medworld_baselines/vendor/GREEN/green_score" / name, official / name)
    atomic(run / "source_manifest.json", {str(p.relative_to(source)):digest(p) for p in source.rglob("*") if p.is_file() and not any(x in p.parts for x in ("weights", "vendor", "metric_vendor"))})
    counts = dict(table1_forward=sum(r["direction"] == "forward" for r in inputs1),
                  table1_backward=sum(r["direction"] == "backward" for r in inputs1),
                  table1_patients=len({r["patient"] for r in inputs1}),
                  classification=sum(r["classification"] for r in inputs2.values()),
                  report=sum(r["report_generation"] for r in inputs2.values()))
    if counts != dict(table1_forward=297, table1_backward=297, table1_patients=94, classification=353, report=507):
        raise ValueError(f"Unexpected unified test counts: {counts}")
    files = list((run / "cohort").glob("*")) + list((run / "images").glob("*"))
    files += [run / name for name in ("models.json", "training_config.json", "data_protocol.json", "source_manifest.json")]
    protocol = dict(version="unified_zero_shot_v1", created_unix=time.time(), training="none; public BF16 checkpoints; no project adapters",
        training_run=str(training), training_config_sha256=digest(training / "config.json"), data_fingerprint=data.fingerprint,
        counts=counts, generation=dict(temperature=0, max_new_tokens=384, qwen_thinking=False),
        ours_evaluation_required_overrides={"max_new_tokens":384},
        table1_input="source image + exact Qwen-capped observation text + signed actual time; no EHR or target evidence",
        table1_labels="Exact target labels from unified temporal observations; never relabel probability references per model",
        table2_input="Exact cached RGB canvases used by unified training; image only; no report",
        image="Common exact 512-square input canvases; Qwen processor min=max=256^2 as in training; MedGemma native processor. Native visual compute differs; Ours also uses JEPA at 384.",
        probability="Unmodified first-token Yes/No log probabilities normalized over the two complete single-token candidates; fixed reference-only masks; no calibration",
        ece_bins=10, direction_f1="unavailable: adjudicated disease/laterality labels missing",
        vqa="pending official data; derived positive-only QA excluded", grounding="pending MS-CXR data; anatomy localization excluded",
        segmentation="N/A: native zero-shot VLM has no pixel decoder", sr="N/A: native zero-shot VLM has no SR decoder",
        historical_scores="0911 protocol used EHR/horizon and different pixel preparation; do not mix",
        public_pretraining_overlap="unknown", model_order=[m["id"] for m in inventory],
        versions={p:importlib.metadata.version(p) for p in ("torch", "transformers", "vllm", "numpy", "Pillow")},
        file_sha256={str(p.relative_to(run)):digest(p) for p in files})
    atomic(run / "protocol.json", protocol)
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--training-run", type=Path, required=True)
    p.add_argument("--models", default="qwen08b,medgemma4b,qwen4b,qwen9b")
    prepare(p.parse_args())
