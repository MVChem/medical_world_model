"""Freeze a proportional sample before inference, without answer-based selection."""

import argparse
import importlib.metadata
import json
import math
import os
import random
import shutil
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

from .common import answer_schema, atomic, digest, read, write_rows

PROJECT = Path(__file__).resolve().parents[2]
DATA = Path("/home/data2/chk/data/MIMIC")
ONTOLOGY_URL = "https://raw.githubusercontent.com/baeseongsu/mimic-cxr-vqa/master/mimiccxrvqa/dataset/ans2idx.json"


def sample_rows(data, n, seed):
    if not 0 < n <= len(data):
        raise ValueError("Sample size must be between 1 and dataset size")
    groups = defaultdict(list)
    for row in data:
        groups[row["semantic_type"], row["content_type"]].append(row)
    keys = sorted(groups)
    quotas = {k: n * len(groups[k]) / len(data) for k in keys}
    counts = {k: math.floor(quotas[k]) for k in keys}
    for k in sorted(keys, key=lambda k: (-(quotas[k] - counts[k]), k))[
        : n - sum(counts.values())
    ]:
        counts[k] += 1
    rng = random.Random(seed)
    selected = [
        r
        for k in keys
        for r in rng.sample(sorted(groups[k], key=lambda r: r["idx"]), counts[k])
    ]
    return sorted(selected, key=lambda r: (r["image_id"], r["idx"]))


def audit_training(args, selected, full):
    from medworld.config import load_config
    from medworld.datasets import TASKS, UnifiedData

    cfg = load_config(args.training_run / "config.json")
    data = UnifiedData(cfg)
    if data.metadata != read(args.training_run / "data_protocol.json"):
        raise ValueError("Training data changed since training protocol was frozen")
    sample_patients = {r["subject_id"] for r in selected}
    full_patients = {r["subject_id"] for r in full}
    overlaps = {}
    train_patients = set()
    for task in (*TASKS, "temporal"):
        for split in ("train", "validate", "test"):
            patients = {
                str(r["patient" if task == "temporal" else "subject_id"])
                for r in data.rows(task, split)
            }
            overlaps[f"{task}/{split}"] = {
                "patients": len(patients),
                "sample_overlap": len(patients & sample_patients),
                "full_test_overlap": len(patients & full_patients),
            }
            if split == "train":
                train_patients |= patients
    return {
        "training_run": str(args.training_run.resolve()),
        "data_fingerprint": data.fingerprint,
        "overlap_by_task": overlaps,
        "sample_train_overlap": len(train_patients & sample_patients),
        "full_test_train_overlap": len(train_patients & full_patients),
        "overlap_sample_ids": [
            str(r["idx"]) for r in selected if r["subject_id"] in train_patients
        ],
        "scope": "Active unified training data only; public checkpoint pretraining exposure is unknown.",
    }


def main(args):
    os.umask(0o077)
    run = args.run.resolve()
    if run.exists() and any(run.iterdir()):
        raise FileExistsError("Preparation requires an empty run directory")
    run.mkdir(parents=True, exist_ok=True)
    dataset = (
        args.data / "mimic-ext-mimic-cxr-vqa-1.0.0/MIMIC-Ext-MIMIC-CXR-VQA/dataset"
    )
    full = read(dataset / "test.json")
    selected = sample_rows(full, args.n, args.seed)
    with urllib.request.urlopen(ONTOLOGY_URL, timeout=30) as f:
        ontology_bytes = f.read()
    (run / "official_ans2idx.json").write_bytes(ontology_bytes)
    ans2idx = json.loads(ontology_bytes)
    vocabulary = sorted(ans2idx, key=ans2idx.get)
    assert len(vocabulary) == 110
    assert all(a in ans2idx for r in full for a in r["answer"])
    atomic(run / "vocabulary.json", vocabulary)
    (run / "images").mkdir()
    image_sources = {}
    inputs, refs = [], []
    for row in selected:
        original = args.data / "MIMIC_CXR/files" / row["image_path"]
        image_path = run / "images" / (row["image_id"] + ".png")
        if row["image_id"] not in image_sources:
            with Image.open(original) as im:
                ImageOps.pad(
                    im.convert("RGB"),
                    (512, 512),
                    method=Image.Resampling.BICUBIC,
                    color=(0, 0, 0),
                ).save(image_path)
            image_sources[row["image_id"]] = {
                "path": str(original.resolve()),
                "sha256": digest(original),
            }
        inputs.append(
            {
                "id": str(row["idx"]),
                "image": str(image_path),
                "question": row["question"],
            }
        )
        refs.append(
            {
                "id": str(row["idx"]),
                "patient": row["subject_id"],
                "image_id": row["image_id"],
                "answer": row["answer"],
                "semantic_type": row["semantic_type"],
                "content_type": row["content_type"],
                "template_program": row["template_program"],
                "regional": bool(row["template_arguments"]["object"]),
            }
        )
    write_rows(run / "inputs.jsonl", inputs)
    write_rows(run / "references.jsonl", refs)
    atomic(run / "image_sources.json", image_sources)
    atomic(run / "training_overlap.json", audit_training(args, selected, full))
    split_patients = {
        split: {r["subject_id"] for r in read(dataset / (split + ".json"))}
        for split in ("train", "valid", "test")
    }
    split_audit = {
        f"{a}__{b}": len(split_patients[a] & split_patients[b])
        for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))
    }
    atomic(run / "official_split_audit.json", split_audit)
    from medworld_baselines.base import models

    specs = {s["id"]: s for s in models()}
    inventory = []
    for name in args.models.split(","):
        model = dict(specs[name])
        model.pop("endpoint", None)
        assert model["tp"] == 1
        root = Path(model["path"])
        files = sorted(
            p
            for p in root.iterdir()
            if p.suffix in (".safetensors", ".json", ".jinja", ".txt")
        )
        assert any(p.suffix == ".safetensors" for p in files)
        model["file_sha256"] = {p.name: digest(p) for p in files}
        model["weight_bytes"] = sum(
            p.stat().st_size for p in files if p.suffix == ".safetensors"
        )
        inventory.append(model)
    atomic(run / "models.json", inventory)
    source = run / "source/medworld_vqa"
    shutil.copytree(
        Path(__file__).parent,
        source,
        ignore=shutil.ignore_patterns("runs", "__pycache__"),
    )
    shutil.copy2(PROJECT / "code/medworld/gpu.py", source / "gpu.py")
    (source / "startup").mkdir()
    shutil.copy2(
        PROJECT / "code/medworld_zero_shot/startup/sitecustomize.py",
        source / "startup/sitecustomize.py",
    )
    atomic(
        run / "source_manifest.json",
        {
            str(p.relative_to(run / "source")): digest(p)
            for p in source.rglob("*")
            if p.is_file()
        },
    )
    file_sha256 = {str(p.relative_to(run)): digest(p) for p in run.glob("*.json*")}
    file_sha256.update(
        {str(p.relative_to(run)): digest(p) for p in (run / "images").glob("*.png")}
    )
    protocol = {
        "version": "mimic_cxr_vqa_pilot_v2",
        "created_unix": time.time(),
        "seed": args.seed,
        "n": len(selected),
        "patients": len({r["subject_id"] for r in selected}),
        "images": len(image_sources),
        "full_test_n": len(full),
        "dataset": "MIMIC-Ext-MIMIC-CXR-VQA 1.0.0 official test subset",
        "dataset_sha256": {
            s: digest(dataset / (s + ".json")) for s in ("train", "valid", "test")
        },
        "sampling": "Proportional semantic_type x content_type strata, largest-remainder quotas, fixed seeded random sampling; no answer filtering",
        "semantic_counts": dict(Counter(r["semantic_type"] for r in selected)),
        "content_counts": dict(Counter(r["content_type"] for r in selected)),
        "empty_answers": sum(not r["answer"] for r in selected),
        "ontology_source": ONTOLOGY_URL,
        "input": "One current chest radiograph + question + complete fixed public 110-label answer vocabulary; no reports/EHR/reference answers",
        "image": "512x512 RGB aspect-preserving black pad; Qwen min_pixels=max_pixels=512^2; MedGemma native processor",
        "generation": {
            "temperature": 0,
            "max_tokens": 192,
            "qwen_thinking": False,
            "seed": args.seed,
            "structured_outputs": {"json": answer_schema(vocabulary)}
            if args.structured
            else None,
        },
        "scoring": "Strict JSON label-set exact match and micro F1, case/whitespace normalized; no synonym matching or judge model. Invalid output counts as wrong with a false-positive sentinel.",
        "diagnosis": "Exploratory current-image clinical-content subset: presence, abnormality, attribute, anatomy, size; excludes plane/gender and ALL patients overlapping any active unified training task. Not future diagnosis.",
        "scope": "Sampled zero-shot pilot; not full official benchmark; local implementation of set metrics, not a verified execution of an official scorer",
        "training": "None; unmodified public BF16 checkpoints; public pretraining overlap unknown",
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "vllm", "Pillow", "numpy")
        },
        "file_sha256": file_sha256,
    }
    atomic(run / "protocol.json", protocol)
    print(
        json.dumps(
            {
                k: v
                for k, v in protocol.items()
                if k
                in (
                    "n",
                    "patients",
                    "images",
                    "semantic_counts",
                    "content_counts",
                    "empty_answers",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--training-run", type=Path, required=True)
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--models", default="qwen08b,qwen4b,qwen9b,medgemma4b")
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--seed", type=int, default=20260916)
    p.add_argument("--structured", action=argparse.BooleanOptionalAction, default=True)
    main(p.parse_args())
