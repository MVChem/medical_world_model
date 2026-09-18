"""Freeze the dense source and emit jobs for code/medworld_common/overnight_queue.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

from common import PROJECT, atomic, digest
from overnight_dense_report import MODELS, report


def make_plan(run, source, python, seeds, epochs, stop_at, microbatch=8, snapshot_name="source_dense_v3",
              qwen_extract_batch=64, medgemma_extract_batch=1):
    # Resolving the venv executable symlink selects its base interpreter and
    # silently loses all packages installed in the virtual environment.
    run, source, python = Path(run).resolve(), Path(source).resolve(), Path(python).absolute()
    manifest = json.loads((run / "data/manifest.json").read_text())
    snapshot = run / snapshot_name
    snapshot.mkdir(exist_ok=True)
    filenames = ("common.py", "heads.py", "frozen_slots_train.py", "frozen_slots_extract.py",
                 "overnight_dense_report.py", "overnight_dense_plan.py", "expanded_data.py", "prepare.py", "features.py")
    for name in filenames:
        destination = snapshot / name
        if destination.exists() and digest(destination) != digest(source / name):
            raise ValueError(f"Existing immutable source snapshot differs: {destination}")
        if not destination.exists():
            shutil.copy2(source / name, destination)
    model_source = PROJECT / "code/medworld_dense_baselines/runs/frozen_slots_20260913/models.json"
    if (run / "models.json").exists() and digest(run / "models.json") != digest(model_source):
        raise ValueError("Expanded run model list differs from the prior fixed six checkpoints")
    if not (run / "models.json").exists():
        shutil.copy2(model_source, run / "models.json")
    env = dict(MEDWORLD_PROJECT=str(PROJECT), MEDWORLD_MODEL_FILE=str(run / "models.json"),
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
               HF_HUB_DISABLE_PROGRESS_BARS="1", TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR="1")
    jobs = []
    for model in MODELS:
        artifact = str(run / model / "slots_complete.json")
        extract_batch = medgemma_extract_batch if model.startswith("medgemma") else qwen_extract_batch
        jobs.append(dict(id=f"dense_extract_{model}", kind="dense_extract", model=model,
                         argv=[str(python), "-u", str(snapshot / "frozen_slots_extract.py"),
                               "--run", str(run), "--model", model, "--batch-size", str(extract_batch)], artifacts=[artifact], deps=[],
                         env=env, gpus=1, allowed_gpus=[0, 1], priority=20,
                         minimum_seconds=4200 if model.startswith("medgemma") else 2600,
                         success_fields={artifact: {"complete": True}}, max_attempts=2))
    primary = []
    for seed_index, seed in enumerate(seeds):
        seed_jobs = []
        for model in ("image_only", *MODELS):
            conditions = ("image_only",) if model == "image_only" else ("slots", "shuffled_slots")
            for condition in conditions:
                for task in ("segmentation", "sr"):
                    out = run / f"seed_{seed}" / model / f"{task}_{condition}"
                    artifact = str(out / "metrics.json")
                    deps = [] if model == "image_only" else [f"dense_extract_{model}"]
                    if seed_index:
                        deps += primary
                    argv = [str(python), "-u", str(snapshot / "frozen_slots_train.py"),
                            "--data-run", str(run), "--run", str(run), "--out", str(out),
                            "--model", model, "--task", task, "--condition", condition,
                            "--epochs", str(epochs), "--seed", str(seed), "--batch-size", "8",
                            "--microbatch", str(microbatch), "--stop-at", stop_at, "--checkpoint-seconds", "120"]
                    job_id = f"dense_{seed}_{model}_{task}_{condition}"
                    jobs.append(dict(id=job_id, kind="dense_train", model=model, condition=condition,
                                     seed=seed, task=task, optional=bool(seed_index), argv=argv, env=env, artifacts=[artifact], deps=deps,
                                     gpus=1, allowed_gpus=[0, 1], priority=100 if seed_index else 30,
                                     minimum_seconds=1800, max_attempts=2,
                                     success_fields={artifact: {"epochs": epochs, "seed": seed,
                                        "complete_requested_epochs": True,
                                        "train_n": manifest["expansion"]["train_count"]}}))
                    seed_jobs.append(job_id)
        if not seed_index:
            primary = seed_jobs
    protocol = dict(models=list(MODELS), seeds=seeds, epochs=epochs,
                    effective_batch=8, microbatch=microbatch,
                    extraction_batch_size=dict(qwen=qwen_extract_batch, medgemma=medgemma_extract_batch),
                    benchmark_sha256={p.name: digest(p) for p in sorted(run.glob("*benchmark*.json"))},
                    batch_equivalence="4-image HR/LR cosine >0.999998; batches up to64 relative L2 <0.3%; neighbor substitution difference zero",
                    train_n=manifest["expansion"]["train_count"],
                    cohort_sha256=manifest["cohort_sha256"], data_manifest_sha256=digest(run / "data/manifest.json"),
                    models_sha256=digest(run / "models.json"),
                    source_sha256={name: digest(snapshot / name) for name in filenames},
                    slots="four frozen native vision depths, slots 5-8; no language model",
                    no_slots="shared matched image-only decoder with zero slots; not a full-token VLM condition",
                    shuffled="fixed one-to-one different-patient donors within each split",
                    stop_at=stop_at, final_deadline="2026-09-14T08:00:00+08:00")
    plan = dict(protocol=protocol, jobs=jobs,
                report_commands=[[str(python), str(snapshot / "overnight_dense_report.py"), "--run", str(run)]])
    path = run / "dense_plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Existing dense plan differs; preserve immutable experiment configuration")
    atomic(path, plan)
    labels = {m["id"]: m["label"] for m in json.loads((run / "models.json").read_text())}
    exports = []
    for seed in seeds:
        for model in ("image_only", *MODELS):
            for condition in (("image_only",) if model == "image_only" else ("slots", "shuffled_slots")):
                out = run / f"seed_{seed}" / model
                method = "Shared image-only decoder" if model == "image_only" else labels[model] + " / " + condition
                exports.append(dict(table=2, method=method,
                    protocol=f"D1: {protocol['train_n']} train; {epochs} epochs; seed {seed}; frozen vision slots 5–8",
                    artifacts=[dict(path=str(out / f"segmentation_{condition}/metrics.json"),
                                    metrics={"Dice pseudo": "metrics.test.dice", "Dice human": "metrics.human_test.dice"}),
                               dict(path=str(out / f"sr_{condition}/metrics.json"),
                                    metrics={"PSNR x4": "metrics.test.psnr", "SSIM x4": "metrics.test.ssim"})]))
    atomic(run / "table_export.json", exports)
    report(run)
    print(json.dumps(dict(plan=str(path), jobs=len(jobs), train_n=protocol["train_n"],
                          source=str(snapshot)), indent=2), flush=True)
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--seeds", default="20260913,20260914")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--microbatch", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--snapshot-name", default="source_dense_v3")
    parser.add_argument("--qwen-extract-batch", type=int, default=64)
    parser.add_argument("--medgemma-extract-batch", type=int, default=1)
    parser.add_argument("--stop-at", default="2026-09-14T07:45:00+08:00")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.epochs < 1 or len(seeds) != len(set(seeds)) or not seeds:
        parser.error("Positive epochs and unique nonempty seeds are required")
    make_plan(args.run, args.source, args.python, seeds, args.epochs, args.stop_at, args.microbatch, args.snapshot_name,
              args.qwen_extract_batch, args.medgemma_extract_batch)
