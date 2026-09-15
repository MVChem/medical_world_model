"""Report expanded dense results without assuming a particular queue format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import atomic, digest

MODELS = ("qwen08b", "qwen4b", "qwen9b", "qwen27b_fp8", "medgemma4b", "medgemma27b")


def report(run):
    run = Path(run).resolve()
    plan = json.loads((run / "dense_plan.json").read_text())
    manifest = json.loads((run / "data/manifest.json").read_text())
    labels = {m["id"]: m["label"] for m in json.loads((run / "models.json").read_text())}
    rows = []
    for job in plan["jobs"]:
        if job.get("kind") != "dense_train":
            continue
        out = Path(job["artifacts"][0]).parent
        metrics_file = out / "metrics.json"
        partial = False
        if not metrics_file.exists() and (out / "partial_metrics.json").exists():
            metrics_file, partial = out / "partial_metrics.json", True
        result = None
        if metrics_file.exists():
            candidate = json.loads(metrics_file.read_text())
            if candidate["cohort_sha256"] != manifest["cohort_sha256"]:
                raise ValueError(f"Result cohort mismatch: {metrics_file}")
            if candidate["contract_sha256"] != digest(out / "contract.json"):
                raise ValueError(f"Result contract mismatch: {metrics_file}")
            if not partial and (candidate["epochs"] != plan["protocol"]["epochs"]
                                or not candidate.get("complete_requested_epochs")):
                raise ValueError(f"Incorrect full-completion marker: {metrics_file}")
            result = candidate
        progress_path = out / "progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        rows.append(dict(job_id=job["id"], model=job["model"], condition=job["condition"],
                         seed=job["seed"], task=job["task"], status="partial" if partial else
                         "complete" if result else progress.get("status", "pending"),
                         artifact=str(metrics_file) if result else None,
                         epochs=result["epochs"] if result else progress.get("epoch", 0),
                         metrics=result["metrics"] if result else None))
    text = ["# Expanded frozen vision-slot dense experiments", "",
            f"Training: {manifest['expansion']['train_count']:,}; validation: 249; test: 447; external human lungs: 138.", "",
            "Frozen slots are four native vision-block summaries (slots 5–8); only the matched decoder is trained. "
            "Image-only uses the same decoder with zero slot inputs and is trained once per task/seed; "
            "its result is shared across all six model comparisons. This is not the eight-slot joint-optimization experiment.", "",
            "Shuffled slots use a different patient within the same split. Super-resolution inputs, including "
            "the native encoder, originate only from the prepared LR image. Segmentation pseudo Dice measures "
            "CXAS teacher agreement; human Dice evaluates Montgomery lungs.", "",
            "Incomplete deadline results are marked partial; missing results remain pending. "
            "All primary-seed training jobs must finish before secondary-seed jobs become eligible.", ""]
    for seed in plan["protocol"]["seeds"]:
        text += [f"## Seed {seed}", "", "| Model / condition | Pseudo Dice | Human Dice | SR PSNR | SR SSIM | Seg / SR status |",
                 "|---|---:|---:|---:|---:|---|"]
        for model in ("image_only", *MODELS):
            conditions = ("image_only",) if model == "image_only" else ("slots", "shuffled_slots")
            for condition in conditions:
                selected = {r["task"]: r for r in rows if (r["seed"], r["model"], r["condition"]) == (seed, model, condition)}
                def metric(task, split, key):
                    record = selected.get(task)
                    value = record and record.get("metrics")
                    return f"{value[split][key]:.4f}" if value else "—"
                title = "Shared image-only decoder" if model == "image_only" else labels[model] + " / " + condition
                statuses = " / ".join(selected.get(t, {}).get("status", "pending") for t in ("segmentation", "sr"))
                text.append(f"| {title} | {metric('segmentation', 'test', 'dice')} | "
                            f"{metric('segmentation', 'human_test', 'dice')} | {metric('sr', 'test', 'psnr')} | "
                            f"{metric('sr', 'test', 'ssim')} | {statuses} |")
        text.append("")
    preview = run / "preview"
    preview.mkdir(exist_ok=True)
    target = preview / "expanded_dense_tables.md"
    temporary = target.with_suffix(".tmp")
    temporary.write_text("\n".join(text) + "\n")
    temporary.replace(target)
    atomic(preview / "expanded_dense_results.json", dict(protocol=plan["protocol"], rows=rows))
    print(str(target), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    report(parser.parse_args().run)
