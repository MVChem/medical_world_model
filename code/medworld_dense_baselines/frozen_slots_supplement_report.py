"""Combine verified historical results with a separate shuffled-slot queue."""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import importlib
import io
import json
import math
from pathlib import Path
import sys


def report(run, source, bootstrap_samples=2000):
    run, source = Path(run).resolve(), Path(source).resolve()
    sys.path.insert(0, str(source))
    queue_module = importlib.import_module("frozen_slots_queue")
    original = importlib.import_module("frozen_slots_report")
    for module in (queue_module, original):
        if Path(module.__file__).resolve().parent != source:
            raise RuntimeError(f"Unexpected source module: {module.__file__}")
    read, digest, atomic = queue_module.read_json, queue_module.digest, queue_module.atomic
    protocol = read(run / "protocol.json", {})
    parent = Path(protocol["parent_run"]).resolve()
    if parent == run:
        raise ValueError("Supplement and parent must be different run directories")
    reference = read(run / "parent_reference.json", {})
    if reference.get("parent_run", str(parent)) != str(parent):
        raise RuntimeError("Historical reference points to a different parent run")
    for name, expected in reference.get("artifacts", reference.get("files_sha256", {})).items():
        path = Path(name)
        path = path if path.is_absolute() else parent / path
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"Historical reference changed: {path}")
    parent_config = read(parent / "queue_config.json", {})
    settings = parent_config["settings"]
    cohort = read(parent / "data/manifest.json", {})
    parent_jobs = {job["id"]: job for job in read(parent / "queue.json", [])}
    historical = read(parent / "preview/frozen_slots_results.json", {})
    if historical.get("run_id") != parent_config["run_id"]:
        raise RuntimeError("Historical report run_id does not match parent queue")
    config = read(run / "queue_config.json", {})
    jobs = read(run / "queue.json", [])
    job_map = {job["id"]: job for job in jobs}
    status = read(run / "status.json", {})
    warnings, pairs = [], []
    metrics = original.METRICS

    def verified_result(root, job, run_id, required=False):
        state = job.get("status", "pending") if job else "pending"
        if state != "complete":
            if required:
                raise RuntimeError(f"Historical job is not complete: {job}")
            return None, state
        if not queue_module.valid_completion(root, job, run_id):
            if required:
                raise RuntimeError(f"Historical completion mismatch: {job['id']}")
            warnings.append(f"Completion provenance mismatch: {job['id']}")
            return None, "provenance_mismatch"
        value = read(job["artifact"])
        if not isinstance(value, dict):
            raise RuntimeError(f"Invalid completed metrics: {job['artifact']}")
        return value, "complete"

    def metric_value(result, task, split, metric):
        value = (result or {}).get("metrics", {}).get(split, {}).get(metric)
        if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise RuntimeError(f"Nonfinite {task}/{split}/{metric}")
        return value

    baseline = {}
    for task in ("segmentation", "sr"):
        baseline[task], _ = verified_result(parent, parent_jobs.get(f"image_only_{task}_image_only"),
                                            parent_config["run_id"], required=True)
    expected_old = [("image_only", "image_only")]
    expected_old.extend((model, "slots") for model in settings["models"])
    expected_old.extend((model, "shuffled_slots") for model in settings["controls"])
    old_rows = {(row["model"], row["condition"]): row for row in historical.get("rows", [])}
    if len(old_rows) != len(historical.get("rows", [])) or set(old_rows) != set(expected_old):
        raise RuntimeError("Historical report rows differ from the original experiment plan")
    rows = []
    for model, condition in expected_old:
        row = copy.deepcopy(old_rows[(model, condition)])
        results = {}
        for task in ("segmentation", "sr"):
            results[task], _ = verified_result(parent, parent_jobs.get(f"{model}_{task}_{condition}"),
                                               parent_config["run_id"], required=True)
            check = original.comparison_contract(parent / "image_only" / f"{task}_image_only",
                                                parent / model / f"{task}_{condition}",
                                                baseline[task], results[task], settings, cohort)
            if row.get(f"comparison_{task}") != check or row.get(f"{task}_status") != "complete":
                raise RuntimeError(f"Historical report contract/status differs: {model}/{condition}/{task}")
        for title, (task, split, metric) in metrics.items():
            value = metric_value(results[task], task, split, metric)
            base = metric_value(baseline[task], task, split, metric)
            delta = value - base if value is not None and base is not None and row[f"comparison_{task}"]["status"] == "matched" else None
            if row.get(title) != value or row.get(f"delta_{title}") != delta:
                raise RuntimeError(f"Historical report differs from raw metrics: {model}/{condition}/{title}")
        row.update(result_origin="historical", source_run=str(parent), source_run_id=parent_config["run_id"])
        rows.append(row)
    pairs.extend(dict(pair, result_origin="historical", source_run=str(parent))
                 for pair in historical.get("paired_patient_bootstrap", []))
    warnings.extend(historical.get("warnings", []))
    labels = {model["id"]: model.get("label", model["id"]) for model in read(run / "models.json", [])}
    new_models = []
    for job in jobs:
        suffix = next((f"_{task}_shuffled_slots" for task in ("segmentation", "sr")
                       if job["id"].endswith(f"_{task}_shuffled_slots")), None)
        if suffix is None:
            raise RuntimeError(f"Supplement queue contains a non-shuffled job: {job['id']}")
        model = job["id"][:-len(suffix)]
        if model not in settings["models"] or (model, "shuffled_slots") in old_rows:
            raise RuntimeError(f"Unexpected or duplicate supplement model: {model}")
        if model not in new_models:
            new_models.append(model)
    for model in new_models:
        condition = "shuffled_slots"
        row = dict(model=model, model_label=labels.get(model, model), condition=condition,
                   epochs=settings["epochs"], result_origin="supplement", source_run=str(run),
                   source_run_id=config.get("run_id"))
        results = {}
        for task in ("segmentation", "sr"):
            job = job_map.get(f"{model}_{task}_{condition}")
            if job is None:
                raise RuntimeError(f"Missing supplement task: {model}/{task}")
            expected_artifact = run / model / f"{task}_{condition}" / "metrics.json"
            if Path(job["artifact"]).resolve() != expected_artifact.resolve():
                raise RuntimeError(f"Unexpected supplement artifact: {job['artifact']}")
            results[task], row[f"{task}_status"] = verified_result(run, job, config.get("run_id"))
            check = original.comparison_contract(parent / "image_only" / f"{task}_image_only",
                                                expected_artifact.parent, baseline[task], results[task],
                                                settings, cohort)
            row[f"comparison_{task}"] = check
            if check["status"] == "mismatch":
                warnings.append(f"{model}/{task}: deltas withheld: " + "; ".join(check["reasons"]))
        for title, (task, split, metric) in metrics.items():
            value = metric_value(results[task], task, split, metric)
            base = metric_value(baseline[task], task, split, metric)
            matched = row[f"comparison_{task}"]["status"] == "matched"
            row[title] = value
            row[f"delta_{title}"] = value - base if value is not None and base is not None and matched else None
            row[f"ci95_delta_{title}"] = None
            if value is not None and base is not None and matched:
                pair = original.paired_patient_bootstrap(
                    parent / "image_only" / f"{task}_image_only" / f"{split}_per_sample.jsonl",
                    run / model / f"{task}_{condition}" / f"{split}_per_sample.jsonl",
                    metric, samples=bootstrap_samples, seed=settings["seed"])
                pair.update(model=model, condition=condition, task=task, split=split, title=title,
                            result_origin="supplement", source_run=str(run))
                if pair["status"] == "complete":
                    row[f"ci95_delta_{title}"] = pair["ci95"]
                    if abs(pair["delta"] - row[f"delta_{title}"]) > 1e-5:
                        warnings.append(f"{model}/{title}: aggregate differs from paired per-image delta")
                else:
                    warnings.append(f"{model}/{title}: paired CI unavailable ({pair['status']})")
                pairs.append(pair)
        rows.append(row)
    counts = {state: sum(job.get("status") == state for job in jobs)
              for state in ("queued", "running", "complete", "failed", "blocked")}
    output = dict(updated=dt.datetime.now(dt.timezone.utc).isoformat(), run=str(run),
                  run_id=config.get("run_id"), parent_run=str(parent), parent_run_id=parent_config["run_id"],
                  historical_report_sha256=digest(parent / "preview/frozen_slots_results.json"),
                  protocol=protocol, queue_settings=config.get("settings", {}),
                  comparison_settings=settings, cohort=cohort, status=dict(status, counts=counts),
                  rows=rows, paired_patient_bootstrap=pairs, warnings=warnings,
                  towers=historical.get("towers", []), duplicate_tower_groups=historical.get("duplicate_tower_groups", []),
                  interpretation=historical.get("interpretation"),
                  provenance="Historical rows are checked against parent completion records, raw metrics and comparison contracts. New rows require supplement-specific completion records.")
    preview = run / "preview"
    preview.mkdir(parents=True, exist_ok=True)
    atomic(preview / "frozen_slots_results.json", output)
    atomic(preview / "paired_patient_bootstrap.json", pairs)
    columns = ["model", "model_label", "condition", "result_origin", "source_run", "epochs", "segmentation_status", "sr_status"]
    for title in metrics:
        columns.extend([title, f"delta_{title}", f"ci95_delta_{title}"])
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "pending" if row.get(key) is None else json.dumps(row[key])
                         if isinstance(row[key], (list, dict)) else row[key] for key in columns})
    original.write_text(preview / "frozen_slots_results.csv", stream.getvalue())
    display = []
    for row in rows:
        item = {"Model": row["model_label"], "Condition": row["condition"], "Origin": row["result_origin"],
                "Seg / SR": f"{row['segmentation_status']} / {row['sr_status']}"}
        for title in metrics:
            scale = 100 if title.startswith("Dice") else 1
            digits = 2 if scale == 100 else 4 if title.startswith("PSNR") else 6
            value, delta = row[title], row[f"delta_{title}"]
            item[title] = original.fmt(None if value is None else value * scale, digits)
            item[f"Δ {title}"] = original.fmt(None if delta is None else delta * scale, digits, True)
        display.append(item)
    local = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S Asia/Shanghai")
    lines = ["# Frozen slots: additional shuffled controls", "", f"Updated: {local}.", "",
             f"Parent experiment: `{parent}`. Historical results are reused with their original completion provenance; the new queue contains only {len(jobs)} additional training jobs.", "",
             f"Matched protocol: {settings['epochs']} epochs, seed {settings['seed']}, {cohort.get('coverage', {}).get('images', {}).get('train', 'unknown')} training images, effective batch {settings['batch_size']}, microbatch {settings['microbatch']}. Encoders and cached slots remain frozen; only the decoder is trained.", "",
             "Shuffled slots use another patient's slots within the same split during training and evaluation, with the original fixed donor mapping algorithm. Segmentation uses HR slots; ×4 super-resolution uses LR slots.", "",
             "All Δ values are relative to the historical image-only baseline. Positive is better. Dice is shown as a percentage and Δ Dice in percentage points; PSNR is in dB and SSIM differences are absolute. Pending measurements remain `pending`.", "",
             "Historical rows are verified against original run-specific completion records and raw metrics. New deltas are withheld unless the original report's cohort, source, initialization, training-contract and epoch-order checks match the baseline.", ""]
    display_columns = ["Model", "Condition", "Origin", "Seg / SR"]
    for title in metrics:
        display_columns.extend([title, f"Δ {title}"])
    lines.extend(original.table(display_columns, display))
    lines.extend(["", "## New queue only", "", f"Counts: `{json.dumps(counts)}`.", "",
                  "Historical training and extraction jobs are excluded from these counts.", ""])
    queue_rows = []
    for job in jobs:
        progress = read(Path(job["artifact"]).parent / "progress.json", {})
        queue_rows.append({"Job": job["id"], "State": job["status"], "GPU": job.get("gpu", "—"),
                           "Progress / reason": job.get("reason") or json.dumps(progress, ensure_ascii=False)})
    lines.extend(original.table(["Job", "State", "GPU", "Progress / reason"], queue_rows))
    lines.extend(["", "## Paired uncertainty", "",
                  "95% percentile intervals resample patient clusters and estimate image-weighted mean paired differences. These are exploratory single-seed comparisons without multiplicity correction. Historical intervals are retained from the verified parent report; new intervals use 2,000 draws by default.", ""])
    ci_rows = []
    for pair in pairs:
        title = pair["title"]
        scale = 100 if title.startswith("Dice") else 1
        interval = pair.get("ci95")
        ci_rows.append({"Model / condition": f"{labels.get(pair['model'], pair['model'])} / {pair['condition']}",
                        "Metric": title, "Origin": pair["result_origin"],
                        "Δ": original.fmt(None if pair.get("delta") is None else pair["delta"] * scale, 6, True),
                        "95% CI": "pending" if interval is None else "[" + ", ".join(original.fmt(value * scale, 6, True) for value in interval) + "]",
                        "Images / patients": f"{pair.get('n', 'pending')} / {pair.get('patients', 'pending')}",
                        "State": pair["status"]})
    lines.extend(original.table(["Model / condition", "Metric", "Origin", "Δ", "95% CI", "Images / patients", "State"], ci_rows))
    if output["duplicate_tower_groups"]:
        lines.extend(["", "MedGemma 4B and 27B share identical frozen vision-backbone weights in the parent experiment. Their results are not independent evidence of a model-size effect."])
    if warnings:
        lines.extend(["", "## Measurement notes", ""] + ["- " + warning for warning in warnings])
    lines.extend(["", "Outputs: `preview/frozen_slots_results.json`, `preview/frozen_slots_results.csv`, `preview/paired_patient_bootstrap.json`. The parent experiment is unchanged.", ""])
    markdown = "\n".join(lines)
    original.write_text(run / "REPORT.md", markdown)
    original.write_text(preview / "frozen_slots_results.md", markdown)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("bootstrap-samples must be positive")
    report(args.run, args.source, args.bootstrap_samples)
