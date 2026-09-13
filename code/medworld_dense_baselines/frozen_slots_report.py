"""Run-scoped frozen-slot metrics, image-only deltas, and paired patient CIs."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import math
from pathlib import Path
import time

from frozen_slots_queue import DEFAULT_RUN, MODELS, atomic, digest, read_json, signature, valid_completion

METRICS = {
    "Dice pseudo": ("segmentation", "test", "dice"),
    "Dice human": ("segmentation", "human_test", "dice"),
    "PSNR x4": ("sr", "test", "psnr"),
    "SSIM x4": ("sr", "test", "ssim"),
}


def write_text(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(value)
    tmp.replace(path)


def sample_rows(path):
    try:
        with Path(path).open() as f:
            return [json.loads(line) for line in f if line.strip()]
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def comparison_contract(baseline_dir, candidate_dir, baseline, candidate, settings, cohort):
    """Reject plausible deltas when cohorts, training or initialization differ."""
    problems = []
    if baseline is None or candidate is None:
        return {"status": "pending", "reasons": []}
    expected = dict(epochs=settings.get("epochs", 20), seed=settings.get("seed", 20260913),
                    train_n=cohort.get("coverage", {}).get("images", {}).get("train"))
    for key in ["epochs", "seed", "train_n", "cohort_sha256", "task"]:
        if baseline.get(key) is None or baseline.get(key) != candidate.get(key):
            problems.append(f"metrics.{key} mismatch or absent")
        if expected.get(key) is not None and candidate.get(key) != expected[key]:
            problems.append(f"metrics.{key} differs from planned {expected[key]}")
    left = read_json(baseline_dir / "contract.json", {})
    right = read_json(candidate_dir / "contract.json", {})
    fields = ["architecture", "slots_shape", "slots_trained", "slot_norm", "image_only", "data_run",
              "seed", "epochs", "batch_size", "microbatch", "learning_rate", "learning_rate_schedule", "loss", "weight_decay", "optimizer",
              "device_type", "amp", "train_n", "split_indices_sha256", "cohort_sha256", "source_sha256",
              "reused_loss_sha256", "pseudo_path", "manifest_sha256", "image_sha256", "lr_image_sha256", "pseudo_sha256", "human_mask_sha256"]
    for key in fields:
        if key not in left or key not in right or left[key] != right[key]:
            problems.append(f"contract.{key} mismatch or absent")
    for label, directory, result in [("baseline", baseline_dir, baseline), ("candidate", candidate_dir, candidate)]:
        path = directory / "contract.json"
        if not path.exists() or digest(path) != result.get("contract_sha256"):
            problems.append(f"{label} contract SHA256 differs from completed metrics")
    left_init = read_json(baseline_dir / "initialization.json", {})
    right_init = read_json(candidate_dir / "initialization.json", {})
    for key in ["state_sha256", "parameters", "trainable_parameters", "slot_parameters", "microbatch", "effective_batch"]:
        if key not in left_init or key not in right_init or left_init[key] != right_init[key]:
            problems.append(f"initialization.{key} mismatch or absent")
    left_epochs = sample_rows(baseline_dir / "epochs.jsonl") or []
    right_epochs = sample_rows(candidate_dir / "epochs.jsonl") or []
    epochs = expected["epochs"]
    if len(left_epochs) != epochs or len(right_epochs) != epochs:
        problems.append("epoch history length differs from planned epochs")
    elif [e.get("epoch") for e in left_epochs] != list(range(1, epochs + 1)) or [e.get("epoch") for e in right_epochs] != list(range(1, epochs + 1)):
        problems.append("epoch numbering is incomplete or duplicated")
    else:
        for left_epoch, right_epoch in zip(left_epochs, right_epochs):
            if not left_epoch.get("order_sha256") or left_epoch["order_sha256"] != right_epoch.get("order_sha256"):
                problems.append(f"epoch {left_epoch.get('epoch')} sample order mismatch")
    return {"status": "matched" if not problems else "mismatch", "reasons": problems}


def paired_patient_bootstrap(baseline_path, candidate_path, metric, samples=2000, seed=20260913):
    """Resample patient clusters; preserve each selected patient's image count.

    The observed statistic is the image-weighted mean paired metric delta.
    All IDs and subject IDs must match; incomplete overlaps are never reported.
    """
    baseline, candidate = sample_rows(baseline_path), sample_rows(candidate_path)
    if baseline is None or candidate is None:
        return {"status": "missing_per_sample_records"}
    if not baseline or not candidate:
        return {"status": "empty_per_sample_records"}
    if any("id" not in row or "subject_id" not in row for row in baseline + candidate):
        return {"status": "missing_pair_or_patient_identifiers"}
    b, c = {str(row["id"]): row for row in baseline}, {str(row["id"]): row for row in candidate}
    if len(b) != len(baseline) or len(c) != len(candidate):
        return {"status": "duplicate_sample_ids"}
    if b.keys() != c.keys():
        return {"status": "cohort_mismatch", "baseline_n": len(b), "candidate_n": len(c),
                "matched_n": len(b.keys() & c.keys())}
    grouped = {}
    for sample_id in sorted(b):
        left, right = b[sample_id], c[sample_id]
        if str(left["subject_id"]) != str(right["subject_id"]) or left["subject_id"] is None:
            return {"status": "patient_identity_mismatch"}
        values = [left.get(metric), right.get(metric)]
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            return {"status": "missing_or_nonfinite_metric", "metric": metric}
        grouped.setdefault(str(left["subject_id"]), []).append(values[1] - values[0])
    if len(grouped) < 2:
        return {"status": "fewer_than_two_patients"}
    import numpy as np
    sums = np.array([sum(grouped[key]) for key in sorted(grouped)], dtype=np.float64)
    counts = np.array([len(grouped[key]) for key in sorted(grouped)], dtype=np.int64)
    rng = np.random.default_rng(seed)
    estimates = []
    for offset in range(0, samples, 128):
        indices = rng.integers(0, len(sums), size=(min(128, samples - offset), len(sums)))
        estimates.extend((sums[indices].sum(axis=1) / counts[indices].sum(axis=1)).tolist())
    low, high = np.quantile(estimates, [0.025, 0.975])
    return dict(status="complete", metric=metric, delta=float(sums.sum() / counts.sum()),
                ci95=[float(low), float(high)], n=len(b), patients=len(grouped),
                bootstrap_samples=samples, seed=seed, unit="patient clusters", aggregation="image-weighted mean")


def fmt(value, digits=4, signed=False):
    if value is None:
        return "pending"
    return format(value, f"{'+' if signed else ''}.{digits}f")


def table(columns, rows):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "pending")).replace("|", "\\|").replace("\n", " ")
                                       for key in columns) + " |")
    return lines


def report(run, bootstrap_samples=2000):
    run = Path(run).resolve()
    out = run / "preview"
    out.mkdir(parents=True, exist_ok=True)
    config = read_json(run / "queue_config.json", {})
    protocol = read_json(run / "protocol.json", {})
    queue = read_json(run / "queue.json", [])
    status = read_json(run / "status.json", {})
    cohort = read_json(run / "data/manifest.json", {})
    specs = read_json(run / "models.json", [])
    labels = {model["id"]: model.get("label", model["id"]) for model in specs}
    settings = config.get("settings", {})
    models = settings.get("models", MODELS)
    controls = settings.get("controls", ["qwen4b", "medgemma4b"])
    run_id = config.get("run_id")
    jobs = {job["id"]: job for job in queue}
    warnings, bootstrap = [], []

    def completed_metrics(model, task, condition):
        job = jobs.get(f"{model}_{task}_{condition}")
        if job is None or job.get("status") != "complete":
            return None, job.get("status", "pending") if job else "pending"
        if not valid_completion(run, job, run_id):
            warnings.append(f"Completion provenance mismatch: {job['id']}")
            return None, "provenance_mismatch"
        result = read_json(job["artifact"], {})
        return result, "complete"

    rows, comparisons = [], [("image_only", "image_only")]
    comparisons.extend((model, "slots") for model in models)
    comparisons.extend((model, "shuffled_slots") for model in controls)
    baseline_results = {task: completed_metrics("image_only", task, "image_only")[0]
                        for task in ["segmentation", "sr"]}
    for model, condition in comparisons:
        row = dict(model=model, model_label="Image-only baseline" if model == "image_only" else labels.get(model, model),
                   condition=condition, epochs=settings.get("epochs", 20))
        results = {}
        for task in ["segmentation", "sr"]:
            results[task], row[f"{task}_status"] = completed_metrics(model, task, condition)
            check = comparison_contract(run / "image_only" / f"{task}_image_only",
                                        run / model / f"{task}_{condition}", baseline_results[task],
                                        results[task], settings, cohort)
            row[f"comparison_{task}"] = check
            if check["status"] == "mismatch":
                warnings.append(f"{model}/{condition}/{task}: DELTAS WITHHELD: " + "; ".join(check["reasons"]))
        for title, (task, split, metric) in METRICS.items():
            result = results[task] or {}
            value = result.get("metrics", {}).get(split, {}).get(metric)
            base = (baseline_results[task] or {}).get("metrics", {}).get(split, {}).get(metric)
            row[title] = value
            matched = row[f"comparison_{task}"]["status"] == "matched"
            row[f"delta_{title}"] = None if value is None or base is None or not matched else value - base
            row[f"ci95_delta_{title}"] = None
            if model != "image_only" and value is not None and base is not None and matched:
                pair = paired_patient_bootstrap(run / "image_only" / f"{task}_image_only" / f"{split}_per_sample.jsonl",
                                                run / model / f"{task}_{condition}" / f"{split}_per_sample.jsonl",
                                                metric, bootstrap_samples, settings.get("seed", 20260913))
                pair.update(model=model, condition=condition, task=task, split=split, title=title)
                if pair["status"] == "complete":
                    row[f"ci95_delta_{title}"] = pair["ci95"]
                    if abs(pair["delta"] - row[f"delta_{title}"]) > 1e-5:
                        pair["aggregate_difference"] = row[f"delta_{title}"] - pair["delta"]
                        warnings.append(f"{model}/{condition}/{title}: aggregate differs from per-image mean; CI estimates per-image delta")
                else:
                    warnings.append(f"{model}/{condition}/{title}: paired CI unavailable ({pair['status']})")
                bootstrap.append(pair)
        rows.append(row)
    towers = []
    for model in models:
        contract = read_json(run / model / "slot_contract.json", {})
        metadata = read_json(run / model / "slots_model.json", {})
        towers.append(dict(model=model, model_label=labels.get(model, model), contract=contract,
                           metadata=metadata,
                           extraction_status=jobs.get(f"extract_{model}", {}).get("status", "pending")))
    duplicate_groups = {}
    for tower in towers:
        metadata = tower["metadata"]
        weights = metadata.get("vision_backbone_weights_sha256")
        if weights:
            processor = dict(metadata.get("processor") or {})
            processor.pop("_name_or_path", None)
            duplicate_groups.setdefault(weights, []).append((tower["model"], signature(processor)))
    duplicate_groups = [dict(models=[item[0] for item in group], vision_backbone_weights_sha256=key,
                             processor_sha256={item[0]: item[1] for item in group},
                             processor_settings_identical=len({item[1] for item in group}) == 1)
                        for key, group in duplicate_groups.items() if len(group) > 1]
    output = dict(updated=dt.datetime.now(dt.timezone.utc).isoformat(), run=str(run), run_id=run_id,
                  protocol=protocol, queue_settings=settings, cohort=cohort, status=status,
                  rows=rows, paired_patient_bootstrap=bootstrap, towers=towers,
                  duplicate_tower_groups=duplicate_groups, warnings=warnings,
                  interpretation="Frozen native vision-tower features; checkpoint family sizes do not constitute a language-model scaling experiment.")
    atomic(out / "frozen_slots_results.json", output)
    atomic(out / "paired_patient_bootstrap.json", bootstrap)
    columns = ["model", "model_label", "condition", "epochs", "segmentation_status", "sr_status", "comparison_segmentation", "comparison_sr"]
    for title in METRICS:
        columns.extend([title, f"delta_{title}", f"ci95_delta_{title}"])
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "pending" if row.get(key) is None else json.dumps(row[key]) if isinstance(row[key], (list, dict))
                         else row[key] for key in columns})
    write_text(out / "frozen_slots_results.csv", stream.getvalue())
    display_columns = ["Model", "Condition", "Seg / SR", "Dice pseudo", "Δ Dice pseudo", "Dice human", "Δ Dice human",
                       "PSNR ×4", "Δ PSNR", "SSIM ×4", "Δ SSIM"]
    display = []
    for row in rows:
        display.append({"Model": row["model_label"], "Condition": row["condition"],
                        "Seg / SR": f'{row["segmentation_status"]} / {row["sr_status"]}',
                        "Dice pseudo": fmt(row["Dice pseudo"]), "Δ Dice pseudo": fmt(row["delta_Dice pseudo"], signed=True),
                        "Dice human": fmt(row["Dice human"]), "Δ Dice human": fmt(row["delta_Dice human"], signed=True),
                        "PSNR ×4": fmt(row["PSNR x4"], 3), "Δ PSNR": fmt(row["delta_PSNR x4"], 3, True),
                        "SSIM ×4": fmt(row["SSIM x4"]), "Δ SSIM": fmt(row["delta_SSIM x4"], signed=True)})
    local = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S Asia/Shanghai")
    lines = ["# Frozen slots 5–8: segmentation and ×4 super-resolution", "",
             f"Updated: {local}. Target: {settings.get('delivery_target', '2026-09-13T20:00:00+08:00')}.", "",
             "All encoder weights and four extracted depth slots (5–8) stay frozen. Only the image decoder and its slot-conditioning layers are trained. "
             f"Comparisons use the same image decoder initialization, patient splits, {settings.get('epochs', 20)} epochs, and seed {settings.get('seed', 20260913)}. "
             "The shared image-only baseline passes zero slot conditions through the same initialized architecture.", "",
             "Slots are native vision-tower intermediate states. Model names identify checkpoint provenance; this experiment does not test language-model parameter scaling. "
             "These four depth groups are an initial fixed representation for method slots 5–8; they have not been trained as semantic slots. "
             "Classification and diagnosis are outside this dense-task run.", "",
             "Segmentation uses HR-derived slots. Super-resolution uses only the shared rounded ×4 LR input for both the image branch and slot extraction; the HR target is excluded from model inputs. "
             "Pseudo Dice measures CXAS three-organ teacher agreement. Human Dice is external Montgomery two-lung evaluation. "
             "All deltas below are relative to the shared image-only baseline; positive values are better. `pending` means unavailable, never zero.", "",
             "Deltas and paired intervals are withheld unless epochs, seed, training count, cohort and split hashes, decoder parameter counts and initial state hash, training hyperparameters, "
             "and every epoch's sample order match the baseline. Contract hashes must agree with each completed metrics artifact.", "",
             "Cohort: `" + json.dumps(cohort.get("coverage", {}).get("images", {}), ensure_ascii=False) + "`.", ""]
    lines.extend(table(display_columns, [item for item in display if item["Condition"] != "shuffled_slots"]))
    if controls:
        lines.extend(["", "## Supplementary shuffled-slot controls", "",
                      "These runs use the same frozen slot distribution with sample assignments shuffled within each split. "
                      "They start after the main comparison completes and help distinguish image-specific slot information from generic conditioning effects.", ""])
        lines.extend(table(display_columns, [item for item in display if item["Condition"] == "shuffled_slots"]))
    bicubic = read_json(run / "bicubic_metrics.json", {})
    if bicubic:
        lines.extend(["", f"Existing matched-data bicubic reference (no training): PSNR {fmt(bicubic.get('psnr'), 3)}, SSIM {fmt(bicubic.get('ssim'))}. "
                      "It is a reference only; learned comparisons use the new image-only baseline above."])
    lines.extend(["", "## Paired differences and uncertainty", "",
                  "95% percentile intervals resample patients with replacement, keeping each selected patient's images together. "
                  f"Each interval uses {bootstrap_samples} bootstrap draws and estimates the image-weighted mean paired difference. "
                  "Intervals require identical sample and patient IDs and complete measurements in both runs. "
                  "These are exploratory single-seed comparisons without multiplicity correction.", ""])
    ci_rows = []
    for pair in bootstrap:
        ci_rows.append({"Model / condition": labels.get(pair["model"], pair["model"]) + " / " + pair["condition"],
                        "Metric": pair["title"], "Paired Δ": fmt(pair.get("delta"), signed=True),
                        "95% CI": "pending" if "ci95" not in pair else "[" + ", ".join(fmt(x, signed=True) for x in pair["ci95"]) + "]",
                        "Images / patients": f"{pair.get('n', 'pending')} / {pair.get('patients', 'pending')}",
                        "State": pair["status"]})
    lines.extend(table(["Model / condition", "Metric", "Paired Δ", "95% CI", "Images / patients", "State"], ci_rows)
                 if ci_rows else ["Pending completed matched baseline and slot measurements."])
    lines.extend(["", "## Encoder provenance", "",
                  "Full extraction contracts, including selected hidden layers, projection rule and native tower details, are retained in `preview/frozen_slots_results.json` and each model's `slot_contract.json`.", ""])
    tower_rows = []
    for item in towers:
        metadata = item["metadata"]
        tower_rows.append({"Model": item["model_label"], "Extraction": item["extraction_status"],
                           "Tower parameters": metadata.get("vision_parameters", "pending"),
                           "Layers": metadata.get("block_indices_one_based", "pending")})
    lines.extend(table(["Model", "Extraction", "Tower parameters", "Layers"], tower_rows))
    if duplicate_groups:
        lines.extend(["", "Identical native visual backbone weights: " + "; ".join(
            ", ".join(labels.get(model, model) for model in group["models"]) +
            (" (processor settings also identical)" if group["processor_settings_identical"] else " (processor metadata differ; inspect contracts)")
            for group in duplicate_groups) +
            ". These rows share the same frozen backbone and serve as a reproducibility check; they are not independent evidence for a model-size effect."])
    lines.extend(["", "## Queue", "", f"Policy: at most {status.get('max_gpus', 4)} GPUs, cards without existing compute processes only. "
                  "Coordinator restart adopts its own live workers; no foreign jobs are stopped.", "",
                  "Counts: `" + json.dumps(status.get("counts", {state: sum(j.get('status') == state for j in queue)
                                                                                 for state in ['queued', 'running', 'complete', 'failed', 'blocked']})) + "`.", ""])
    queue_rows = []
    for job in queue:
        progress = None
        if job["id"].startswith("extract_"):
            progress = read_json(run / job["id"].removeprefix("extract_") / "slots_progress.json")
        else:
            progress = read_json(Path(job["artifact"]).parent / "progress.json")
        queue_rows.append({"Job": job["id"], "State": job["status"], "GPU": job.get("gpu", "—"),
                           "Attempt": job.get("attempts", 0),
                           "Progress / reason": job.get("reason") or (json.dumps(progress, ensure_ascii=False)[:240] if progress else "—")})
    lines.extend(table(["Job", "State", "GPU", "Attempt", "Progress / reason"], queue_rows))
    if warnings:
        lines.extend(["", "## Measurement notes", ""] + ["- " + warning for warning in warnings])
    lines.extend(["", "Outputs: `preview/frozen_slots_results.csv`, `preview/frozen_slots_results.json`, "
                  "`preview/paired_patient_bootstrap.json`, per-task metrics/checkpoints/logs. "
                  "Only artifacts with matching run-specific worker completion records enter the table.", ""])
    markdown = "\n".join(lines)
    write_text(out / "frozen_slots_results.md", markdown)
    write_text(run / "REPORT.md", markdown)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    report(args.run, args.bootstrap_samples)
