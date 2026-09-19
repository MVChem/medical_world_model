"""One checkpoint for current four-task evaluation and bidirectional prediction.

Exports reports and references for the existing clinical scorers. This runner
does not substitute token overlap for clinical report metrics.
"""
import argparse
from pathlib import Path

from ..gpu import acquire_gpu


from ..downstream_tasks.classification import classification_metrics
from ..downstream_tasks.report import report_diagnostics
from ..downstream_tasks.segmentation import segmentation_metrics
from ..downstream_tasks.super_resolution import super_resolution_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--task", choices=("all", "current", "temporal", "classification", "report", "segmentation", "sr"), default="all")
    parser.add_argument("--split", choices=("validate", "test", "human_test"), default="test")
    parser.add_argument("--limit", type=int, help="Optional smoke subset; omit for all rows")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--gpu", default="auto")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        parser.error("Choose a new evaluation output folder")
    lock, device = acquire_gpu(args.gpu)
    import json
    import numpy as np
    import torch
    from ..datasets import TASKS, UnifiedData
    from ..datasets.protocol import _sha256
    from ..downstream_tasks.registry import FINDINGS
    from ..model import TEMPORAL_COLUMNS
    from ..runtime import atomic_json, load_model
    model, saved = load_model(args.checkpoint, device)
    data = UnifiedData(model.cfg)
    if data.fingerprint != saved["data_fingerprint"]:
        raise ValueError("Evaluation data differs from the checkpoint's audited protocol")
    tasks = (*TASKS, "temporal") if args.task == "all" else TASKS if args.task == "current" else (args.task,)
    if args.split == "human_test" and tasks != ("segmentation",):
        parser.error("human_test is available only with --task segmentation")
    out.mkdir(parents=True, exist_ok=True)
    summary = {"checkpoint_sha256": _sha256(Path(args.checkpoint)), "split": args.split,
               "limit": args.limit, "data_fingerprint": data.fingerprint, "tasks": {}}
    with torch.no_grad():
        for task in tasks:
            rows = data.rows(task, args.split)
            count = min(len(rows), args.limit) if args.limit else len(rows)
            records, labels, probabilities, dense_scores = [], [], [], []
            for index in range(count):
                batch = data.batch(task, args.split, [index], source_only=task == "temporal")
                if task == "temporal":
                    predicted = model.predict_state(**batch)
                    text = model.decode_state(predicted, args.max_new_tokens)[0]
                    probs = model.classification(predicted)[:, TEMPORAL_COLUMNS].sigmoid()[0].cpu().tolist()
                    # References are accessed only after prediction, with no target encoding.
                    row = rows[index]
                    reference = data.temporal.lookup[row["target"]]
                    records.append({"id": row["id"], "patient": str(row["patient"]),
                                    "direction": row["direction"], "delta_hours": row["delta_hours"],
                                    "prediction": text, "reference": reference["report"],
                                    "probabilities": probs, "labels": reference["labels"]})
                    labels.append(reference["labels"])
                    probabilities.append(probs)
                elif task in ("classification", "report"):
                    state = model.encode(batch["images"])
                    if task == "report":
                        records.append({"id": batch["ids"][0], "patient": batch["subject_ids"][0],
                                        "prediction": model.decode_state(state, args.max_new_tokens)[0],
                                        "reference": batch["report_targets"][0]})
                    else:
                        probs = model.classification(state).sigmoid()[0].cpu().tolist()
                        target = batch["labels"][0].tolist()
                        labels.append(target)
                        probabilities.append(probs)
                        records.append({"id": batch["ids"][0], "patient": batch["subject_ids"][0],
                                        "labels": target, "probabilities": probs})
                else:
                    state = model.encode(batch["images"], spatial=True)
                    prediction = getattr(model, task)(batch["pixels"].to(device), state).cpu()
                    target, mask = batch["targets"], batch["mask"]
                    score = (segmentation_metrics(prediction, target, mask) if task == "segmentation"
                             else super_resolution_metrics(prediction, target, mask))
                    dense_scores.append(score)
                    records.append({"id": batch["ids"][0], "patient": batch["subject_ids"][0], **score})
                if (index + 1) % 25 == 0:
                    print(f"{task}: {index + 1}/{count}", flush=True)
            metrics = {"n": count}
            if labels:
                names = [FINDINGS[i] for i in TEMPORAL_COLUMNS] if task == "temporal" else FINDINGS
                metrics.update(classification_metrics(labels, probabilities, names))
            if task in ("report", "temporal"):
                metrics.update(report_diagnostics(records))
            if task == "temporal":
                metrics["by_direction"] = {}
                for direction in ("forward", "backward"):
                    selected = [r for r in records if r["direction"] == direction]
                    if selected:
                        metrics["by_direction"][direction] = {**report_diagnostics(selected),
                            **classification_metrics([r["labels"] for r in selected],
                                                     [r["probabilities"] for r in selected], names)}
            if dense_scores:
                if task == "segmentation":
                    metrics.update(mean_dice=float(np.mean([r["mean_dice"] for r in dense_scores])),
                                   dice_per_organ=np.mean([r["dice_per_organ"] for r in dense_scores], axis=0).tolist(),
                                   target_kind="human two-lung masks" if args.split == "human_test" else "CXAS pseudo three-organ masks")
                else:
                    metrics.update({key: float(np.mean([r[key] for r in dense_scores])) for key in ("psnr", "ssim")})
            with (out / f"{task}.jsonl").open("w") as handle:
                for row in records:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            summary["tasks"][task] = metrics
            atomic_json(out / "summary.json", summary)
            print(f"Completed {task}: {count} examples", flush=True)
    if lock is not None:
        lock.close()


if __name__ == "__main__":
    main()
