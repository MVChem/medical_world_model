"""Evaluate the current Table 1 with source-only inputs and fixed references."""
import argparse
import json
import time
from pathlib import Path

from .protocol import FUTURE_TASKS, metric_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--task", choices=FUTURE_TASKS, required=True)
    parser.add_argument("--split", choices=("validate", "test"), default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gpu", default="auto")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Choose a new evaluation output folder")
    from ..gpu import acquire_gpu
    lock, device = acquire_gpu(args.gpu)
    try:
        import torch
        from ..runtime import load_model, atomic_json
        from ..datasets import UnifiedData
        from ..datasets.protocol import _sha256
        from .future_common import reference_for_row, scoring_protocol, load_radgraph
        from .future_metrics import score_future_task, reference_fingerprint, DIRECTIONS
        model, saved = load_model(args.checkpoint, device)
        if not model.cfg["future_enabled"]:
            raise ValueError("Table 1 requires trained future-task heads")
        data = UnifiedData(model.cfg)
        if data.fingerprint != saved["data_fingerprint"]:
            raise ValueError("Future-task data differ from the training protocol")
        rows = data.rows(args.task, args.split)
        count = len(rows) if args.limit is None else min(len(rows), args.limit)
        if not count:
            raise ValueError(f"Empty {args.task}/{args.split} cohort")
        args.out.mkdir(parents=True)
        references, predictions = [], {}
        with (args.out / f"{args.task}.jsonl").open("w", buffering=1) as journal, torch.inference_mode():
            for index in range(count):
                # Prediction is completed before target reports or labels are read.
                batch = data.batch(args.task, args.split, [index], source_only=True)
                value = model.predict(args.task, batch)[0]
                if torch.is_tensor(value):
                    value = value.item()
                if args.task == "progression":
                    value = DIRECTIONS[int(value)]
                reference = reference_for_row(args.task, rows[index], data.future)
                references.append(reference)
                predictions[reference["id"]] = value
                journal.write(json.dumps({**reference, "prediction": value}, ensure_ascii=False) + "\n")
                atomic_json(args.out / "status.json", {"state": "predicting", "task": args.task,
                            "completed": index + 1, "total": count, "heartbeat_unix": time.time()})
                if (index + 1) % 25 == 0:
                    print(f"{args.task}: {index + 1}/{count}", flush=True)
        scorer = load_radgraph(model.cfg) if args.task == "future_report" else None
        protocol = scoring_protocol(args.task, references, model.cfg, scorer)
        metrics = score_future_task(args.task, references, predictions, protocol=protocol, radgraph_scorer=scorer)
        atomic_json(args.out / f"{args.task}_protocol.json", protocol)
        summary = {"metric_protocol": metric_protocol("table1"),
                   "checkpoint_sha256": _sha256(Path(args.checkpoint)),
                   "data_fingerprint": data.fingerprint,
                   "future_data_fingerprint": data.future.fingerprint,
                   "slot_conditioning": model.cfg["slot_conditioning"],
                   "split": args.split, "limit": args.limit,
                   "partial": args.limit is not None,
                   "references": {args.task: {"n": count, "references_sha256": reference_fingerprint(references)}},
                   "predictions_sha256": {args.task: _sha256(args.out / f"{args.task}.jsonl")},
                   "tasks": {args.task: metrics}}
        atomic_json(args.out / "summary.json", summary)
        atomic_json(args.out / "status.json", {"state": "complete" if metrics["complete"] or args.limit is not None else "failed",
                    "partial": args.limit is not None,
                    "completed": count, "total": count, "heartbeat_unix": time.time()})
        if not metrics["complete"] and args.limit is None:
            raise ValueError(f"Incomplete {args.task} evaluation; inspect summary.json")
    finally:
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    main()
