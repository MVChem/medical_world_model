from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a partial or completed serious-training run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless the requested 0/8/16/24-hour evidence is complete.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}: {number}")
    return number


def evaluation_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evaluations = [row for row in rows if row.get("event") == "evaluation"]
    if not evaluations:
        return []
    initial_l1 = _finite(
        evaluations[0]["metrics"]["prediction"]["l1"], "initial prediction L1"
    )
    result: list[dict[str, Any]] = []
    for row in evaluations:
        metrics = row["metrics"]
        prediction_l1 = _finite(metrics["prediction"]["l1"], "prediction L1")
        copy_l1 = _finite(metrics["copy_state"]["l1"], "copy L1")
        result.append(
            {
                "phase": row.get("phase", "periodic"),
                "step": int(row["step"]),
                "active_hours": _finite(row["elapsed_seconds"], "elapsed") / 3600.0,
                "examples": int(row["examples"]),
                "prediction_l1": prediction_l1,
                "prediction_cosine": _finite(
                    metrics["prediction"]["cosine_distance"], "prediction cosine"
                ),
                "copy_l1": copy_l1,
                "copy_normalized_gain": 1.0 - prediction_l1 / max(copy_l1, 1.0e-12),
                "zero_l1": _finite(metrics["zero_query"]["l1"], "zero L1"),
                "shuffled_l1": _finite(metrics["shuffled_query"]["l1"], "shuffled L1"),
                "query_change_zero": _finite(
                    row["paired_query_effect"]["prediction_change_l1_zero"],
                    "zero query change",
                ),
                "query_change_shuffled": _finite(
                    row["paired_query_effect"]["prediction_change_l1_shuffled"],
                    "shuffled query change",
                ),
                "delta_l1_from_initial": prediction_l1 - initial_l1,
                "relative_l1_improvement": (initial_l1 - prediction_l1)
                / max(initial_l1, 1.0e-12),
            }
        )
    return result


def training_summary(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    train = [row for row in rows if row.get("event") == "train"]
    if not train:
        return None
    steps = [int(row["step"]) for row in train]
    elapsed = [_finite(row["elapsed_seconds"], "train elapsed") for row in train]
    losses = [_finite(row["loss_l1"], "train loss") for row in train]
    gradients = [_finite(row["gradient_norm"], "gradient norm") for row in train]
    if steps != sorted(set(steps)) or elapsed != sorted(elapsed):
        raise ValueError(
            "training metrics must have strictly increasing steps and time"
        )
    tail_size = min(10, len(losses))
    throughput = steps[-1] / elapsed[-1] * 3600.0 if elapsed[-1] > 0 else None
    return {
        "points": len(train),
        "latest_step": steps[-1],
        "active_hours": elapsed[-1] / 3600.0,
        "first_logged_loss": losses[0],
        "latest_loss": losses[-1],
        "tail_median_loss": statistics.median(losses[-tail_size:]),
        "latest_gradient_norm": gradients[-1],
        "optimizer_steps_per_hour": throughput,
        "latest_learning_rates": train[-1]["learning_rates"],
    }


def completion_audit(
    *,
    run: dict[str, Any],
    evaluations: list[dict[str, Any]],
    result: dict[str, Any] | None,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    settings = run["config"]["train"]
    expected_hours = [0.0, 8.0, 16.0, 24.0]
    observed_hours = [float(row["active_hours"]) for row in evaluations]
    cadence_ok = len(observed_hours) >= 4 and all(
        abs(observed - expected) <= 0.25
        for observed, expected in zip(observed_hours[:4], expected_hours, strict=True)
    )
    periodic = sorted(checkpoint_dir.glob("checkpoint_step-*.pt"))
    final_checkpoint = checkpoint_dir / "checkpoint_last.pt"
    return {
        "requested_world_size": int(settings["expected_world_size"]),
        "requested_duration_hours": float(settings["max_duration_hours"]),
        "requested_eval_interval_hours": float(settings["eval_interval_hours"]),
        "requested_checkpoint_interval_hours": float(
            settings["checkpoint_interval_hours"]
        ),
        "observed_evaluation_hours": observed_hours,
        "evaluation_cadence_complete": cadence_ok,
        "periodic_checkpoint_count": len(periodic),
        "final_checkpoint_exists": final_checkpoint.is_file(),
        "result_exists": result is not None,
        "complete": (
            result is not None
            and cadence_ok
            and len(periodic) >= 2
            and final_checkpoint.is_file()
            and float(result["elapsed_seconds"]) >= 24.0 * 3600.0
        ),
    }


def summarize(run_dir: Path, checkpoint_dir: Path | None = None) -> dict[str, Any]:
    run = _load_json(run_dir / "run.json")
    rows = _load_jsonl(run_dir / "metrics.jsonl")
    resolved_checkpoint_dir = checkpoint_dir or Path(run["checkpoint_dir"])
    result_path = run_dir / "result.json"
    result = _load_json(result_path) if result_path.is_file() else None
    evaluations = evaluation_summary(rows)
    return {
        "run_dir": str(run_dir.resolve()),
        "checkpoint_dir": str(resolved_checkpoint_dir.resolve()),
        "train_examples": int(run["train_examples"]),
        "eval_examples": int(run["eval_examples"]),
        "world_size": int(run["world_size"]),
        "training": training_summary(rows),
        "evaluations": evaluations,
        "audit": completion_audit(
            run=run,
            evaluations=evaluations,
            result=result,
            checkpoint_dir=resolved_checkpoint_dir,
        ),
    }


def main() -> None:
    args = parse_args()
    summary = summarize(args.run_dir, args.checkpoint_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.require_complete and not summary["audit"]["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
