"""Run the established isolated Table-1 clinical scorer on native forecasts."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from run import atomic_json, digest

HERE = Path(__file__).resolve().parent
TABLE1 = HERE.parent / "medworld_table1"
FALLBACK = TABLE1 / "runs/qwen9b_ablation_20260914/slots/config.json"


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_generation(out, condition):
    generation = json.loads((out / "generation.json").read_text())
    if generation.get("state") != "complete" or generation.get("smoke_only"):
        raise ValueError("Clinical scoring requires complete formal predictions")
    if generation.get("split") != "test" or generation.get("condition") != condition:
        raise ValueError("Generation split/condition differs from the requested clinical scoring job")
    cfg = json.loads((out / "config.json").read_text())
    expected = rows(Path(cfg["cache"]) / "test.jsonl")
    predicted = rows(out / "predictions.jsonl")
    if len(expected) != 297 or generation.get("count") != 297:
        raise ValueError("The fixed final Table-1 cohort must contain all 297 pairs")
    if [row["id"] for row in predicted] != [row["id"] for row in expected]:
        raise ValueError("Clinical scoring prediction IDs/order differ from the 297-pair cohort")
    return cfg, expected, predicted


def scoring_config(cfg, fallback=FALLBACK):
    result = dict(cfg)
    previous = json.loads(Path(fallback).read_text())
    for key in ("vjepa_checkpoint", "data_root"):
        if key not in result:
            result[key] = previous[key]
    result["retrieval_negatives"] = 3
    return result


def score(args):
    os.umask(0o077)
    out = Path(args.out).resolve()
    cfg, expected, _ = validate_generation(out, args.condition)
    scoring = scoring_config(cfg, args.fallback_config)
    config_path = out / "scoring_config.json"
    atomic_json(config_path, scoring)
    provenance = dict(architecture="native_qwen_with_optional_predicted_4plus4_slots_v1",
        condition=args.condition, probability_source="same supervised finding head sigmoid across native/slots/shuffled",
        prediction_sha256=digest(out / "predictions.jsonl"), generation_sha256=digest(out / "generation.json"),
        scorer=str(TABLE1 / "score.py"), scorer_sha256=digest(TABLE1 / "score.py"),
        clinical_evaluate_sha256=digest(TABLE1 / "evaluate.py"), scoring_config_sha256=digest(config_path),
        train_config_unchanged=True, retrieval_negatives=3, expected_pairs=len(expected))
    atomic_json(out / "scoring_provenance.json", provenance)
    atomic_json(out / "clinical_status.json", dict(state="running", pid=os.getpid(), updated_unix=time.time()))
    command = [sys.executable, str(TABLE1 / "score.py"), "--config", str(config_path),
               "--mode", "ours", "--split", "test", "--out", str(out), "--score-only"]
    try:
        subprocess.run(command, cwd=TABLE1, env=dict(os.environ, HF_HUB_OFFLINE="1"), check=True)
        metrics = json.loads((out / "metrics.json").read_text())
        if metrics.get("n") != 297:
            raise ValueError("Clinical scorer returned a different cohort size")
        metrics.update(architecture=provenance["architecture"], condition=args.condition,
                       mode="native_forecast_" + args.condition, state_condition=args.condition,
                       probability_source=provenance["probability_source"], native_forecast_provenance=provenance)
        unavailable = []
        for key in ("ap", "auroc", "brier", "ece", "transition_f1", "radgraph_f1", "chexbert_f1"):
            value = metrics.get(key)
            if not isinstance(value, (float, int)) or not math.isfinite(value):
                unavailable.append(key)
        metrics["native_forecast_clinical_complete"] = not unavailable
        metrics["table1_ready"] = False
        metrics["native_forecast_pending"] = ["GREEN", "adjudicated Direction labels"] + unavailable
        atomic_json(out / "metrics.json", metrics)
        atomic_json(out / "clinical_status.json", dict(state="complete" if not unavailable else "unavailable",
            n=297, unavailable=unavailable, radgraph_status=metrics.get("radgraph_status"),
            condition=args.condition, updated_unix=time.time(), table1_ready=False))
        return 0 if not unavailable else 2
    except BaseException as error:
        atomic_json(out / "clinical_status.json", dict(state="failed", error_type=type(error).__name__,
                    error=str(error), updated_unix=time.time(), table1_ready=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Existing evaluation_test directory with complete predictions")
    parser.add_argument("--condition", required=True, choices=("native", "slots", "shuffled"))
    parser.add_argument("--fallback-config", default=str(FALLBACK))
    return score(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
