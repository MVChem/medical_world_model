"""Adapt all three native forecasting branches to the established GREEN runner."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from evaluate import atomic_rows
from run import atomic_json, digest
from score_results import rows, validate_generation

HERE = Path(__file__).resolve().parent
CONDITIONS = ("native", "slots", "shuffled")


def build_bridge(root, resume=False):
    bridge = root / "green"
    batches = {condition: validate_generation(root / condition / "evaluation_test", condition)
               for condition in CONDITIONS}
    cfg, cohort, _ = batches["slots"]
    expected = [row["id"] for row in cohort]
    obs = {row["id"]: row for row in rows(Path(cfg["cache"]) / "observations.jsonl")}
    references = [dict(id=row["id"], target_report=obs[row["target"]]["report"]) for row in cohort]
    contract = dict(expected_pairs=297, conditions=list(CONDITIONS), cache=str(Path(cfg["cache"]).resolve()),
        cohort_sha256=digest(Path(cfg["cache"]) / "test.jsonl"),
        observations_sha256=digest(Path(cfg["cache"]) / "observations.jsonl"),
        predictions_sha256={condition: digest(root / condition / "evaluation_test/predictions.jsonl") for condition in CONDITIONS})
    for condition, (other_cfg, other_cohort, _) in batches.items():
        if [row["id"] for row in other_cohort] != expected or other_cfg["cache"] != cfg["cache"]:
            raise ValueError(f"GREEN branch {condition} has a different reference cohort")
    if bridge.exists() and any(bridge.iterdir()):
        if not resume or not (bridge / "bridge_contract.json").exists() or json.loads((bridge / "bridge_contract.json").read_text()) != contract:
            raise FileExistsError("Existing GREEN bridge requires --resume with identical cohort and predictions")
    (bridge / "cohort").mkdir(parents=True, exist_ok=True)
    atomic_json(bridge / "bridge_contract.json", contract)
    atomic_rows(bridge / "cohort/table1_references_test.jsonl", references)
    for condition, (_, _, predicted) in batches.items():
        directory = bridge / condition / "test"
        directory.mkdir(parents=True, exist_ok=True)
        atomic_rows(directory / "responses.jsonl", [dict(key="table1_report|" + row["id"] + "|", ok=True, text=row["report"])
                                                     for row in predicted])
        atomic_json(bridge / condition / "inference_finished.json", dict(status="complete", count=297))
    atomic_json(bridge / "models.json", [dict(id=condition) for condition in CONDITIONS])
    return bridge, expected


def validate_green(root, bridge, expected):
    results, complete = {}, True
    for condition in CONDITIONS:
        directory = bridge / condition / "test"
        metrics_path, responses_path = directory / "green_metrics.json", directory / "green_responses.jsonl"
        result = json.loads(metrics_path.read_text()) if metrics_path.exists() else dict(status="unavailable", reason="GREEN metrics missing")
        responses = rows(responses_path) if responses_path.exists() else []
        ok = (result.get("status") == "complete" and result.get("n") == 297 and result.get("completed") == 297
              and [row["id"] for row in responses] == expected and isinstance(result.get("mean"), (float, int))
              and math.isfinite(result["mean"]))
        result["validated_complete"] = ok
        results[condition] = result
        complete &= ok
        clinical_path = root / condition / "evaluation_test/metrics.json"
        if clinical_path.exists():
            clinical = json.loads(clinical_path.read_text())
            clinical["green"] = result.get("mean") if ok else None
            clinical["green_status"] = result.get("status", "unavailable")
            clinical["green_provenance"] = dict(metrics_path=str(metrics_path),
                metrics_sha256=digest(metrics_path) if metrics_path.exists() else None,
                responses_sha256=digest(responses_path) if responses_path.exists() else None,
                invalid_outputs=result.get("invalid_outputs"), completed=result.get("completed"), n=result.get("n"))
            clinical["native_forecast_available_metrics_complete"] = ok and bool(clinical.get("native_forecast_clinical_complete"))
            clinical["native_forecast_pending"] = ["adjudicated Direction labels"] + ([] if ok else ["GREEN"])
            clinical["table1_ready"] = False
            atomic_json(clinical_path, clinical)
    summary = dict(status="complete" if complete else "unavailable", expected_pairs=297, conditions=results,
                   table1_ready=False, pending=["adjudicated Direction labels"], updated_unix=time.time())
    atomic_json(bridge / "bridge_status.json", summary)
    return complete


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    os.umask(0o077)
    root = args.run.resolve()
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu.isdigit() or int(gpu) not in (0, 1, 2, 3, 4, 7):
        raise ValueError("GREEN requires exactly one queue-allocated physical GPU from 0,1,2,3,4,7")
    bridge, expected = build_bridge(root, args.resume)
    script = HERE.parent / "medworld_baselines/green_eval.py"
    atomic_json(bridge / "execution.json", dict(script=str(script), sha256=digest(script), gpu=int(gpu)))
    atomic_json(bridge / "bridge_status.json", dict(status="running", pid=os.getpid(), expected_pairs=297))
    code = subprocess.run([sys.executable, str(script), "--run", str(bridge), "--gpu", gpu],
                          cwd=script.parent, env=dict(os.environ, HF_HUB_OFFLINE="1")).returncode
    complete = validate_green(root, bridge, expected)
    if code:
        status = json.loads((bridge / "bridge_status.json").read_text())
        status.update(status="failed", returncode=code)
        atomic_json(bridge / "bridge_status.json", status)
    return 0 if code == 0 and complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
