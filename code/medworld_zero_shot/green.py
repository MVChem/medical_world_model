"""Reuse the frozen official GREEN evaluator, preserving the two time directions."""
import argparse
import os
from pathlib import Path
import subprocess
import shutil
import sys
from .common import atomic, digest, read, rows, write_rows


def main(args):
    run = args.run.resolve()
    bridge = run / "green"
    refs = rows(run / "cohort/table1_references_test.jsonl")
    available = []
    inventory = [dict(id=name) for name in args.models.split(",")] if args.models else read(run / "models.json")
    for model in inventory:
        out = run / model["id"] / "test"
        if not (out / "temporal.jsonl").exists():
            continue
        predicted = rows(out / "temporal.jsonl")
        if [r["id"] for r in predicted] != [r["id"] for r in refs]:
            raise ValueError("GREEN cohort differs")
        directory = bridge / model["id"] / "test"
        write_rows(directory / "responses.jsonl", [dict(key="table1_report|" + r["id"] + "|", text=r["prediction"], ok=True) for r in predicted])
        atomic(bridge / model["id"] / "inference_finished.json", {"complete":True})
        available.append(model)
    if not available:
        raise ValueError("No complete predictions for GREEN")
    write_rows(bridge / "cohort/table1_references_test.jsonl", refs)
    atomic(bridge / "models.json", available)
    # Run a frozen local copy; generalize only its historical cohort-count description.
    origin = Path(__file__).resolve().parents[1] / "medworld_baselines/green_eval.py"
    frozen_official = origin.parent / "green_official"
    if frozen_official.exists():
        shutil.copytree(frozen_official, bridge / "green_official", dirs_exist_ok=True)
    script = bridge / "green_eval.py"
    source = origin.read_text().replace("all 297 records retained in denominator", "all frozen reference records retained in denominator")
    source = source.replace("CUDA_VISIBLE_DEVICES=str(args.gpu)", "CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES', str(args.gpu))")
    script.write_text(source)
    atomic(bridge / "provenance.json", dict(original_sha256=digest(origin), executed_sha256=digest(script),
        change="generalize cohort-size description; preserve the queue's reserved GPU UUID; official GREEN prompt/parser/inference unchanged",
        gpu_visibility=os.environ.get("CUDA_VISIBLE_DEVICES"), references_sha256=digest(bridge / "cohort/table1_references_test.jsonl")))
    subprocess.run([sys.executable, str(script), "--run", str(bridge), "--gpu", str(args.gpu)], check=True)
    for model in available:
        out = run / model["id"] / "test"
        evaluated = rows(bridge / model["id"] / "test/green_responses.jsonl")
        if [r["id"] for r in evaluated] != [r["id"] for r in refs]:
            raise ValueError("Incomplete GREEN output")
        metrics = read(out / "metrics.json")
        for direction in ("forward", "backward"):
            panel = [p for p, r in zip(evaluated, refs) if r["direction"] == direction]
            metrics["table1_" + direction].update(green=sum(p["score"] for p in panel)/len(panel),
                green_n=len(panel), green_invalid_outputs=sum(not p["valid"] for p in panel))
        atomic(out / "metrics.json", metrics)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--run", type=Path, required=True); p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--models", help="Optional comma-separated names, including a separately scored Ours export")
    main(p.parse_args())
