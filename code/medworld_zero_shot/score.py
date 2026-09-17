"""Common complete-cohort scorer, also accepts the unified model's exported predictions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from .common import atomic, digest, read, rows, write_rows, require_complete, verify_contract, CURRENT_FINDINGS, FUTURE_FINDINGS
from stats import probability_metrics


def export_predictions(run, model, ours=None):
    refs1 = rows(run / "cohort/table1_references_test.jsonl")
    refs2 = {r["id"]:r for r in rows(run / "cohort/table2_references_test.jsonl")}
    inputs2 = rows(run / "cohort/table2_inputs_test.jsonl")
    out = run / model / "test"
    out.mkdir(parents=True, exist_ok=True)
    if ours:
        summary = read(ours / "summary.json")
        protocol = read(run / "protocol.json")
        if summary["data_fingerprint"] != protocol["data_fingerprint"] or summary["split"] != "test" or summary["limit"]:
            raise ValueError("Ours must use the same full test data fingerprint")
        provenance = read(ours / "zero_shot_comparison.json")
        if provenance["protocol_sha256"] != digest(run / "protocol.json") or provenance["max_new_tokens"] != 384:
            raise ValueError("Ours generation budget was not validated")
        temporal = rows(ours / "temporal.jsonl")
        current = rows(ours / "classification.jsonl")
        reports = rows(ours / "report.jsonl")
        require_complete(refs1, temporal)
        for ref, pred in zip(refs1, temporal):
            if ref["target_report"] != pred["reference"] or ref["labels"] != pred["labels"]:
                raise ValueError("Ours temporal references differ")
        for ref, pred in zip([refs2[r["id"]] for r in inputs2 if r["classification"]], current):
            if ref["labels"] != pred["labels"]:
                raise ValueError("Ours classification references differ")
        for ref, pred in zip([refs2[r["id"]] for r in inputs2 if r["report_generation"]], reports):
            if ref["report"] != pred["reference"]:
                raise ValueError("Ours report references differ")
    else:
        done = {r["key"]:r for r in rows(out / "responses.jsonl") if r.get("ok")}
        def prediction(task, ref, findings):
            values = [done[task + "_prob|" + ref["id"] + "|" + f]["probability"] for f in findings]
            return dict(id=ref["id"], probabilities=values)
        temporal = []
        for ref in refs1:
            pred = prediction("table1", ref, FUTURE_FINDINGS)
            response = done["table1_report|" + ref["id"] + "|"]
            pred.update(prediction=response["text"], finish_reason=response["finish_reason"])
            temporal.append(pred)
        current = [prediction("table2", r, CURRENT_FINDINGS) for r in inputs2 if r["classification"]]
        reports = [dict(id=r["id"], prediction=done["table2_report|" + r["id"] + "|"]["text"],
                        finish_reason=done["table2_report|" + r["id"] + "|"]["finish_reason"])
                   for r in inputs2 if r["report_generation"]]
    require_complete(refs1, temporal)
    require_complete([r for r in inputs2 if r["classification"]], current)
    require_complete([r for r in inputs2 if r["report_generation"]], reports)
    for ref, pred in zip(refs1, temporal):
        pred.update(reference=ref["target_report"], current_report=ref["current_report"], labels=ref["labels"],
                    current_labels=ref["current_labels"], direction=ref["direction"], patient=ref["patient"])
    for pred in current:
        pred["labels"] = refs2[pred["id"]]["labels"]
    for pred in reports:
        pred["reference"] = refs2[pred["id"]]["report"]
    for name, records in (("temporal", temporal), ("classification", current), ("report", reports)):
        write_rows(out / (name + ".jsonl"), records)
    return out, temporal, current, reports


def score(args):
    os.umask(0o077)
    run = args.run.resolve()
    verify_contract(run)
    out, temporal, current, reports = export_predictions(run, args.model, args.ours)
    result = dict(model=args.model, protocol_sha256=digest(run / "protocol.json"),
                  probability_source="trained head sigmoid" if args.ours else "raw Yes/No conditional likelihood",
                  table2_classification=probability_metrics([r["labels"] for r in current], [r["probabilities"] for r in current], CURRENT_FINDINGS))
    panels = {"table1_"+direction:[r for r in temporal if r["direction"] == direction] for direction in ("forward", "backward")}
    for name, records in panels.items():
        result[name] = probability_metrics([r["labels"] for r in records], [r["probabilities"] for r in records], FUTURE_FINDINGS)
        result[name].update(direction_f1=None, direction_status="adjudicated disease/laterality references unavailable")
    atomic(out / "metrics.json", result)
    if args.quick:
        return
    clinical_root = Path(__file__).resolve().parents[1] / "medworld_table1"
    sys.path.insert(0, str(clinical_root))
    sys.path.insert(0, str(clinical_root / "metric_vendor"))
    from clinical import CheXbert
    from metrics import clinical_metrics
    import numpy as np
    import torch
    panels["table2_report"] = reports
    extractor = CheXbert(device="cuda")
    for name, records in panels.items():
        findings = CURRENT_FINDINGS if name == "table2_report" else FUTURE_FINDINGS
        signature = hashlib.sha256(json.dumps([(r["prediction"], r["reference"]) for r in records]).encode()).hexdigest()
        path = out / (name + "_chexbert.json")
        cached = read(path) if path.exists() else {}
        if cached.get("sha256") != signature:
            truth = extractor.labels([r["reference"] for r in records], findings)
            predicted = extractor.labels([r["prediction"] for r in records], findings)
            # Current reports are clinical transition references, not the truncated model inputs.
            source = extractor.labels([r["current_report"] for r in records], findings) if name.startswith("table1") else truth
            cached = dict(sha256=signature, truth=truth, predicted=predicted, source=source, provenance=extractor.provenance)
            atomic(path, cached)
        section = result.setdefault(name, {})
        clinical = clinical_metrics(cached["source"], cached["truth"], cached["predicted"], None, findings)
        section.update(n=len(records), chexbert_f1=clinical["chexbert_f1"], clinical_details=clinical,
                       empty_reports=sum(not r["prediction"].strip() for r in records),
                       unique_reports=len({r["prediction"] for r in records}),
                       truncated_reports=sum(r.get("finish_reason") == "length" for r in records))
        if name.startswith("table1"):
            section["transition_f1"] = clinical["transition_f1"]
            section["cached_vs_extracted_label_disagreements"] = int((np.array(cached["truth"]) != np.array([r["labels"] for r in records])).sum())
        atomic(out / "metrics.json", result)
    del extractor
    torch.cuda.empty_cache()
    from radgraph import F1RadGraph
    scorer = F1RadGraph(reward_level="all", model_type="radgraph-xl", cuda=0,
                       model_cache_dir=str(clinical_root / "weights/radgraph"))
    for name, records in panels.items():
        path = out / (name + "_radgraph.json")
        signature = hashlib.sha256(json.dumps([(r["prediction"], r["reference"]) for r in records]).encode()).hexdigest()
        cached = read(path) if path.exists() else dict(sha256=signature, per_report=[])
        if cached["sha256"] != signature:
            raise ValueError("RadGraph cache predictions changed")
        for start in range(len(cached["per_report"]), len(records), 24):
            chunk = records[start:start+24]
            _, rewards, _, _ = scorer(hyps=[r["prediction"] for r in chunk], refs=[r["reference"] for r in chunk])
            if len(rewards[1]) != len(chunk):
                raise ValueError("RadGraph returned a different cohort size")
            cached["per_report"].extend(float(x) for x in rewards[1])
            atomic(path, cached)
            print(args.model, name, "RadGraph", len(cached["per_report"]), "/", len(records), flush=True)
        result[name]["radgraph_f1"] = float(np.mean(cached["per_report"]))
        atomic(out / "metrics.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--ours", type=Path)
    p.add_argument("--quick", action="store_true")
    score(p.parse_args())
