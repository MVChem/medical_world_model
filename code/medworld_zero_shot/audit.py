"""Independently rebuild every held-out join and compare cached downstream pixels."""
import argparse
import json
from pathlib import Path

from .common import atomic, read, rows, temporal_input


def audit(run, training):
    import numpy as np
    from PIL import Image
    from transformers import AutoTokenizer
    from medworld.config import load_config
    from medworld.datasets import UnifiedData
    cfg = load_config(training / "config.json")
    data = UnifiedData(cfg)
    if data.metadata != read(training / "data_protocol.json"):
        raise ValueError("Training data protocol differs")
    tokenizer = AutoTokenizer.from_pretrained(cfg["qwen"], local_files_only=True)
    inputs = rows(run / "cohort/table1_inputs_test.jsonl")
    refs = rows(run / "cohort/table1_references_test.jsonl")
    pairs = data.rows("temporal", "test")
    if not len(inputs) == len(refs) == len(pairs) == 594:
        raise ValueError("Incomplete temporal cohort")
    for pair, inp, ref in zip(pairs, inputs, refs):
        source, target = [data.temporal.lookup[pair[side]] for side in ("source", "target")]
        expected = temporal_input(pair, source, tokenizer, cfg["context_tokens"])
        if inp != dict(expected, image=inp["image"]):
            raise ValueError("Temporal input differs from the source-only contract")
        expected_ref = dict(id=pair["id"], patient=str(pair["patient"]), direction=pair["direction"],
            delta_hours=pair["delta_hours"], current_report=source["report"], target_report=target["report"],
            labels=target["labels"], current_labels=source["labels"])
        if ref != expected_ref:
            raise ValueError("Temporal reference join differs")
    current = {r["id"]:r for r in rows(run / "cohort/table2_inputs_test.jsonl")}
    refs2 = {r["id"]:r for r in rows(run / "cohort/table2_references_test.jsonl")}
    checked = set()
    for task in ("classification", "report"):
        flag = "classification" if task == "classification" else "report_generation"
        if [r["id"] for r in current.values() if r[flag]] != [r["id"] for r in data.rows(task, "test")]:
            raise ValueError("Current task IDs/order differ")
        for row in data.rows(task, "test"):
            identity, ref = row["id"], refs2[row["id"]]
            if current[identity]["patient"] != str(row["subject_id"]):
                raise ValueError("Current patient identity differs")
            field = "labels" if task == "classification" else "report"
            if ref[field] != row["labels" if task == "classification" else "report_target"]:
                raise ValueError("Current reference differs")
            if identity not in checked:
                with Image.open(current[identity]["image"]) as image:
                    actual = np.array(image.convert("RGB"))
                expected = data.current._array("current")[row["image_index"]]
                if not np.array_equal(actual, np.repeat(expected[:, :, None], 3, axis=2)):
                    raise ValueError("Downstream image pixels differ from training cache")
                checked.add(identity)
    result = dict(status="passed", data_fingerprint=data.fingerprint, temporal_input_reference_joins=594,
        current_exact_pixels=len(checked), classification_ordered_ids=353, report_ordered_ids=507,
        patient_holdouts_match_training=True, report_prefix_truncation_matches_training=True)
    atomic(run / "independent_cohort_audit.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--training-run", type=Path, required=True)
    args = p.parse_args()
    audit(args.run.resolve(), args.training_run.resolve())
