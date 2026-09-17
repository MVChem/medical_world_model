import os
from pathlib import Path
import sys

PROJECT = Path(os.environ.get("MEDWORLD_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
os.environ.setdefault("MEDWORLD_PROJECT", str(PROJECT))
BASELINES = Path(__file__).resolve().parents[1] / "medworld_baselines"
sys.path.insert(0, str(BASELINES))
from base import atomic, digest, models, read, rows, write_rows, CURRENT_FINDINGS, FUTURE_FINDINGS

__all__ = ["PROJECT", "atomic", "digest", "models", "read", "rows", "write_rows",
           "CURRENT_FINDINGS", "FUTURE_FINDINGS", "verify_contract", "require_complete",
           "temporal_input", "prompt_for", "groups", "key"]


def verify_contract(run):
    protocol = read(run / "protocol.json")
    for name, expected in protocol["file_sha256"].items():
        if digest(run / name) != expected:
            raise ValueError(f"Frozen evaluation artifact changed: {name}")
    return protocol


def require_complete(expected, predictions):
    if len(predictions) != len(expected) or [p["id"] for p in predictions] != [r["id"] for r in expected]:
        raise ValueError("Predictions must cover the entire ordered reference cohort")


def temporal_input(pair, source, tokenizer, context_tokens):
    # Exactly the observation text retained by MedWorld.encode, including its prefix.
    text = "Chest radiograph observation.\nReport:\n" + source["report"]
    ids = tokenizer(text, truncation=True, max_length=context_tokens)["input_ids"]
    return dict(id=pair["id"], patient=str(pair["patient"]), direction=pair["direction"],
                delta_hours=pair["delta_hours"], source_observation_text=tokenizer.decode(ids, skip_special_tokens=True),
                source_text_token_ids=ids)


def prompt_for(task, row, finding=None):
    if task.startswith("table1_"):
        delta = row["delta_hours"]
        if (delta > 0) != (row["direction"] == "forward") or delta == 0:
            raise ValueError("Signed time and requested direction disagree")
        relation = "later" if delta > 0 else "earlier"
        context = (row["source_observation_text"] + f"\n\nRequested target observation: {abs(delta):.8g} hours {relation} "
                   "than the supplied observation. Only the supplied observation is available.\n")
        if task == "table1_report":
            return context + "Predict the target chest radiograph report. Write only FINDINGS and IMPRESSION, without reasoning or introductory text."
        return context + f"Will {finding.lower()} be present on the target chest radiograph? Answer exactly Yes or No."
    if task == "table2_report":
        return "Describe this chest radiograph. Write only FINDINGS and IMPRESSION, without reasoning or introductory text."
    if task == "table2_prob":
        return f"Is {finding.lower()} present on this chest radiograph? Answer exactly Yes or No."
    raise ValueError(task)


def groups(run):
    temporal = rows(run / "cohort/table1_inputs_test.jsonl")
    current = rows(run / "cohort/table2_inputs_test.jsonl")
    return {
        "table1_report": [(r, None) for r in temporal],
        "table1_prob": [(r, f) for r in temporal for f in FUTURE_FINDINGS],
        "table2_report": [(r, None) for r in current if r["report_generation"]],
        "table2_prob": [(r, f) for r in current if r["classification"] for f in CURRENT_FINDINGS],
    }


def key(task, row, finding):
    return task + "|" + row["id"] + "|" + (finding or "")
