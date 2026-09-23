"""Shared future-task references and scorers for trained and native Qwen arms."""
import json
from pathlib import Path

from .future_metrics import (SCHEMA, DIRECTIONS, OfficialRadGraphScorer,
                             reference_fingerprint)
from ..datasets.vqa import VOCABULARY


def reference_for_row(task, row, future_data):
    reference = {"id": row["id"], "patient": row["patient"]}
    if task == "future_vqa":
        target = row["answer"]
    elif task == "progression":
        target = DIRECTIONS[int(row["label"])]
        reference["finding"] = row["finding"]
    elif task == "future_report":
        target = (future_data.cxr_root / row["target_report_file"]).read_text(encoding="utf-8")
        if not target.strip():
            raise ValueError("Empty future report reference")
    elif task in ("mortality_30d", "remaining_los"):
        target = float(row["label"])
    else:
        raise ValueError(f"Unsupported future task: {task}")
    reference["target"] = target
    return reference


def load_radgraph(cfg):
    root = Path(cfg["radgraph_assets"]).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    return OfficialRadGraphScorer(
        manifest["provenance"], model_cache_dir=root / manifest["model_cache_dir"],
        tokenizer_cache_dir=root / manifest["tokenizer_cache_dir"], cuda=-1,
        worker_runtime=manifest["worker_runtime"])


def scoring_protocol(task, references, cfg, radgraph_scorer=None):
    protocol = {"schema": SCHEMA, "task": task,
                "unit": "days" if task == "remaining_los" else "fraction",
                "reference_sha256": reference_fingerprint(references)}
    if task == "future_vqa":
        protocol["vocabulary"] = list(VOCABULARY)
    elif task == "progression":
        protocol["findings"] = sorted({row["finding"] for row in references})
    elif task == "mortality_30d":
        protocol["horizon_days"] = 30
    elif task == "future_report":
        if radgraph_scorer is None:
            root = Path(cfg["radgraph_assets"])
            protocol["radgraph"] = json.loads((root / "manifest.json").read_text())["provenance"]
        else:
            protocol["radgraph"] = radgraph_scorer.provenance
    return protocol
