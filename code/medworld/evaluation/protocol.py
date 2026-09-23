"""The current paper's two metric tables; earlier result schemas are rejected."""
from copy import deepcopy

PROTOCOL = "medworld-tables-v1"
FUTURE_TASKS = ("future_vqa", "progression", "future_report", "mortality_30d", "remaining_los")
TABLE2_METRICS = {
    "classification": ("macro_auroc", "macro_ap"),
    "vqa": ("exact_match", "micro_f1"),
    "segmentation": ("mean_dice", "mean_iou"),
}
TABLE1_METRICS = {
    "future_vqa": "accuracy", "progression": "balanced_accuracy",
    "future_report": "radgraph_f1", "mortality_30d": "auroc", "remaining_los": "mae_days",
}

_DEFINITIONS = {
    "table1": {
        "metrics": {task: [name] for task, name in TABLE1_METRICS.items()},
        "inputs": "Source image/report and prespecified request only; no follow-up image/report or outcome as input",
        "future_vqa": "One official categorical reference answer; fixed normalized vocabulary; accuracy",
        "progression": "Human source-anchored region/finding test labels: improved/stable/worsened; pooled three-class balanced accuracy and class support",
        "future_report": "Official RadGraph-XL partial F1; full observed target report; scorer and model artifacts pinned separately",
        "mortality_30d": "Observed death within 30 days of source examination; continuous risk AUROC; ambiguous/censored labels excluded before inference",
        "remaining_los": "Admission discharge minus source examination in days; continuous nonnegative prediction MAE; fixed 24-hour latent readout request",
        "cohorts": "Patient-disjoint fixed task references; outcome cohorts include patients without a subsequent image",
        "missing_predictions": "Never drop references; incomplete jobs have no aggregate score",
    },
    "table2": {
        "metrics": {task: list(names) for task, names in TABLE2_METRICS.items()},
        "classification": "13 report-derived findings; reference labels 0/1 only; macro AUROC and AP over findings with both classes",
        "vqa": "Fixed official answer vocabulary; lowercased and surrounding-whitespace-trimmed JSON answer sets; exact match and pooled micro F1; malformed answers count as errors",
        "segmentation": "Human-reviewed masks only; sigmoid>0.5 inside annotated ROI; empty/empty score 1; MRI counts summed by volume before channel/volume averaging",
        "segmentation_main_datasets": ["mimic_cxr_human", "ucsf_alptdg", "mu_glioma_post"],
        "segmentation_external_dataset": "montgomery",
        "segmentation_table_aggregation": "Equal mean of the three internal dataset scores; each dataset and external Montgomery also reported separately",
        "native_segmentation": "N/A: native Qwen has no segmentation head",
        "references": "Exact ordered IDs and canonical reference SHA256, including segmentation target and ROI digest",
    },
}


def metric_protocol(table="table2"):
    if table not in _DEFINITIONS:
        raise ValueError(f"Unknown metric table: {table}")
    return {"schema": PROTOCOL, "table": table, **deepcopy(_DEFINITIONS[table])}


def validate_metric_protocol(summary, table="table2"):
    if not isinstance(summary, dict) or summary.get("metric_protocol") != metric_protocol(table):
        raise ValueError(f"Results must use the current {table} metric protocol; previous metric formats are unsupported")


def training_tasks(cfg):
    from ..downstream_tasks.registry import TASKS
    return (*TASKS, *FUTURE_TASKS) if cfg["future_enabled"] else tuple(TASKS)
