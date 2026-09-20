"""Explicit label-set metrics: invalid/missing responses never receive empty-answer credit."""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .common import atomic, digest, read, rows, write_rows

INVALID = "__invalid_output__"


def normalize(value):
    return " ".join(value.lower().split())


def parse(text, vocabulary):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)[:-3].strip()
    try:
        values = json.loads(text)
    except (ValueError, TypeError):
        return {INVALID}, "invalid_json"
    if not isinstance(values, list) or any(not isinstance(x, str) for x in values):
        return {INVALID}, "not_string_array"
    predicted = {normalize(v) for v in values}
    if not predicted <= set(vocabulary):
        return predicted | {INVALID}, "unknown_label"
    return predicted, None


def comparison(reference, prediction, vocabulary):
    target = {normalize(v) for v in reference["answer"]}
    if prediction is None or not prediction.get("ok"):
        predicted, error = {INVALID}, "missing_or_failed_request"
    elif prediction.get("finish_reason") == "length":
        predicted, error = {INVALID}, "truncated_output"
    else:
        predicted, error = parse(prediction["text"], vocabulary)
    # Unknown labels invalidate the full response; no partial credit for malformed output.
    if error:
        predicted = {INVALID}
    return {
        "id": reference["id"],
        "patient": reference["patient"],
        "semantic_type": reference["semantic_type"],
        "content_type": reference["content_type"],
        "regional": reference["regional"],
        "target": sorted(target),
        "predicted": sorted(predicted),
        "error": error,
        "exact": int(target == predicted),
        "tp": len(target & predicted),
        "fp": len(predicted - target),
        "fn": len(target - predicted),
        "empty_target": not target,
    }


def summarize(records, bootstrap=0, seed=20260916):
    n = len(records)
    if not n:
        return {"n": 0, "exact_match": None, "micro_f1": None}
    tp, fp, fn = (sum(r[k] for r in records) for k in ("tp", "fp", "fn"))
    result = {
        "n": n,
        "patients": len({r["patient"] for r in records}),
        "exact_match": sum(r["exact"] for r in records) / n,
        "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "sample_f1": sum(
            2 * r["tp"] / (2 * r["tp"] + r["fp"] + r["fn"])
            if 2 * r["tp"] + r["fp"] + r["fn"]
            else 1.0
            for r in records
        )
        / n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "invalid": sum(r["error"] is not None for r in records),
        "errors": dict(Counter(r["error"] for r in records if r["error"])),
        "empty_reference_n": sum(r["empty_target"] for r in records),
    }
    if bootstrap:
        patients = sorted({r["patient"] for r in records})
        index = {p: i for i, p in enumerate(patients)}
        stats = np.zeros((len(patients), 5))
        for r in records:
            stats[index[r["patient"]]] += [1, r["exact"], r["tp"], r["fp"], r["fn"]]
        rng = np.random.default_rng(seed)
        draws = stats[
            rng.integers(0, len(patients), size=(bootstrap, len(patients)))
        ].sum(1)
        exact = draws[:, 1] / draws[:, 0]
        den = 2 * draws[:, 2] + draws[:, 3] + draws[:, 4]
        f1 = np.divide(2 * draws[:, 2], den, out=np.zeros_like(den), where=den != 0)
        result["patient_bootstrap_95ci"] = {
            "replicates": bootstrap,
            "seed": seed,
            "exact_match": np.quantile(exact, [0.025, 0.975]).tolist(),
            "micro_f1": np.quantile(f1, [0.025, 0.975]).tolist(),
        }
    return result


def score(run, name):
    references = rows(run / "references.jsonl")
    predictions = {}
    for r in rows(run / name / "predictions.jsonl"):
        if r.get("ok") or r["id"] not in predictions:
            predictions[r["id"]] = r
    if set(predictions) - {r["id"] for r in references}:
        raise ValueError("Unexpected prediction IDs")
    vocabulary = read(run / "vocabulary.json")
    records = [comparison(r, predictions.get(r["id"]), vocabulary) for r in references]
    excluded_ids = set(read(run / "training_overlap.json")["overlap_sample_ids"])
    clean = [r for r in records if r["id"] not in excluded_ids]
    diagnosis_all = [r for r in records if r["content_type"] not in ("plane", "gender")]
    diagnosis = [r for r in diagnosis_all if r["id"] not in excluded_ids]
    metrics = {
        "model": name,
        "complete": all(predictions.get(r["id"], {}).get("ok") for r in references),
        "protocol_sha256": digest(run / "protocol.json"),
        "predictions_sha256": digest(run / name / "predictions.jsonl"),
        "overall": summarize(records, 1000),
        "diagnosis": summarize(diagnosis, 1000),
        "diagnosis_all_sampled": summarize(diagnosis_all, 1000),
        "vqa_train_disjoint": summarize(clean, 1000),
        "by_semantic": {
            s: summarize([r for r in records if r["semantic_type"] == s])
            for s in ("verify", "choose", "query")
        },
        "by_content": {
            c: summarize([r for r in records if r["content_type"] == c])
            for c in sorted({r["content_type"] for r in records})
        },
        "by_region": {
            str(flag): summarize([r for r in records if r["regional"] == flag])
            for flag in (True, False)
        },
        "empty_answers": summarize([r for r in records if r["empty_target"]]),
        "nonempty_answers": summarize([r for r in records if not r["empty_target"]]),
    }
    write_rows(run / name / "scored_predictions.jsonl", records)
    atomic(run / name / "metrics.json", metrics)
    return metrics


def render(run):
    protocol = read(run / "protocol.json")
    models = read(run / "models.json")
    audit = read(run / "training_overlap.json")
    excluded = set(audit["overlap_sample_ids"])
    diagnosis_refs = [
        r
        for r in rows(run / "references.jsonl")
        if r["id"] not in excluded and r["content_type"] not in ("plane", "gender")
    ]
    diagnosis_n = len(diagnosis_refs)
    diagnosis_patients = len({r["patient"] for r in diagnosis_refs})
    table = []
    lines = [
        "# MIMIC-CXR-VQA 第一版抽样结果",
        "",
        f"官方 test 抽样 {protocol['n']:,} 题 / {protocol['patients']} 患者，seed={protocol['seed']}。",
        "所有数字为百分数；零样本、BF16、同图同题同词表、GPU 0 串行推理。",
        "统一启用 JSON 数组与公开 110 标签词表约束解码；这属于受限词表 VQA。"
        if protocol["generation"].get("structured_outputs")
        else "使用自由生成，按 JSON 数组格式评分。",
        "",
        "| 模型 | 状态 | Diagnosis EM | Diagnosis μF1 | 全部 VQA EM | 全部 VQA μF1 | Verify Acc | Choose EM | Query μF1 | 无效输出 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    tex = [
        "% Sampled current-image diagnosis pilot; separate from future-state metrics.",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Diagnosis EM & Diagnosis $\mu$F1 & VQA EM & VQA $\mu$F1 \\",
        r"\midrule",
    ]
    for model in models:
        name = model["id"]
        path = run / name / "metrics.json"
        metrics = read(path) if path.exists() else None
        status_path = run / name / "status.json"
        status = read(status_path) if status_path.exists() else {}
        entry = {"model": model["label"], "status": status.get("status", "pending")}
        if metrics and metrics["complete"]:
            entry.update(
                status="complete",
                diagnosis_n=metrics["diagnosis"]["n"],
                diagnosis_em=metrics["diagnosis"]["exact_match"],
                diagnosis_micro_f1=metrics["diagnosis"]["micro_f1"],
                vqa_em=metrics["overall"]["exact_match"],
                vqa_micro_f1=metrics["overall"]["micro_f1"],
                verify_acc=metrics["by_semantic"]["verify"]["exact_match"],
                choose_em=metrics["by_semantic"]["choose"]["exact_match"],
                query_micro_f1=metrics["by_semantic"]["query"]["micro_f1"],
                invalid=metrics["overall"]["invalid"],
            )
            values = [
                f"{entry[k] * 100:.2f}"
                for k in (
                    "diagnosis_em",
                    "diagnosis_micro_f1",
                    "vqa_em",
                    "vqa_micro_f1",
                    "verify_acc",
                    "choose_em",
                    "query_micro_f1",
                )
            ]
            lines.append(
                "| "
                + " | ".join(
                    [model["label"], "complete", *values, str(entry["invalid"])]
                )
                + " |"
            )
            tex.append(model["label"] + " & " + " & ".join(values[:4]) + r" \\")
        else:
            state = f"{entry['status']} {status.get('completed', 0)}/{protocol['n']}"
            lines.append(
                "| " + " | ".join([model["label"], state, *(["—"] * 8)]) + " |"
            )
            tex.append(model["label"] + " & " + " & ".join(["TBD"] * 4) + r" \\")
        table.append(entry)
    lines += [
        "",
        f"Diagnosis 使用临床内容且与当前统一训练患者无交集的 {diagnosis_n} 题 / {diagnosis_patients} 患者；排除 plane/gender 和全部训练重叠患者。仍包含器械和技术评估相关题，不能解释为纯疾病分类。",
        "EM 是整组答案完全一致的比例；μF1 按答案标签累计 TP/FP/FN。[] 与 [] 的 EM=1；空集不增加 micro F1 的 TP。",
        "仅接受 JSON 字符串数组（可带代码围栏），忽略大小写和多余空格；不做同义词补救、LLM 评分或答案筛选。非法、缺失、截断输出均计错。",
        "患者 bootstrap 95% CI、题型/内容/区域/空答案分项见各模型 metrics.json。",
        "",
        f"当前统一训练数据审计：抽样患者与任一任务 train 重叠 {audit['sample_train_overlap']} 人；官方完整 test 重叠 {audit['full_test_train_overlap']} 人。",
        "这是公开预训练模型的第一版抽样比较，未微调；公开预训练是否见过这些患者无法从本地验证。",
        "本地集合评分实现尚未与官方 evaluator 逐项对齐，因此不宣称可直接复现论文全量 benchmark 数值。",
        "若加入 Table 1，应独立标注 Current-image Diagnosis (CXR-VQA, sampled)，与现有 future-state AP/AUROC 的队列和任务分开。",
        "",
        "来源：[PhysioNet 数据说明](https://physionet.org/content/mimic-ext-mimic-cxr-vqa/1.0.0/)；",
        "[公开答案词表](https://github.com/baeseongsu/mimic-cxr-vqa/blob/master/mimiccxrvqa/dataset/ans2idx.json)。",
        "",
        "复现、进度和逐题结果均位于本目录；protocol.json 固定样本与预处理，source/ 为执行源码快照。",
    ]
    (run / "REPORT.md").write_text("\n".join(lines) + "\n")
    tex += [
        r"\bottomrule",
        r"\end{tabular}",
        "% Scores x100. Diagnosis excludes plane/gender AND all patients in active project training.",
        f"% Diagnosis: {diagnosis_n} questions / {diagnosis_patients} patients; VQA: {protocol['n']} questions / {protocol['patients']} patients.",
        f"% {protocol['n']} sampled questions / {protocol['patients']} patients; pilot only.",
    ]
    (run / "table1_diagnosis_pilot.tex").write_text("\n".join(tex) + "\n")
    keys = [
        "model",
        "status",
        "diagnosis_n",
        "diagnosis_em",
        "diagnosis_micro_f1",
        "vqa_em",
        "vqa_micro_f1",
        "verify_acc",
        "choose_em",
        "query_micro_f1",
        "invalid",
    ]
    with (run / "table1_diagnosis_pilot.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(table)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--model")
    args = p.parse_args()
    if args.model:
        score(args.run, args.model)
    render(args.run)
