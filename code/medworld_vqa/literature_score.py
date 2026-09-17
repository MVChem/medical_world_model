"""AOR metric definitions with an explicitly local, conservative text-list adapter.

No answer-aware extraction, semantic judge or test-derived synonym dictionary.
Also execute the two unmodified Meissa scoring functions, as a sensitivity check.
"""

import ast
import csv
import json
import re
from collections import Counter

from .common import atomic, digest, read, rows, write_rows
from .score import INVALID, summarize


def text_labels(text, semantic_type, vocabulary):
    """Short canonical lists; prose/unknown items remain explicit false positives.

    Verify accepts a leading yes/no answer with a token boundary. Other questions
    allow comma/semicolon/newline/and-separated canonical labels, optional JSON,
    Markdown bullets, and the gender spellings used by CheXagent. No substring
    search: 'No pleural effusion' must not become a positive effusion prediction.
    """
    text = text.strip().lower().replace("**", "").replace("`", "")
    if not text:
        return {INVALID}, "empty_response"
    text = re.sub(r"^(?:the\s+)?answer\s*(?:is\s*|:\s*)", "", text).strip()
    if semantic_type == "verify":
        match = re.match(r"^(yes|no)(?=$|[\s,.;:!])", text)
        if match:
            return {match.group(1)}, None
    if text.rstrip(".! ") in {
        "none",
        "no",
        "nothing",
        "no abnormalities",
        "no abnormality",
    }:
        return set(), None
    try:
        values = json.loads(text)
    except (ValueError, TypeError):
        values = None
    if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
        values = re.split(r"[,;\n]|\s+and\s+", text)
    predicted, unknown = set(), False
    for value in values:
        label = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", value)
        label = " ".join(label.strip(" .!\"'").split())
        if not label:
            continue
        label = {"female": "f", "male": "m"}.get(label, label)
        if label in vocabulary:
            predicted.add(label)
        else:
            predicted.add(INVALID)
            unknown = True
    if not predicted and values != []:
        return {INVALID}, "unrecognized_text"
    return predicted, "noncanonical_text" if unknown else None


def comparison(ref, prediction, vocabulary):
    target = set(ref["answer"])
    if not prediction or not prediction.get("ok"):
        predicted, error = {INVALID}, "missing_or_failed_request"
    elif prediction.get("finish_reason") == "length":
        predicted, error = {INVALID}, "truncated_output"
    else:
        predicted, error = text_labels(
            prediction["text"], ref["semantic_type"], vocabulary
        )
    return {
        "id": ref["id"],
        "patient": ref["patient"],
        "semantic_type": ref["semantic_type"],
        "content_type": ref["content_type"],
        "regional": ref["regional"],
        "target": sorted(target),
        "predicted": sorted(predicted),
        "error": error,
        "exact": int(target == predicted),
        "tp": len(target & predicted),
        "fp": len(predicted - target),
        "fn": len(target - predicted),
        "empty_target": not target,
    }


def published_meissa(run):
    """Compile only the exact normalize/check_correct AST nodes, without agent imports."""
    source = run / "literature/Meissa/run_mimic_cxr_vqa.py"
    tree = ast.parse(source.read_text())
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in {"normalize", "check_correct"}
    ]
    assert len(functions) == 2
    namespace = {"re": re}
    exec(  # noqa: S102 -- only two inspected, pinned pure scoring functions
        compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace["check_correct"]


def subsets(records, excluded):
    return {
        "overall": records,
        "vqa_train_disjoint": [r for r in records if r["id"] not in excluded],
        "diagnosis": [
            r
            for r in records
            if r["id"] not in excluded and r["content_type"] not in {"plane", "gender"}
        ],
    }


def score(run, name):
    refs = rows(run / "references.jsonl")
    predictions = {p["id"]: p for p in rows(run / name / "predictions.jsonl")}
    assert not set(predictions) - {r["id"] for r in refs}
    vocabulary = set(read(run / "vocabulary.json"))
    records = [comparison(r, predictions.get(r["id"]), vocabulary) for r in refs]
    check = published_meissa(run)
    for rec, ref in zip(records, refs, strict=True):
        pred = predictions.get(ref["id"], {})
        # Source-exact scorer does not check truncation. Preserve that behavior here.
        rec["published_meissa_correct"] = bool(
            check(pred.get("text", ""), ref["answer"])
        )
    excluded = set(read(run / "training_overlap.json")["overlap_sample_ids"])
    result = {
        "model": name,
        "complete": all(predictions.get(r["id"], {}).get("ok") for r in refs),
        "protocol_sha256": digest(run / "protocol.json"),
        "predictions_sha256": digest(run / name / "predictions.jsonl"),
        "subsets": {},
        "finish_reasons": dict(
            Counter(p.get("finish_reason") for p in predictions.values())
        ),
    }
    for subset, selected in subsets(records, excluded).items():
        summary = summarize(selected, 1000)
        summary["by_semantic"] = {
            s: summarize([r for r in selected if r["semantic_type"] == s], 1000)
            for s in ("verify", "choose", "query")
        }
        summary["published_meissa_any_substring_accuracy"] = sum(
            r["published_meissa_correct"] for r in selected
        ) / len(selected)
        summary["query_empty"] = summarize(
            [r for r in selected if r["semantic_type"] == "query" and r["empty_target"]]
        )
        result["subsets"][subset] = summary
    write_rows(run / name / "scored_predictions.jsonl", records)
    atomic(run / name / "metrics.json", result)
    return result


def render(run):
    lines = [
        "# MIMIC-CXR-VQA：文献方法对齐的零样本复测",
        "",
        "固定同一批 1,024 题；LLaVA 官方短答提示；自由文本生成。",
        "AOR 分题型指标定义 + 本地保守文本解析；不是 AOR 训练或官方评分器复现。",
        "",
        "| 模型 | 状态 | Verify Acc | Choose Acc | Query label μF1 | 非规范/截断输出 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    table = []
    for m in read(run / "models.json"):
        path = run / m["id"] / "metrics.json"
        if path.exists() and (metrics := read(path))["complete"]:
            full = metrics["subsets"]["overall"]
            typed = full["by_semantic"]
            entry = {
                "model": m["label"],
                "verify_acc": typed["verify"]["exact_match"],
                "choose_acc": typed["choose"]["exact_match"],
                "query_label_micro_f1": typed["query"]["micro_f1"],
                "noncanonical_or_truncated": full["invalid"],
            }
            lines.append(
                f"| {m['label']} | complete | {100 * entry['verify_acc']:.2f} | "
                f"{100 * entry['choose_acc']:.2f} | {100 * entry['query_label_micro_f1']:.2f} | "
                f"{full['invalid']} |"
            )
            table.append(entry)
        else:
            status = run / m["id"] / "status.json"
            completed = read(status).get("completed", 0) if status.exists() else 0
            lines.append(f"| {m['label']} | {completed}/1024 | — | — | — | — |")
    lines += [
        "",
        "Query 的自由文本到标签转换是保守自动评分，完整解释、同义词可能被低估。",
        "Meissa 源码评分另存为敏感性检查；其 any-substring 规则不惩罚多余答案，不能视为标签集合正确率。",
    ]
    (run / "REPORT.md").write_text("\n".join(lines) + "\n")
    if table:
        with (run / "literature_vqa.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
