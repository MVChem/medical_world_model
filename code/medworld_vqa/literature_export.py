"""Validate the frozen rerun, then export aggregate results and protocol comparisons."""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from .common import atomic, digest, read, rows, verify
from .literature_score import published_meissa, subsets, text_labels
from .score import INVALID


def validate(run):
    protocol = verify(run)
    assert read(run / "status.json")["status"] == "complete"
    previous = Path(protocol["prior_run"])
    verify(previous)
    assert digest(previous / "protocol.json") == protocol["prior_protocol_sha256"]
    assert digest(previous / "references.jsonl") == digest(run / "references.jsonl")
    old_inputs, new_inputs = rows(previous / "inputs.jsonl"), rows(run / "inputs.jsonl")
    assert len(old_inputs) == len(new_inputs) == protocol["n"]
    for old, new in zip(old_inputs, new_inputs, strict=True):
        assert (old["id"], old["question"]) == (new["id"], new["question"])
        assert digest(old["image"]) == digest(new["image"])
    refs = {r["id"]: r for r in rows(run / "references.jsonl")}
    vocabulary = read(run / "vocabulary.json")
    binarizer = MultiLabelBinarizer(classes=[*vocabulary, INVALID]).fit([[]])
    excluded = set(read(run / "training_overlap.json")["overlap_sample_ids"])
    validation = {}
    for model in read(run / "models.json"):
        name = model["id"]
        raw = rows(run / name / "predictions.jsonl")
        pred = {p["id"]: p for p in raw}
        assert len(raw) == len(pred) == len(refs) and set(pred) == set(refs)
        assert all(p["ok"] for p in raw)
        records = rows(run / name / "scored_predictions.jsonl")
        assert len(records) == len(refs) and {r["id"] for r in records} == set(refs)
        metrics = read(run / name / "metrics.json")
        assert metrics["predictions_sha256"] == digest(run / name / "predictions.jsonl")
        assert metrics["protocol_sha256"] == digest(run / "protocol.json")
        assert read(run / name / "status.json")["completed"] == len(refs)
        for record in records:
            ref, prediction = refs[record["id"]], pred[record["id"]]
            assert set(record["target"]) == set(ref["answer"])
            parsed = (
                {INVALID}
                if prediction["finish_reason"] == "length"
                else text_labels(
                    prediction["text"], ref["semantic_type"], set(vocabulary)
                )[0]
            )
            assert set(record["predicted"]) == parsed
        validation[name] = {}
        for subset, selected in subsets(records, excluded).items():
            for kind in ("all", "verify", "choose", "query"):
                selected_kind = (
                    selected
                    if kind == "all"
                    else [r for r in selected if r["semantic_type"] == kind]
                )
                yt = binarizer.transform([r["target"] for r in selected_kind])
                yp = binarizer.transform([r["predicted"] for r in selected_kind])
                exact, f1 = (
                    accuracy_score(yt, yp),
                    f1_score(yt, yp, average="micro", zero_division=0),
                )
                saved = metrics["subsets"][subset]
                if kind != "all":
                    saved = saved["by_semantic"][kind]
                assert saved["n"] == len(selected_kind)
                np.testing.assert_allclose(
                    [exact, f1], [saved["exact_match"], saved["micro_f1"]], atol=1e-12
                )
                validation[name][f"{subset}/{kind}"] = {
                    "n": len(selected_kind),
                    "sklearn_accuracy": exact,
                    "sklearn_micro_f1": f1,
                }
    result = {
        "status": "passed",
        "models": validation,
        "protocol_sha256": digest(run / "protocol.json"),
        "validation_source_sha256": digest(__file__),
        "checks": [
            "frozen source/artifact hashes",
            "identical prior sample, questions, references and image bytes",
            "complete unique successful response coverage",
            "raw-to-parsed replay",
            "independent sklearn aggregation on 12 groups per model",
        ],
        "limitation": "Label parsing is replayed, not independently adjudicated for semantic correctness.",
    }
    atomic(run / "literature_validation.json", result)
    return result


def main(run, out):
    validation = validate(run)
    out.mkdir(parents=True, exist_ok=True)
    protocol = read(run / "protocol.json")
    models = read(run / "models.json")
    metrics = {m["id"]: read(run / m["id"] / "metrics.json") for m in models}
    atomic(out / "metrics.json", metrics)
    atomic(out / "validation.json", validation)
    shutil.copy2(run / "literature_vqa.csv", out / "literature_vqa.csv")
    shutil.copy2(run / "literature_sources.json", out / "literature_sources.json")
    check = published_meissa(run)
    refs = rows(run / "references.jsonl")
    all_labels = ", ".join(read(run / "vocabulary.json"))
    sensitivity = {
        "rule": "Unmodified Meissa any normalized reference-label substring in prediction, any one label is enough; empty reference list always false",
        "examples": {
            "negated_positive_label_is_marked_correct": check(
                "No pleural effusion.", ["pleural effusion"]
            ),
            "missing_one_of_two_labels_is_marked_correct": check(
                "Pneumonia.", ["pneumonia", "pleural effusion"]
            ),
            "empty_answer_cannot_be_correct": check("None", []),
        },
        "all_110_labels_without_images_accuracy": sum(
            check(all_labels, r["answer"]) for r in refs
        )
        / len(refs),
        "nonempty_reference_fraction": sum(bool(r["answer"]) for r in refs) / len(refs),
    }
    atomic(out / "scoring_sensitivity.json", sensitivity)
    provenance = {
        "run": str(run),
        "protocol_sha256": digest(run / "protocol.json"),
        "source_manifest_sha256": digest(run / "source_manifest.json"),
        "export_source_sha256": digest(__file__),
        "exported": datetime.now().astimezone().isoformat(),
        "gpu": read(run / "status.json")["gpu"],
        "n": protocol["n"],
        "patients": protocol["patients"],
        "semantic_counts": protocol["semantic_counts"],
        "prompt_style": protocol["prompt_style"],
        "generation": protocol["generation"],
        "image": protocol["image"],
        "scoring": protocol["scoring"],
        "training": protocol["training"],
        "prior_protocol_sha256": protocol["prior_protocol_sha256"],
        "inference_seconds": {
            m["id"]: read(run / m["id"] / "status.json")["seconds"] for m in models
        },
    }
    atomic(out / "provenance.json", provenance)
    lines = [
        (run / "REPORT.md").read_text(),
        "\n## 与上一轮的同题比较\n",
        "前后都是同一批 1,024 题。前轮为词表 JSON，复测为 LLaVA 短答提示；输出解析和 token 上限也改变，因此变化不能单独归因于提示词。\n",
        "| 模型 | Verify：旧 → 新 | Choose：旧 → 新 | Query label μF1：旧 → 新 | 新轮截断 |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in models:
        name = m["id"]
        old = read(Path(protocol["prior_run"]) / name / "metrics.json")["by_semantic"]
        new = metrics[name]["subsets"]["overall"]["by_semantic"]
        cells = [
            f"{100 * old[t][k]:.2f} → {100 * new[t][k]:.2f}"
            for t, k in (
                ("verify", "exact_match"),
                ("choose", "exact_match"),
                ("query", "micro_f1"),
            )
        ]
        lines.append(
            f"| {m['label']} | {' | '.join(cells)} | {metrics[name]['finish_reasons'].get('length', 0)} |"
        )
    lines += [
        "\n## 患者级 95% CI\n",
        "1,000 次患者级 bootstrap，按完整 1,024 题中的对应题型重采样。\n",
        "| 模型 | Verify Acc | Choose Acc | Query label μF1 |",
        "|---|---:|---:|---:|",
    ]
    for m in models:
        typed = metrics[m["id"]]["subsets"]["overall"]["by_semantic"]
        cells = []
        for kind, key in (
            ("verify", "exact_match"),
            ("choose", "exact_match"),
            ("query", "micro_f1"),
        ):
            low, high = typed[kind]["patient_bootstrap_95ci"][key]
            cells.append(
                f"{100 * typed[kind][key]:.2f} [{100 * low:.2f}, {100 * high:.2f}]"
            )
        lines.append(f"| {m['label']} | {' | '.join(cells)} |")
    lines += [
        "\n## 当前项目 Diagnosis 子集\n",
        "沿用已冻结的 884 题 / 382 患者，排除当前统一训练患者及 plane/gender；含器械和技术质量，属于当前图像任务。\n",
        "| 模型 | Verify Acc | Choose Acc | Query label μF1 |",
        "|---|---:|---:|---:|",
    ]
    tex = [
        "% Exploratory current-image CXR-VQA; local conservative text adapter, not future forecasting.",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & Verify Acc & Choose Acc & Query $\mu$F1 \\",
        r"\midrule",
    ]
    for m in models:
        typed = metrics[m["id"]]["subsets"]["diagnosis"]["by_semantic"]
        cells = [
            f"{100 * typed[t][k]:.2f}"
            for t, k in (
                ("verify", "exact_match"),
                ("choose", "exact_match"),
                ("query", "micro_f1"),
            )
        ]
        lines.append(f"| {m['label']} | {' | '.join(cells)} |")
        tex.append(m["label"] + " & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / "table1_current_vqa_candidate.tex").write_text("\n".join(tex) + "\n")
    lines += [
        "\n## 同一批输出换用公开 Meissa 评分代码\n",
        "这是评分敏感性检查，不是运行 Meissa agent 或复现其模型结果。\n",
        "| 模型 | Any-substring accuracy |",
        "|---|---:|",
    ]
    for m in models:
        v = metrics[m["id"]]["subsets"]["overall"][
            "published_meissa_any_substring_accuracy"
        ]
        lines.append(f"| {m['label']} | {100 * v:.2f} |")
    lines += [
        f"\n不看图像、每题输出全部 110 标签，同一评分器即可得到 **{100 * sensitivity['all_110_labels_without_images_accuracy']:.2f}%**。这项分数不作为主要结果。",
        "\n## 解析与使用边界\n",
        "Verify 接受句首独立 yes/no；Choose/Query 接受规范标签文本列表、JSON 列表、标点和项目符号，female/male 映射至 f/m。none/no/nothing/no abnormalities/no abnormality 的独立回答视为空集。",
        "其他同义词和完整解释不做语义匹配；非规范片段保留一个无效标签 FP，规范列表项保留部分分数；截断、空响应和失败整题计错。无法解析不等于正确空集。",
        "因此 Query label μF1 是保守自动解析结果，不能直接等同于临床答案质量或 AOR 官方实现。论文主表正式使用前，需要验证集确定更可靠的答案规范化或任务微调，并重新评测。",
        "[指标解释、文献核对与协议差异](../../research_notes/0916_mimic_cxr_vqa_literature.md)。",
        f"\n完整本地输入、逐题输出、协议与源码快照：[运行目录]({run})。本目录只导出汇总，不包含患者标识。",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"validation": "passed", "export": str(out)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.out.resolve())
