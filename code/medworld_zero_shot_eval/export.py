"""Export aggregate-only, checked pilot results for review and manuscript use."""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from .common import atomic, digest, read
from .validate import main as validate


def main(run, out):
    validate(run)
    out.mkdir(parents=True, exist_ok=True)
    protocol, inventory = read(run / "protocol.json"), read(run / "models.json")
    audit = read(run / "training_overlap.json")
    metrics = {m["id"]: read(run / m["id"] / "metrics.json") for m in inventory}
    for filename in (
        "table1_diagnosis_pilot.csv",
        "table1_diagnosis_pilot.tex",
        "validation.json",
        "constant_sanity_checks.json",
    ):
        shutil.copy2(run / filename, out / filename)
    atomic(out / "metrics.json", metrics)
    provenance = {
        "run": str(run),
        "protocol_sha256": digest(run / "protocol.json"),
        "source_manifest_sha256": digest(run / "source_manifest.json"),
        "completed": datetime.now().astimezone().isoformat(),
        "gpu": read(run / "status.json")["gpu"],
        "sampling": protocol["sampling"],
        "seed": protocol["seed"],
        "n": protocol["n"],
        "patients": protocol["patients"],
        "semantic_counts": protocol["semantic_counts"],
        "content_counts": protocol["content_counts"],
        "empty_reference_answers": protocol["empty_answers"],
        "training_overlap_patients": audit["sample_train_overlap"],
        "training_overlap_questions": len(audit["overlap_sample_ids"]),
        "training_data_fingerprint": audit["data_fingerprint"],
        "generation": protocol["generation"],
        "image": protocol["image"],
        "scoring": protocol["scoring"],
        "versions": protocol["versions"],
        "model_labels": {m["id"]: m["label"] for m in inventory},
        "inference_seconds": {
            m["id"]: read(run / m["id"] / "status.json")["seconds"] for m in inventory
        },
        "export_source_sha256": digest(__file__),
        "note": "Aggregate artifacts only. Per-question data, image IDs and patient IDs remain in the ignored local run directory.",
    }
    atomic(out / "provenance.json", provenance)
    report = (run / "REPORT.md").read_text()
    report = report.replace(
        "复现、进度和逐题结果均位于本目录；protocol.json 固定样本与预处理，source/ 为执行源码快照。",
        "本目录仅含汇总结果。复现、逐题输出、protocol.json 与执行源码快照位于文末链接的本地运行目录。",
    )
    report += "\n## Diagnosis 的患者级 95% CI\n\n"
    report += (
        "| 模型 | EM (95% CI) | micro-F1 (95% CI) | 推理分钟 |\n|---|---:|---:|---:|\n"
    )
    for model in inventory:
        m = metrics[model["id"]]["diagnosis"]
        cells = []
        for key in ("exact_match", "micro_f1"):
            low, high = m["patient_bootstrap_95ci"][key]
            cells.append(f"{100 * m[key]:.2f} [{100 * low:.2f}, {100 * high:.2f}]")
        report += f"| {model['label']} | {' | '.join(cells)} | {provenance['inference_seconds'][model['id']] / 60:.2f} |\n"
    report += "\n区间来自 1,000 次患者级 bootstrap；未做多重比较或显著性声明。推理时间不含权重校验和服务加载。\n"
    baseline = read(run / "constant_sanity_checks.json")["baselines"]
    report += "\n## 不看图像的常量输出检查\n\n"
    report += "| 固定输出 | Diagnosis EM | Diagnosis micro-F1 |\n|---|---:|---:|\n"
    for name, label in (
        ("always_empty", "始终 []"),
        ("always_no", '["no"]'),
        ("always_yes", '["yes"]'),
    ):
        m = baseline[name]["diagnosis"]
        report += (
            f"| {label} | {100 * m['exact_match']:.2f} | {100 * m['micro_f1']:.2f} |\n"
        )
    report += (
        "\n这些是事后的空集、no、yes 三种常量输出检查，没有拟合参数，也不属于模型实测结果。"
        "它们说明类别比例、空答案与多余标签会影响整体分数，需结合 Verify/Choose/Query 分项解释。\n"
        "\n## 协议与结果使用\n\n"
        "最终四模型统一使用公开词表 JSON 约束。最初 Qwen-0.8B 自由生成轮有 416/1,024 条格式错误或截断，"
        "已单独保留；为降低格式干扰，在其余模型推理前切换协议，并将全部模型按同一规则重跑。"
        "两轮样本、图像、提示词相同。最终结果属于探索性 pilot。\n\n"
        "Diagnosis 的过滤针对当前统一训练目录，各任务 train 均排除；其他历史 checkpoint 的训练来源需另行审计。"
        "公开模型预训练暴露未知。本轮未评价 Ours，也未把当前问答成绩替换进未来预测的 AP/AUROC。\n\n"
        "table1_diagnosis_pilot.tex 是独立的 Table 1 候选块。CSV 为 0–1 原始比例，Markdown/LaTeX 为百分数。"
        "机器可读全部分项、CI、计数在 metrics.json；独立 scikit-learn 复算和哈希检查见 validation.json。\n\n"
        f"逐题输出、输入与执行源码快照：[本地运行目录]({run})。\n"
    )
    (out / "README.md").write_text(report)
    print(f"Exported checked aggregate results to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.out.resolve())
