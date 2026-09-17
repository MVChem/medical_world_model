"""A single live entry point for the matched Table 1 / Table 2 results."""
import csv
import io
from pathlib import Path
from .common import atomic, read


def render(run):
    state = read(run / "status.json") if (run / "status.json").exists() else {}
    lines = ["# 与融合模型匹配的 zero-shot 评测", "", f"队列状态：{state.get('phase', 'prepared')}；仅使用一张 GPU。", "",
        "原始公开 BF16 checkpoint；不训练、不加载项目 adapter。旧 09-11 分数不混入本表。", "",
        "Table 1：同一 297 对／94 患者；正向、回溯分别汇总。Table 2：分类 353，报告 507。",
        "输入像素、样本 ID、源报告截断和实际时间与当前训练协议对齐；报告统一最多 384 tokens。", "",
        "Qwen 使用训练所用 256² 原生视觉处理；MedGemma 使用原生处理器，视觉计算量不同。", "",
        "| 模型 | 推理进度 | 临床评分 |", "|---|---:|---|"]
    metrics = {}
    for model in read(run / "models.json"):
        name = model["id"]
        path = run / name / "test/status.json"
        status = read(path) if path.exists() else {}
        tasks = status.get("tasks", {})
        n = sum(v["completed"] for v in tasks.values())
        expected = 594 * 7 + 507 + 353 * 13
        stage = state.get("models", {}).get(name, {})
        lines.append(f"| {model['label']} | {n}/{expected} ({status.get('status', 'pending')}) | {stage.get('clinical', 'pending')} |")
        path = run / name / "test/metrics.json"
        metrics[name] = read(path) if path.exists() else {}
    def value(x):
        return "—" if x is None else f"{100*x:.2f}"
    tables = {}
    for panel, title, columns in (
        ("table1_forward", "Table 1：未来预测", ("ap", "auroc", "transition_f1", "radgraph_f1", "green", "brier", "ece")),
        ("table1_backward", "补充：回溯预测", ("ap", "auroc", "transition_f1", "radgraph_f1", "green", "brier", "ece")),
        ("table2", "Table 2：当前状态", ("classification_auroc", "classification_ap", "report_radgraph_f1", "report_chexbert_f1"))):
        lines += ["", "## " + title, "", "数值 ×100；AP/AUROC/F1/GREEN 越高越好，Brier/ECE 越低越好。", "",
                  "| 模型 | " + " | ".join(columns) + " |", "|---|" + "---:|" * len(columns)]
        table = []
        for model in read(run / "models.json"):
            m = metrics[model["id"]]
            if panel == "table2":
                scores = [m.get("table2_classification", {}).get(k) for k in ("auroc", "ap")]
                scores += [m.get("table2_report", {}).get(k) for k in ("radgraph_f1", "chexbert_f1")]
            else:
                scores = [m.get(panel, {}).get(k) for k in columns]
            lines.append("| " + model["label"] + " | " + " | ".join(value(x) for x in scores) + " |")
            table.append(dict(model=model["id"], **dict(zip(columns, scores))))
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["model", *columns])
        writer.writeheader(); writer.writerows(table)
        (run / (panel + ".csv")).write_text(output.getvalue())
        tables[panel] = table
    lines += ["", "— 表示尚未完成或没有合格参考，不是零分。Direction F1 缺经核验标签；官方 VQA、MS-CXR grounding 待数据。",
              "原生文本 VLM 的 segmentation、SR 为 N/A；训练 heads 的历史结果不属于 zero-shot。",
              "概率为 Yes/No 原始条件似然；参考掩码固定，不做测试集调参。不同模型公开预训练数据交叠未知。", "",
              "协议：[protocol.json](protocol.json)；完整状态：[status.json](status.json)；逐模型 test/metrics.json 保存逐类指标。",
              "生成失败会阻止该完整面板计分，不筛选模型各自成功的子集。"]
    (run / "REPORT.md").write_text("\n".join(lines) + "\n")
    atomic(run / "tables.json", tables)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--run", type=Path, required=True)
    render(p.parse_args().run.resolve())
