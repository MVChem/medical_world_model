"""Refresh Table 1/2 experiment previews from attributed completed artifacts."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = [('qwen08b', 'Qwen3.5-0.8B'), ('qwen4b', 'Qwen3.5-4B'),
          ('qwen9b', 'Qwen3.5-9B'), ('qwen27b_fp8', 'Qwen3.5-27B-FP8'),
          ('medgemma4b', 'MedGemma-1.5-4B'), ('medgemma27b', 'MedGemma-v1-27B')]
RUNS = {'dense': 'code/medworld_dense_baselines/runs/expanded_overnight_20260913',
        'forecast': 'code/medworld_table1/runs/overnight_20260913',
        'joint': 'code/medworld_joint/runs/overnight_20260913'}
T1 = ['Method', 'Protocol', 'Status', 'AP', 'AUROC', 'Transition F1', 'Direction F1', 'RadGraph F1', 'GREEN', 'Brier', 'ECE']
T2 = ['Method', 'Protocol', 'Status', 'Classification AUROC', 'Classification AP', 'VQA Accuracy', 'VQA Micro F1',
      'Report RadGraph F1', 'Report CheXbert F1', 'Anatomy mIoU', 'Anatomy Acc50', 'Dice pseudo', 'Dice human', 'PSNR x4', 'SSIM x4']


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def fmt(value):
    if value is None:
        return 'pending'
    return f'{value:.4f}' if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)


def table(columns, rows):
    return '\n'.join(['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |'] +
                     ['| ' + ' | '.join(fmt(row.get(key)).replace('|', '/') for key in columns) + ' |' for row in rows])


def report(root, out):
    provenance, warnings = [], []
    def tracked(path):
        data = read(path)
        if data is not None:
            provenance.append(dict(path=str(Path(path).resolve()), sha256=sha(path)))
        return data
    queue_jobs, streams = [], []
    for stream, rel in RUNS.items():
        run = root / rel
        state = tracked(run / 'scheduler/status.json') or {}
        jobs = tracked(run / 'scheduler/queue.json') or []
        queue_jobs.extend(jobs)
        streams.append(dict(stream=stream, phase=state.get('phase', 'preparing'), counts=state.get('counts', {}), run=str(run)))
    verified = set()
    for job in queue_jobs:
        if job['status'] != 'complete':
            continue
        for path, expected in job.get('artifact_sha256', {}).items():
            if Path(path).exists() and sha(path) == expected:
                verified.add(str(Path(path).resolve()))
            else:
                warnings.append(f'Completion digest mismatch: {job["id"]} / {path}')
    def new_metrics(path):
        if str(Path(path).resolve()) not in verified:
            return None
        return tracked(path)
    t1, t2, direction = [], [], []
    raw = root / 'code/medworld_baselines/runs/raw_models_20260911'
    prior_dense = root / 'code/medworld_dense_baselines/runs/dense_20260912'
    for mid, label in MODELS:
        all_metrics = tracked(raw / mid / 'test/metrics.json') or {}
        forecast = all_metrics.get('table1', {})
        row = dict(Method=label + ' (zero-shot)', Protocol='F0: 297 future pairs; prior-EHR/source inputs', Status='historical complete')
        if forecast.get('n') == 297:
            for dest, src in [('AP', 'ap'), ('AUROC', 'auroc'), ('Transition F1', 'transition_f1'),
                              ('RadGraph F1', 'radgraph_f1'), ('Brier', 'brier'), ('ECE', 'ece')]:
                row[dest] = forecast.get(src)
            green = tracked(raw / mid / 'test/green_metrics.json') or {}
            if green.get('status') == 'complete' and green.get('n') == 297:
                row['GREEN'] = green.get('mean')
        else:
            row['Status'] = 'missing or cohort mismatch'
        t1.append(row)
        cls, rep = all_metrics.get('table2_classification', {}), all_metrics.get('table2_report', {})
        t2.append(dict(Method=label + ' (zero-shot)', Protocol='C0: current image; classification/report cohorts differ',
                       Status='historical complete' if all_metrics else 'pending',
                       **{'Classification AUROC': cls.get('auroc'), 'Classification AP': cls.get('ap'),
                          'Report RadGraph F1': rep.get('radgraph_f1'), 'Report CheXbert F1': rep.get('chexbert_f1')}))
        d = tracked(prior_dense / mid / 'direction_metrics.json') or {}
        direction.append(dict(Model=label, **{'Direction F1': d.get('macro_f1') if d.get('complete') else None,
                         'Invalid outputs': d.get('invalid_outputs'), 'Pairs': 82}))
    # Task generators can supply exact new metric mappings; every artifact is
    # additionally gated by the scheduler's successful worker receipt + digest.
    for stream, rel in RUNS.items():
        mapping = read(root / rel / 'table_export.json', [])
        for item in mapping:
            columns = T1 if item['table'] == 1 else T2
            target = t1 if item['table'] == 1 else t2
            row = dict(Method=item['method'], Protocol=item['protocol'], Status='pending')
            available, expected = 0, len(item['artifacts'])
            for artifact in item['artifacts']:
                path = Path(artifact['path'])
                if not path.is_absolute():
                    path = root / rel / path
                data = new_metrics(path)
                if data is None:
                    continue
                available += 1
                for dest, key in artifact.get('metrics', {}).items():
                    value = data
                    for part in key.split('.'):
                        value = value.get(part) if isinstance(value, dict) else None
                    if dest not in columns:
                        raise ValueError(f'Unknown metric column {dest}')
                    row[dest] = value
            if available:
                row['Status'] = 'complete' if available == expected else f'{available}/{expected} tasks complete'
            target.append(row)
    out.mkdir(parents=True, exist_ok=True)
    for name, columns, rows in [('table1', T1, t1), ('table2', T2, t2)]:
        with (out / f'{name}.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        write(out / f'{name}.md', table(columns, rows) + '\n')
        # Self-contained longtable body for review/export; does not replace the
        # paper's protocol-specific main tables while experiments are running.
        def escape(value):
            return fmt(value).replace('\\', '\\textbackslash{}').replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')
        tex = ['% Generated from completed metrics; see provenance.json and REPORT.md.',
               '\\begin{tabular}{' + 'l' * len(columns) + '}',
               ' & '.join(escape(c) for c in columns) + r' \\', r'\hline']
        tex += [' & '.join(escape(row.get(c)) for c in columns) + r' \\' for row in rows]
        tex.append('\\end{tabular}')
        write(out / f'{name}.tex', '\n'.join(tex) + '\n')
    stamp = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds')
    lines = ['# 09-13 overnight Table 1 / Table 2', '', f'Updated: {stamp} (Asia/Shanghai).',
             '', '最多 5 张 GPU：扩展冻结对照 0/1，未来预测 3，联合优化 2/4。07:45 停止训练并保存；09-14 08:00 强制释放本轮 GPU 进程。',
             '', '新结果只接收本轮调度器确认成功、SHA256 匹配的完整任务指标；未完成项目保持 pending。历史零样本数值明确标记 historical，未经本轮重跑。',
             '', '## Queue', '', table(['stream', 'phase', 'counts', 'run'], streams),
             '', 'GPU 实时采样：[利用率记录](gpu_utilization.json)。每 30 秒记录计算利用率、显存和功耗，区分数据准备阶段与正式训练。',
             '', '## Table 1: future prediction', '', table(T1, t1),
             '', 'Direction 在原 297 对测试集上尚无合格标签，主表保持 pending。下面是已有独立 82 对测试集的结果，输入仅源图、源报告和时间段；不可合并称为同一测试协议。',
             '', table(['Model', 'Direction F1', 'Invalid outputs', 'Pairs'], direction),
             '', '## Table 2: downstream tasks', '', table(T2, t2),
             '', 'F0：历史未来预测，297 对；C0：历史当前分类/报告。D1：本轮 18,708 train、249 validation、447 test、Montgomery 138 人工双肺测试，20 epochs。',
             '', '冻结多层实验仅运行原生视觉塔，第 5–8 个 slots。Image-only 是共享无条件 decoder，并非完整 VLM tokens 基线；真实无 slot 压缩的对照在联合优化中单独命名 full tokens。',
             '', '联合优化 J1 使用在线 VLM、8 个可训练读出和 LoRA；分割与 SR 读取全部 8 个 slots。这是本轮记录的任务路由变体。27B 冻结行不代表完成 27B 联合优化。',
             '', '分割 pseudo Dice 衡量 CXAS 三器官伪标签一致性；human Dice 衡量 Montgomery 人工双肺。SR 所有模型输入仅来自同一 ×4 LR，HR 只作目标。不同协议和训练预算分行展示。',
             '', 'MedGemma 4B/27B 的冻结视觉塔权重相同，两个冻结行不能证明语言模型规模收益。官方 VQA 缺少本轮正式结果；派生疾病问答没有填入 VQA 列。',
             '', '历史冻结 slots/shuffled 全结果：[09-13 既有报告](../../code/medworld_dense_baselines/runs/frozen_slots_shuffled_remaining_20260913/REPORT.md)。',
             '', '导出：[Table 1 CSV](table1.csv) · [Table 2 CSV](table2.csv) · [来源指纹](provenance.json)。']
    if warnings:
        lines += ['', '## Provenance checks', ''] + warnings
    write(out / 'REPORT.md', '\n'.join(lines) + '\n')
    write(out / 'provenance.json', json.dumps(dict(updated=stamp, sources=provenance, warnings=warnings), ensure_ascii=False, indent=2) + '\n')
    return dict(table1_rows=len(t1), table2_rows=len(t2), warnings=warnings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=ROOT)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    print(json.dumps(report(args.project.resolve(), args.out or args.project / 'results/overnight_20260913')))
