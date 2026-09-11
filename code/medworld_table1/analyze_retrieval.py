"""Post hoc small-gallery retrieval audit using cached test labels only."""
import collections
import json
import os
from pathlib import Path

import numpy as np

from metrics import candidate_pools, retrieval

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/pilot_20260908_8h'
OUT = RUN / 'analysis_20260909'
os.umask(0o077)
OUT.mkdir(exist_ok=True, mode=0o700)


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


cohort = rows(ROOT / 'data/pilot/test.jsonl')
names = ['copy', 'direct', 'matched', 'ours']
labels = {}
for name in names:
    directory = RUN / name / ('evaluation' if name != 'copy' else '')
    assert [r['id'] for r in rows(directory / 'predictions.jsonl')] == [r['id'] for r in cohort]
    labels[name] = read(directory / 'chexbert_labels.json')
    assert labels[name]['current'] == labels['copy']['current']
    assert labels[name]['target'] == labels['copy']['target']

current, target = np.asarray(labels['copy']['current']), np.asarray(labels['copy']['target'])
result = dict(protocol='Post hoc; original exact matching, fractional tie credit, shared pools; only candidate count changes.',
              primary_seed=42, sensitivity_seeds=list(range(42, 52)), n_total=len(cohort), candidate_counts={})
for k in [2, 4, 8, 16, 32]:
    pools = candidate_pools(cohort, current, target, seed=42, negatives=k-1)
    for pool in pools:
        assert len(pool['candidates']) == k
        assert len({cohort[i]['patient'] for i in pool['candidates']}) == k
        assert any(pool['mask'])
    output = dict(queries=len(pools), patients=len({cohort[p['query']]['patient'] for p in pools}),
                  coverage=len(pools)/len(cohort), random_ranking_reference=1/k,
                  mask_field_counts=dict(collections.Counter(sum(p['mask']) for p in pools)),
                  scores={name: retrieval(labels[name]['predicted'], target, pools)['future_r1'] for name in names})
    if pools:
        counts = [len(np.unique(target[np.asarray(p['candidates'])][:, np.asarray(p['mask'])], axis=0)) for p in pools]
        output['all_candidates_same_schema_queries'] = sum(n == 1 for n in counts)
        output['target_schema_oracle_r1'] = retrieval(target, target, pools)['future_r1']
        sensitivity = {name: [] for name in names}
        for seed in range(42, 52):
            selected = pools if seed == 42 else candidate_pools(cohort, current, target, seed=seed, negatives=k-1)
            assert [p['query'] for p in selected] == [p['query'] for p in pools]
            for name in names:
                sensitivity[name].append(retrieval(labels[name]['predicted'], target, selected)['future_r1'])
        output['candidate_seed_sensitivity'] = {name: dict(mean=float(np.mean(values)), min=min(values), max=max(values)) for name, values in sensitivity.items()}
    result['candidate_counts'][str(k)] = output

(OUT / 'retrieval_supplement.json').write_text(json.dumps(result, indent=2)+'\n')
lines = [
    '# Future R@1 补充诊断（2026-09-09）', '',
    '沿用原实现的 horizon、view、当前阳性数量分组、真值可评估字段匹配，以及不同患者负例和并列分数规则；仅改变候选总数。所有模型共享候选池，主结果 seed=42。这里只读取已有测试预测和标签，无需重新训练或生成报告。', '',
    '| 候选数 | 可纳入检查对 | 患者数 | 覆盖率 | 随机排序参考 | Copy Current | 直接 Qwen | 冻结 Stage-1 | MedWorld |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
]
for k, v in result['candidate_counts'].items():
    scores = ['' if v['scores'][n] is None else f'{v["scores"][n]:.4f}' for n in names]
    lines.append(f'| {k} | {v["queries"]} | {v["patients"]} | {v["coverage"]:.1%} | {v["random_ranking_reference"]:.4f} | '+' | '.join(scores)+' |')
lines.extend(['',
    '原定 32 候选结果仍不可用，主表不被这些事后小候选池结果替换。不同候选数对应不同纳入子集，不能把它们当作同一人群的可直接比较曲线。随机排序参考为 1/K；这里只比较点估计，未检验模型是否显著高于或低于随机。', '',
    '8 候选纳入 120 对、53 位患者，其中 25 对的全部候选在可评估疾病字段上完全相同，因此即使正确预测未来疾病标签也无法区分这些候选。六疾病 schema 的区分能力与候选池规模都需要检查。', '',
    '| 方法 | 4 候选：10 个候选池 seed 均值 [最小, 最大] | 8 候选：10 个候选池 seed 均值 [最小, 最大] |',
    '|---|---:|---:|',
])
for name in names:
    values = []
    for k in ['4', '8']:
        v = result['candidate_counts'][k]['candidate_seed_sensitivity'][name]
        values.append(f'{v["mean"]:.4f} [{v["min"]:.4f}, {v["max"]:.4f}]')
    lines.append('| '+name+' | '+' | '.join(values)+' |')
lines.extend(['', '该 seed 范围只是负例抽样敏感性，不是患者置信区间，也不是多次训练结果。', '',
              '如要补齐原定主表，应先检查扩大的独立测试候选库能否在相同匹配规则下满足 32 候选；若仍不足，需要在验证集上重新确定匹配规则或更丰富的评价 schema，并对所有方法统一重评。增加库中真实候选的报告标签不需要重新训练预测器。', '',
              '可复现脚本：[analyze_retrieval.py](../../../analyze_retrieval.py)；完整结果：[retrieval_supplement.json](retrieval_supplement.json)。', ''])
(OUT / 'retrieval_supplement.md').write_text('\n'.join(lines))
print('\n'.join(lines))
