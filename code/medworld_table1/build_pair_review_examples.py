"""Render six local CXR/IV review examples from existing records; no model use."""
import base64
import hashlib
import html
import json
from collections import defaultdict
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from audit_pair_quality import table, rows, time, match_ids

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
OUT = PROJECT / 'results/mimic_linked_examples_20260909'
FINDINGS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion', 'Pneumothorax']
STATE = {-2: 'not_mentioned', -1: 'uncertain', 0: 'absent', 1: 'present'}


def from_existing(row, case, title, note, source_summary, target_summary):
    packet = row['mimic_cxr_transition']
    context = row.get('retrospective_context_for_audit_only', {})
    endpoints = []
    for side in ['current_state', 'future_state']:
        s = packet[side]
        endpoints.append(dict(timestamp=s['timestamp'], study_id=s['study_id'],
            image=s['image'], report=s.get('text') or s.get('text_for_evaluation_only'),
            labels={k: (s.get('structured_findings') or s.get('structured_findings_for_evaluation_only'))[k] for k in FINDINGS}))
    return dict(case=case, title=title, review_note=note, source_summary=source_summary, target_summary=target_summary,
        origin='既有 appendix / MIMIC_example 人工挑选的展示样本；不是随机样本',
        selection='Existing representative_linked_output_10 example selected for an explicit discussion category.',
        source_databases=['MIMIC-CXR-JPG 2.0.0', 'MIMIC-IV 3.1'], subject_id=row['subject_id'],
        transition_id=row['transition_id'], split=packet['split'],
        gap_hours=(time(endpoints[1]['timestamp'])-time(endpoints[0]['timestamp'])).total_seconds()/3600,
        source=endpoints[0], target=endpoints[1], linkage_status=row['linkage_status'],
        retrospective_clinical_context=context)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    existing_path = ROOT.parent / 'MIMIC_example/representative_linked_output_10/linked_transitions.jsonl'
    existing = rows(existing_path)
    cases = [
        from_existing(existing[0], 'A', '同次住院，报告描述明显变化',
            '两端同一 SICU stay；报告从无明确肺水肿转为肺水肿、积液。当前体位元数据缺失，不能算入“体位两端已知一致”的严格子集。区间存在通气/用药记录，不能据此认定变化的原因。',
            '报告：无明确肺水肿、积液；有支持装置。', '报告：肺水肿、小量右侧积液；支持装置描述改变。'),
        from_existing(existing[1], 'B', '稳定随访，仍有真实临床记录',
            '同次 SICU stay、相同 portable AP 和 Erect；两份报告均描述积液/肺不张且变化不大。区间有 Furosemide 等记录；影像稳定不代表没有治疗。该类型应保留为 persistence 对照。',
            '报告：双肺底积液和压迫性肺不张。', '报告：积液、肺不张仍在，整体变化很小。'),
        from_existing(existing[4], 'C', '置管相关随访与标签可判断性变化',
            '同次住院，转科记录为 CCU → Vascular；未来报告明确写已放置右侧 pigtail。当前报告称 CT 所见气胸在本片难以辨认，弱标签 uncertain → present，不应直接当作新发气胸。所选 ICU 细表无对应事件记录，不代表没有干预。',
            '报告：肺水肿；CT 所见气胸在本片不易辨认。', '报告：右侧 pigtail 已放置；残留小气胸，肺水肿改善。'),
        from_existing(existing[3], 'D', '未来状态不确定，不能硬标成消失',
            '同住院普通病房，无匹配 ICU stay。未来报告受皮下气肿/肺大疱遮挡，不能确定小气胸；CheXpert present → uncertain。当前报告 findings 与 impression 对气胸增减的措辞也不一致，值得人工核验。',
            '报告：仍有小气胸；findings 与 impression 的增减措辞有分歧。', '报告：受遮挡影响，疑似小气胸，无法自信判断。'),
    ]
    observations = {r['id']: r for r in rows(ROOT / 'data/pilot/observations.jsonl')}
    train = rows(ROOT / 'data/pilot/train.jsonl')
    patients = {r['patient'] for r in train}
    adm, stays = defaultdict(list), defaultdict(list)
    for r in table(Path('/home/data1/data/MIMIC/mimic-iv-3.1/hosp/admissions.csv.gz')):
        if r['subject_id'] in patients:
            start = min(x for x in [time(r['edregtime']), time(r['admittime'])] if x is not None)
            adm[r['subject_id']].append((r['hadm_id'], start, time(r['dischtime'])))
    for r in table(Path('/home/data1/data/MIMIC/mimic-iv-3.1/icu/icustays.csv.gz')):
        if r['subject_id'] in patients:
            stays[r['subject_id']].append((r['stay_id'], time(r['intime']), time(r['outtime'])))
    wanted = {Path(observations[r[side]]['image']).stem for r in train for side in ['source', 'target']}
    metadata = {r['dicom_id']: r for r in table(Path('/home/data1/data/MIMIC/MIMIC_CXR/mimic-cxr-2.0.0-metadata.csv')) if r['dicom_id'] in wanted}
    selected = {}
    for pair in train:
        a, b = [observations[pair[side]] for side in ['source', 'target']]
        links = [match_ids(time(s['timestamp']), adm[pair['patient']]) for s in [a, b]]
        if any(len(s) != 1 for s in links):
            continue
        same = links[0] == links[1]
        m = [metadata[Path(s['image']).stem] for s in [a, b]]
        proc = [(r['PerformedProcedureStepDescription'] or '').strip().upper() for r in m]
        orient = [(r['PatientOrientationCodeSequence_CodeMeaning'] or '').strip().upper() for r in m]
        kind = None
        if not same and 168 < pair['realized_gap_hours'] <= 720 and 'E' not in selected:
            kind = 'E'
        if same and pair['realized_gap_hours'] <= 72 and all(proc) and all(orient) and orient[0] != orient[1] and 'F' not in selected:
            kind = 'F'
        if kind:
            selected[kind] = (pair, a, b, links, m)
        if len(selected) == 2:
            break
    assert set(selected) == {'E', 'F'}
    for kind in ['E', 'F']:
        pair, a, b, links, metadata_rows = selected[kind]
        endpoints = []
        for s, m in zip([a, b], metadata_rows):
            endpoints.append(dict(timestamp=s['timestamp'], study_id=s['study'],
                image=dict(path=s['image'], view=s['view'], procedure=m['PerformedProcedureStepDescription'],
                           orientation=m['PatientOrientationCodeSequence_CodeMeaning']),
                report={'Findings / Impression': s['report']}, labels={name: STATE[v] for name, v in zip(FINDINGS, s['labels'])}))
        cases.append(dict(case=kind,
            title='同患者但跨住院：不能混入同次住院预测' if kind == 'E' else '相同 AP/PA，但体位不一致',
            review_note='原 pilot 允许这种配对；两端各自能连接 IV，但属于不同 hadm_id。应从同次住院主任务排除，或单列跨就诊随访任务。' if kind == 'E' else '同次住院且 6–72 小时，但体位变化。图像差异可能同时包含采集因素和疾病变化，须看原片与报告；本例未提取区间治疗事件。',
            source_summary='原始报告及标签见下方。', target_summary='原始报告及标签见下方。',
            origin='现有 pilot 训练缓存中的真实 pair', selection='First qualifying pair in the existing deterministic hash order, based only on admission/gap/acquisition criteria; no label or model-performance selection.',
            source_databases=['MIMIC-CXR-JPG 2.0.0', 'MIMIC-IV 3.1'], subject_id=pair['patient'], transition_id=pair['id'], split='train',
            gap_hours=pair['realized_gap_hours'], source=endpoints[0], target=endpoints[1],
            linkage_status='different_admissions' if kind == 'E' else 'unique_common_admission',
            retrospective_clinical_context=dict(source_hadm_ids=sorted(links[0]), target_hadm_ids=sorted(links[1]),
                source_stay_ids=sorted(match_ids(time(a['timestamp']), stays[pair['patient']])),
                target_stay_ids=sorted(match_ids(time(b['timestamp']), stays[pair['patient']])),
                interval_events='Not extracted for this example; no inference about treatment absence.')))
    for case in cases:
        assert 6 <= case['gap_hours'] <= 720
        assert case['source']['image']['view'] == case['target']['image']['view']
        for side in ['source', 'target']:
            path = Path(case[side]['image']['path'])
            assert path.is_file()
            case[side]['image']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / 'examples.json').write_text(json.dumps(dict(
        note='Local review examples, deliberately chosen categories; not random prevalence estimates or clinically adjudicated transitions. No model training/inference. Clinical interval records and future observations are retrospective context, not current-only inputs.',
        cases=cases), indent=2, ensure_ascii=False)+'\n')
    esc = lambda value: html.escape(str(value))
    cards = []
    for case in cases:
        side_html = []
        for side, heading in [('source', '当前观测 · t0'), ('target', f'真实随访 · +{case["gap_hours"]:.1f} 小时')]:
            s = case[side]
            img = s['image']
            encoded = base64.b64encode(Path(img['path']).read_bytes()).decode()
            report = ''.join(f'<h5>{esc(k.upper())}</h5><p>{esc(v)}</p>' for k, v in s['report'].items() if v)
            side_html.append(f'<div class="observation {side}"><h3>{esc(heading)}</h3><img alt="{esc(case["case"]+" "+side)}" src="data:image/jpeg;base64,{encoded}"><div class="metadata">{esc(img["view"])} · {esc(img.get("procedure") or "检查类型缺失")}<br>体位：{esc(img.get("orientation") or "缺失")} · {esc(s["timestamp"])}</div><p>{esc(case[side+"_summary"])}</p><details><summary>展开原始 Findings / Impression</summary>{report}</details></div>')
        ctx = case['retrospective_clinical_context']
        if ctx.get('admission'):
            admission_text = '同一 hadm_id：'+str(ctx['admission']['hadm_id'])
            locations = ' → '.join(str(ctx.get(k, {}).get('careunit') or '未匹配科室') for k in ['source_location', 'target_location'])
            summary = ctx.get('interval_event_summary', {})
            events = '; '.join(f'{name} × {count}' for block in ['procedureevent_counts', 'inputevent_counts'] for name, count in (summary.get(block) or {}).items()) or '所选 ICU 表中无区间事件记录，不能推断没有治疗。'
        else:
            admission_text = 'hadm_id：'+', '.join(ctx['source_hadm_ids'])+' → '+', '.join(ctx['target_hadm_ids'])
            locations = 'ICU stay：'+(', '.join(ctx['source_stay_ids']) or '未匹配')+' → '+(', '.join(ctx['target_stay_ids']) or '未匹配')
            events = '本例只核验住院/ICU 区间，未提取用药/操作明细。'
        label_rows = ''.join('<tr><td>'+esc(k)+'</td><td>'+esc(case['source']['labels'][k])+'</td><td>'+esc(case['target']['labels'][k])+'</td></tr>' for k in FINDINGS)
        details = esc(json.dumps(ctx, ensure_ascii=False, indent=2))
        middle = f'<div class="context"><h3>MIMIC-IV 临床连接</h3><p>{esc(admission_text)}</p><p>{esc(locations)}</p><div class="review">{esc(case["review_note"])}</div><h4>与随访区间重叠的记录</h4><p>{esc(events)}</p><small>包括当前已持续的记录；不能全部视为新发生。区间实际治疗用于回顾性核验，不能直接加入 t0 预测输入。</small><details><summary>查看临床记录和时间戳</summary><pre>{details}</pre></details></div>'
        cards.append(f'<article id="case-{case["case"]}" data-case="{case["case"]}"><h2>{case["case"]} · {esc(case["title"])}</h2><p class="origin">{esc(case["origin"])} · {esc(case["split"])}</p><div class="columns">{side_html[0]}{middle}{side_html[1]}</div><details><summary>比较六疾病弱标签：未提及与不确定均不等于阴性</summary><table><tr><th>疾病</th><th>当前 CheXpert</th><th>未来 CheXpert</th></tr>{label_rows}</table></details><details><summary>核验来源标识符</summary><p>subject_id {esc(case["subject_id"])} · transition {esc(case["transition_id"])} · source study {esc(case["source"]["study_id"])} → target study {esc(case["target"]["study_id"])}</p></details></article>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CXR + IV 配对核验示例</title>
<style>body{font:16px/1.65 system-ui,sans-serif;color:#203040;background:#eef2f5;margin:0}main{max-width:1460px;margin:auto;padding:24px}header,article{background:white;border-radius:12px;padding:24px;margin-bottom:24px}h1,h2,h3{line-height:1.35}.columns{display:grid;grid-template-columns:1fr .8fr 1fr;gap:24px}img{display:block;width:100%;height:410px;object-fit:contain;background:#111}.metadata,small,.origin{font-size:13px;color:#596674}.context{background:#f0f5fa;padding:16px;border-radius:8px}.review{border-left:4px solid #2c779d;padding:12px;background:white}details{margin:12px 0;padding:10px;background:#f7f9fb;overflow-wrap:anywhere}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;font:12px/1.5 monospace;max-height:500px;overflow:auto}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:8px;border-bottom:1px solid #dde3e8}button{background:#e4edf4;border:0;border-radius:6px;padding:10px 18px;margin:4px;cursor:pointer}button.active{background:#205879;color:white}@media(max-width:900px){.columns{grid-template-columns:1fr}img{height:auto;max-height:560px}}@media print{button{display:none}article{break-inside:avoid}}</style>
<main><header><h1>MIMIC-CXR + MIMIC-IV：看真实配对</h1><p>2026-09-09 · 六组本地核验示例。当前和随访图像均来自真实记录，没有生成或增强影像。A–D 来自既有 appendix 示例库；E–F 来自实际 pilot 训练缓存。按讨论问题选例，不用于估计噪声比例，也没有做新的临床判读。</p><p><strong>读图顺序：</strong>先看当前片和报告，再核对住院/临床时间线，最后看真实随访。区间内实际发生的治疗和未来报告用于回顾性核验；输入时可用的 EHR 还需核查发生时间与记录时间。</p><nav><button class="active" data-select="all">全部示例</button>'''+''.join(f'<button data-select="{c["case"]}">{c["case"]}</button>' for c in cases)+'''</nav></header>'''+''.join(cards)+'''</main><script>document.querySelectorAll('[data-select]').forEach(b=>b.onclick=()=>{document.querySelectorAll('[data-select]').forEach(x=>x.classList.toggle('active',x===b));document.querySelectorAll('article').forEach(x=>x.hidden=b.dataset.select!=='all'&&x.dataset.case!==b.dataset.select);});</script></html>'''
    (OUT / 'index.html').write_text(page)
    # Scientific comparison sheet: preserve full image field and aspect ratio.
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fig, axes = plt.subplots(3, 3, figsize=(13.4, 13.2), gridspec_kw={'width_ratios': [1, 1, .85]})
    fig.suptitle('真实 CXR + IV 配对示例：变化、稳定与置管随访', fontproperties=font, fontsize=18, y=.985)
    for axes_row, case in zip(axes, cases[:3]):
        for ax, side, title in zip(axes_row[:2], ['source', 'target'], ['当前 t0', f'随访 +{case["gap_hours"]:.1f}h']):
            ax.imshow(plt.imread(case[side]['image']['path']), cmap='gray')
            ax.set_title(case['case']+' · '+title, fontproperties=font, fontsize=12)
            ax.axis('off')
        ax = axes_row[2]
        ax.axis('off')
        body = '\n\n'.join('\n'.join(textwrap.wrap(t, 21)) for t in [case['title'], case['source_summary'], case['target_summary'], case['review_note']])
        ax.text(0, .95, body, va='top', fontproperties=font, fontsize=10.5, linespacing=1.6)
    fig.text(.04, .025, '描述依据原始报告和临床记录；示例按问题挑选。完整报告、弱标签及另三类样本见 HTML。', fontproperties=font, fontsize=10)
    fig.subplots_adjust(top=.95, bottom=.055, left=.025, right=.98, hspace=.16, wspace=.07)
    fig.savefig(OUT / 'preview.png', dpi=140, facecolor='white')
    plt.close(fig)
    (OUT / 'README.md').write_text('# CXR + IV 配对核验示例\n\n打开 `index.html`：图片已嵌入，无需联网或单独图片文件。`examples.json` 保留原始报告、六疾病标签和临床连接记录；`preview.png` 为前三组概览。\n\n示例为本地讨论挑选，不是随机抽样或临床裁决。A–D 为既有 appendix 示例，E–F 为实际 pilot 训练样本。未进行模型训练或推理。\n')
    print(json.dumps({'output': str(OUT), 'html_bytes': (OUT/'index.html').stat().st_size,
                      'cases': [{'case': c['case'], 'title': c['title'], 'gap_hours': c['gap_hours'], 'linkage':c['linkage_status']} for c in cases]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
