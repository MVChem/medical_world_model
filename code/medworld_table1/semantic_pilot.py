"""Small train-only local LLM schema pilot; leaves existing data/metrics untouched."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import urllib.parse
import urllib.request

from semantic_schema import FINDINGS, VERSION, protocol_hash, request, validate
from semantic_metrics import RULE_VERSION, pair_events, score


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read_rows(path):
    with Path(path).open() as f: return [json.loads(line) for line in f if line.strip()]


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.chmod(temp, 0o600); temp.replace(path)


def http(url, payload=None):
    req = urllib.request.Request(url, data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=240) as response: return json.load(response)


SYNTHETIC = {
    'current_large': 'Large right pleural effusion.',
    'future_small': 'Small right pleural effusion, decreased compared with the supplied current examination.',
    'wrong_unchanged': 'Large right pleural effusion, unchanged compared with the supplied current examination.',
    'wrong_resolved': 'No pleural effusion.',
    'omitted': 'The endotracheal tube is in place.',
    'uncertain': 'Possible small right pleural effusion.',
    'mixed_current': 'Large right and small left pleural effusions.',
    'mixed_future': 'Small right and large left pleural effusions.',
    'range_current': 'Moderate-to-large right pleural effusion.',
    'range_future': 'Moderate right pleural effusion.',
}


def build_inputs(cache, count):
    rows = read_rows(cache/'train.jsonl')
    obs = {r['id']: r for r in read_rows(cache/'observations.jsonl')}
    rows.sort(key=lambda r: sha('semantic-pilot-v1:'+r['id']))
    selected, patients, reports = [], set(), {}
    for row in rows:
        if row['patient'] in patients: continue
        patients.add(row['patient'])
        c, t = obs[row['source']], obs[row['target']]
        if row.get('split', 'train') != 'train' or c['split'] != 'train' or t['split'] != 'train':
            raise ValueError('Schema development is restricted to training reports')
        current, future = sha(c['report']), sha(t['report'])
        reports[current] = c['report']; reports[future] = t['report']
        selected.append(dict(id=row['id'], patient=row['patient'], current=current, future=future))
        if len(selected) == count: break
    synthetic = {k: sha(v) for k, v in SYNTHETIC.items()}
    reports.update({sha(v): v for v in SYNTHETIC.values()})
    return dict(selection='Deterministic train pair hash, one pair per patient; no outcome/model filter',
        selected_pairs=selected, synthetic=synthetic, reports=reports)


def extract(report_id, report, out, endpoint, model, signature):
    path = out/'extractions'/(report_id+'.json')
    if path.exists():
        result = json.loads(path.read_text())
        if result['signature'] != signature or result['report_sha256'] != report_id:
            raise ValueError('Extraction cache provenance mismatch')
        return report_id, result
    result = dict(signature=signature, report_sha256=report_id, report=report,
        model=model, protocol_version=VERSION, protocol_sha256=protocol_hash())
    try:
        response = http(endpoint+'/v1/chat/completions', request(report, model))
        result['response'] = response
        choice = response['choices'][0]
        if choice['finish_reason'] != 'stop': raise ValueError('Incomplete LLM generation: '+str(choice['finish_reason']))
        annotation = json.loads(choice['message']['content'])
        errors = validate(annotation, report)
        result.update(annotation=annotation, validation_errors=errors, valid=not errors)
    except Exception as exc:
        result.update(valid=False, validation_errors=[type(exc).__name__+': '+str(exc)])
    save(path, result)
    return report_id, result


def render(out, inputs, extracted, summary):
    esc = html.escape
    def panel(report_id):
        r = extracted[report_id]
        data = r.get('annotation', {})
        rows = []
        for name in FINDINGS:
            for m in data.get('findings', {}).get(name, []):
                rows.append('<tr>'+''.join('<td>'+esc(str(v))+'</td>' for v in
                    [name, m['assertion'], m['laterality'], m['site'], m['degree'], m['change'], m['evidence']])+'</tr>')
        return '<pre>'+esc(r['report'])+'</pre><p>'+('结构和原文引用校验通过；语义待核验' if r['valid'] else esc(str(r['validation_errors'])))+'</p>'+\
            '<div class="scroll"><table><tr><th>疾病</th><th>状态</th><th>侧别</th><th>部位原文</th><th>程度原文</th><th>报告变化词</th><th>证据</th></tr>'+''.join(rows)+'</table></div>'+\
            '<details><summary>完整抽取（含其他疾病、设备、技术因素）</summary><pre>'+esc(json.dumps(data, ensure_ascii=False, indent=2))+'</pre></details>'
    blocks = []
    for i, row in enumerate(inputs['selected_pairs'], 1):
        blocks.append(f'<section><h2>训练样本 {i}</h2><div class="pair"><div><h3>当前报告</h3>'+panel(row['current'])+
            '</div><div><h3>未来真实报告</h3>'+panel(row['future'])+'</div></div>')
        c, t = extracted[row['current']], extracted[row['future']]
        if c['valid'] and t['valid']:
            events = [r for r in pair_events(c['annotation'], t['annotation']) if r['event'] or r['reported_change']]
            blocks.append('<details open><summary>固定规则产生的变化及待确认比较</summary><pre>'+esc(json.dumps(events, ensure_ascii=False, indent=2))+'</pre></details>')
        blocks.append('</section>')
    for name, report_id in inputs['synthetic'].items():
        blocks.append('<section><h2>人工构造的验证句：'+esc(name)+'</h2>'+panel(report_id)+'</section>')
    demos = json.loads((out/'synthetic_checks.json').read_text()) if (out/'synthetic_checks.json').exists() else {}
    demo_rows = []
    names = {'future_small': '正确：少量，较前减少', 'wrong_unchanged': '错误：仍大量，无变化',
        'wrong_resolved': '错误：无积液', 'omitted': '漏写积液', 'uncertain': '只写可能少量积液'}
    for key, label in names.items():
        if key not in demos: continue
        r = demos[key]
        values = [label, r['presence_macro_f1'], r['endpoint_events']['overall']['change_macro_f1'], r['degree']['exact_accuracy']]
        demo_rows.append('<tr>'+''.join('<td>'+esc(str(v))+'</td>' for v in values)+'</tr>')
    demo = '<h2>验证例：当前大量 → 真实未来少量</h2><p>人工构造的句子经 LLM 抽取后由固定规则计算。以下仅是该疾病、该例的演示分数。</p><table><tr><th>预测内容</th><th>疾病存在 F1</th><th>改善事件 F1</th><th>程度字段准确率</th></tr>'+''.join(demo_rows)+'</table>' if demo_rows else ''
    content='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>报告语义评价原型</title>
<style>body{max-width:1500px;margin:32px auto;padding:0 24px;font:15px/1.6 sans-serif;color:#173047;background:#f4f7fa}section{background:white;padding:22px;margin:22px 0;border:1px solid #ccd8e1}h1,h2{color:#16486b}.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px;min-width:0}.pair>div{min-width:0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf3f7;padding:12px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #cad8e0;padding:7px;text-align:left;vertical-align:top}.scroll{overflow-x:auto}details{margin-top:16px}@media(max-width:900px){.pair{grid-template-columns:1fr}}</style>
<h1>报告语义抽取 → 固定规则评价：小样本原型</h1>
<p>仅使用训练集报告＋人工构造的验证句。原文引用校验不等于临床正确率，未获人工金标准验收。未修改原 Table 1。两份报告分别独立抽取；LLM 不给模型打分。</p>'''+\
        f"<p><b>{summary['train_pairs']} 对训练病例 · {summary['unique_reports']} 份报告（含验证句） · {summary['schema_grounding_valid']} 份通过结构与原文引用校验</b></p>"+\
        '<details><summary>完整运行统计与未映射字段</summary><pre>'+esc(json.dumps(summary, ensure_ascii=False, indent=2))+'</pre></details>'+demo+''.join(blocks)+'</html>'
    (out/'index.html').write_text(content)


def render_saved(out):
    """Read-only replay of frozen annotations; no inference and no reclassification."""
    manifest = json.loads((out/'manifest.json').read_text())
    extracted = {rid: json.loads((out/'extractions'/(rid+'.json')).read_text()) for rid in manifest['inputs']['reports']}
    summary = json.loads((out/'summary.json').read_text())
    render(out, manifest['inputs'], extracted, summary)
    save(out/'render_provenance.json', dict(renderer_sha256=sha(Path(__file__).read_text()),
        source_manifest_signature=manifest['signature'], annotations_recomputed=False))


def run(args):
    os.umask(0o077)
    parsed = urllib.parse.urlparse(args.endpoint)
    if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('This pilot only uses local inference endpoints')
    inputs = build_inputs(args.cache, args.pairs)
    model_response = http(args.endpoint+'/v1/models')['data'][0]
    # Dynamic response timestamps and permission IDs are not model identity.
    model = {key: model_response.get(key) for key in ['id', 'root', 'parent', 'owned_by']}
    code_hashes = {name: sha(Path(__file__).with_name(name).read_text()) for name in
        ['semantic_schema.py', 'semantic_metrics.py', 'semantic_pilot.py']}
    manifest = dict(inputs=inputs, model=model, endpoint=args.endpoint, protocol_sha256=protocol_hash(),
        rule_version=RULE_VERSION, code_hashes=code_hashes)
    signature = sha(json.dumps(manifest, sort_keys=True))
    args.out.mkdir(parents=True, exist_ok=True)
    existing = args.out/'manifest.json'
    if existing.exists() and json.loads(existing.read_text())['signature'] != signature:
        raise ValueError('Pilot provenance differs; choose a new output directory')
    save(existing, dict(signature=signature, **manifest))
    (args.out/'source').mkdir(exist_ok=True)
    for name in code_hashes:
        shutil.copy2(Path(__file__).with_name(name), args.out/'source'/name)
    extracted = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(extract, rid, text, args.out, args.endpoint, model['id'], signature)
                   for rid, text in inputs['reports'].items()]
        for future in as_completed(futures):
            rid, result = future.result(); extracted[rid] = result
            print(f"Extracted {len(extracted)}/{len(futures)}; valid={result['valid']}", flush=True)
    event_counts, unsupported_degrees, extras = Counter(), Counter(), Counter()
    valid_pairs = 0
    for row in inputs['selected_pairs']:
        c, t = extracted[row['current']], extracted[row['future']]
        if c['valid'] and t['valid']:
            valid_pairs += 1
            for e in pair_events(c['annotation'], t['annotation']):
                if e['scope'] == 'overall': event_counts[e['event'] or 'unassessable'] += 1
    from semantic_metrics import degree_interval
    for rid in {r[k] for r in inputs['selected_pairs'] for k in ['current', 'future']}:
        result = extracted[rid]
        if not result['valid']: continue
        for name, mentions in result['annotation']['findings'].items():
            for m in mentions:
                if m['degree'] and degree_interval(name, m['degree']) is None: unsupported_degrees[name+': '+m['degree']] += 1
        extras.update(m['name'] for m in result['annotation']['other_findings'])
    demos = {}
    syn = inputs['synthetic']
    if all(extracted[rid]['valid'] for rid in syn.values()):
        ann = {k: extracted[rid]['annotation'] for k, rid in syn.items()}
        def anchors(key):
            return {'Pleural Effusion|'+scope: [m['evidence'] for m in ann[key]['findings']['Pleural Effusion']
                    if m['change'] != 'not_stated'] for scope in ['overall', 'right']}
        for name in ['future_small', 'wrong_unchanged', 'wrong_resolved', 'omitted', 'uncertain']:
            row = dict(id='synthetic', current=ann['current_large'], reference=ann['future_small'], prediction=ann[name],
                reference_comparisons=anchors('future_small'), prediction_comparisons=anchors(name))
            demos[name] = score([row])
        demos['mixed_sides'] = pair_events(ann['mixed_current'], ann['mixed_future'])
        demos['overlapping_range'] = pair_events(ann['range_current'], ann['range_future'])
    save(args.out/'synthetic_checks.json', demos)
    synthetic_audit = []
    from semantic_metrics import state
    expected = {'current_large': (1, 'not_stated'), 'future_small': (1, 'decreased'),
        'wrong_unchanged': (1, 'unchanged'), 'wrong_resolved': (0, 'not_stated'),
        'omitted': (None, 'not_stated'), 'uncertain': (None, 'not_stated'),
        'mixed_current': (1, 'not_stated'), 'mixed_future': (1, 'not_stated'),
        'range_current': (1, 'not_stated'), 'range_future': (1, 'not_stated')}
    for name, (presence, change) in expected.items():
        result = extracted[syn[name]]
        passed = False
        actual = None
        if result['valid']:
            ann = result['annotation']
            actual = dict(presence=state(ann, 'Pleural Effusion')['presence'],
                changes=sorted({m['change'] for m in ann['findings']['Pleural Effusion']}))
            passed = actual['presence'] == presence and all(x == change for x in actual['changes'])
        synthetic_audit.append(dict(name=name, passed=passed, expected_presence=presence,
            expected_change=change, actual=actual, validation_errors=result['validation_errors']))
    save(args.out/'synthetic_semantic_audit.json', synthetic_audit)
    summary = dict(protocol=VERSION, train_pairs=len(inputs['selected_pairs']), unique_reports=len(extracted),
        schema_grounding_valid=sum(r['valid'] for r in extracted.values()), valid_train_pairs=valid_pairs,
        overall_endpoint_events=dict(event_counts), unsupported_degree_phrases=dict(unsupported_degrees),
        other_findings_for_schema_review=dict(extras), synthetic_checks_available=bool(demos),
        synthetic_presence_change_checks_passed=sum(x['passed'] for x in synthetic_audit),
        synthetic_presence_change_checks_total=len(synthetic_audit),
        semantics_human_adjudicated=False, used_for_table1=False,
        caveat='Counts describe extraction feasibility and reference coverage, not clinical accuracy. Unanchored comparisons are retained but not scored.')
    save(args.out/'summary.json', summary)
    render(args.out, inputs, extracted, summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, default=Path(__file__).parent/'data/linked_20260909')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--pairs', type=int, default=12)
    p.add_argument('--endpoint', default='http://127.0.0.1:8120')
    p.add_argument('--concurrency', type=int, default=2)
    p.add_argument('--render-only', action='store_true')
    a = p.parse_args()
    if a.pairs < 1 or a.concurrency < 1: p.error('pairs and concurrency must be positive')
    if a.render_only: render_saved(a.out)
    else: run(a)
