"""Standalone, aggregate-only local dataset audit page; no external resources."""
import argparse
import json
from pathlib import Path

PAGE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>MIMIC 数据现状 · 2026-09-10</title>
<style>
:root{color-scheme:light;--blue:#356a91;--orange:#bd703c;--gray:#88949c}*{box-sizing:border-box}body{margin:0;background:#f2f5f7;color:#183041;font:16px/1.65 system-ui,sans-serif}main{max-width:1140px;margin:32px auto;padding:0 22px}h1{font-size:30px;margin-bottom:8px}h2{font-size:21px;margin:0 0 14px}h3{font-size:17px}.muted{color:#59707f}.card{background:white;padding:23px;border:1px solid #dce4e9;border-radius:12px;margin:18px 0}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:20px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.stat{background:#edf3f7;padding:14px;border-radius:8px}.stat b{display:block;font-size:25px}.stat span{font-size:14px;color:#59707f}table{width:100%;border-collapse:collapse;font-size:15px}th,td{border-bottom:1px solid #e3e9ed;padding:9px 8px;text-align:left}td.num,th.num{text-align:right}select{font:inherit;padding:8px;border:1px solid #aebdc8;border-radius:6px;max-width:100%}.barrow{display:grid;grid-template-columns:116px 1fr 126px;align-items:center;gap:10px;margin:14px 0;font-size:14px}.track{height:20px;background:#eef1f3;border-radius:3px}.fill{height:100%;border-radius:3px}.number{text-align:right}.note{background:#fff5e8;border-left:4px solid #c68547;padding:13px 17px;margin-top:16px}.timeline{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}.timeline div{padding:18px;background:#edf3f7;border-radius:8px}.timeline strong{display:block;color:var(--blue)}a{color:#256792}svg{width:100%;height:220px}summary{cursor:pointer;font-weight:600}footer{font-size:14px;color:#59707f;padding:18px 0 35px}@media(max-width:780px){.grid,.timeline{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}.barrow{grid-template-columns:95px 1fr 112px}.card{padding:17px}h1{font-size:25px}}
</style><main>
<h1>MIMIC-CXR + IV：我们现在的数据</h1>
<p class="muted">2026-09-10 本地审计。所有数字来自已有原始数据、训练清单与 Qwen 结果；此页不修改数据或启动训练。</p>
<section class="card"><h2>目前的数据规模</h2>
<p><b>昨晚训练 6,000 对。</b>保留同样的严格条件和每患者最多 4 对，可扩到 <b>16,426 对</b>；不加患者配额，严格候选为 <b>23,009 对</b>。放宽采集字段一致性后，训练候选为 <b>40,410 对</b>。</p>
<table><tr><th>范围</th><th class="num">Train</th><th class="num">Validate</th><th class="num">Test</th></tr>
<tr><td>同住院、6–72h、图像 QC、当前前临床记录</td><td class="num">40,410</td><td class="num">403</td><td class="num">529</td></tr>
<tr><td>再要求采集字段已知且一致</td><td class="num">23,009</td><td class="num">230</td><td class="num">297</td></tr>
<tr><td>昨晚实际使用</td><td class="num">6,000</td><td class="num">230</td><td class="num">297</td></tr></table>
<p class="muted">本地全库 377,110 张图像、227,835 次检查、65,379 位患者；上述 pair 是筛选后的子集。图像、检查、pair 和患者不能混用，也不能把嵌套子集相加。</p></section>
<section class="card"><h2>切换队列查看时间和变化比例</h2><label>统计范围：<select id="scope">
<option value="actual_train">昨晚实际训练：6,000 对</option>
<option value="actual_validate">昨晚验证：230 对</option>
<option value="actual_test">昨晚测试：297 对</option>
<option value="same_admission_6h_72h_acquisition_matched_image_qc_prior_ehr/train">完整严格训练候选：23,009 对</option>
<option value="same_admission_6h_72h_image_qc_prior_ehr/train">放宽采集一致性的训练候选：40,410 对</option>
</select></label><div id="cards" class="stats" style="margin-top:20px"></div>
<div class="grid"><div><h3>从当前到相邻未来检查的间隔</h3><div id="hist"></div><p id="horizons" class="muted"></p></div>
<div><h3>按原始六疾病标签统计</h3><div id="official"></div><p class="muted">“未观察到变化”仅指前后明确的二值字段没翻转；可能仍有严重程度变化。缺少可比较字段不能算稳定。</p></div></div>
<div class="grid"><div><h3>Qwen 原始语义判定 · 仅作参考</h3><div id="qwen"></div><p id="stable"></p></div>
<div><h3>Qwen 保守复核分流 · 不自动删样本</h3><div id="review"></div><p id="flags" class="muted"></p></div></div>
<div class="note">Qwen 同时看两张片和两份报告；不是独立、盲法的影像评审。原始标签和 Qwen 的“变化”定义不同。模型判为 changed 不等于已取得可靠的临床变化真值；待复核也不等于数据无用。</div></section>
<section class="card"><h2>时间点与预测输入</h2><div class="timeline">
<div><strong>t0 以前</strong>化验/ICU 监护：前 48h<br>eMAR：前 24h<br>事件时间、storetime 均 ≤ t0</div>
<div><strong>t0：当前片采集</strong>当前胸片＋原始当前报告<br>＋已有临床值<br>＋6–24h / >24–72h 档位</div>
<div><strong>t1：下一次相邻检查</strong>间隔 6–72h，实际时间不同<br>未来图像、报告/疾病标签用于监督<br>推理输入不包含未来证据</div></div>
<p class="muted">当前报告签发时刻不可核实，采用回顾性“报告已可用”假设。模型输入时间档位，不输入精确实测间隔。</p>
<p>实际 6,000 个训练当前端中，5,963 对有选定近期 EHR 值：化验 5,651 对、ICU 监护 4,308 对、eMAR 2,621 对。三类重叠；没选到记录不等于临床正常。</p></section>
<section class="card"><h2>Qwen 全量运行状态</h2><p><b>136,383 / 136,385 对完成，99.9985%。</b>剩余 2 对在 2,048 输出 token 上限仍未正常结束。昨晚所用严格训练/验证/测试子集全部已有结果。</p>
<table><tr><th>扩展审阅清单的原始判定</th><th class="num">数量</th><th class="num">占已完成结果</th></tr>
<tr><td>有变化</td><td class="num">98,691</td><td class="num">72.36%</td></tr><tr><td>稳定</td><td class="num">37,577</td><td class="num">27.55%</td></tr><tr><td>无法判断</td><td class="num">115</td><td class="num">0.08%</td></tr></table>
<p class="muted">这份清单允许视角不一致、跨住院和更长/更短间隔，并非全部满足当前预测任务。保守分流还有 105,723 对待复核；这里的比例不能当作人工审定的病情变化率。</p></section>
<section class="card"><h2>如何理解“有用”</h2><p>结构和时间条件合格的数据可以用于继续扩展训练。原始疾病标签未知的样本仍可能提供图像、报告或状态监督；稳定随访可帮助模型学习持续异常和合理的无变化预测。</p>
<p>若要专门监督疾病变化，完整严格训练池中有 3,043 对原始六疾病可观察变化、2,310 对 Qwen 保守暂定变化。两组有交集、定义不同，不能相加，也都不能直接称为人工真值。</p>
<p><a href="../../research_notes/0910_dataset_summary.md">完整中文报告与解释</a> · <a href="statistics.json">聚合统计 JSON</a> · <a href="training_data_overview.png">训练子集图 PNG</a> · <a href="training_data_overview.pdf">PDF</a></p></section>
<footer>只包含聚合统计。Qwen 标签未用于昨晚的训练抽样、数据平衡或监督。已选样本按 epoch 无放回采样，6,000 个不同 pair 在 Stage 2 共读取 25,352 次（约 4.23 遍）。</footer>
</main><script>const DATA=__DATA__;
const fmt=x=>Number(x).toLocaleString('en-US');
function bars(id,entries,n){document.getElementById(id).innerHTML=entries.map(([label,count,color])=>`<div class="barrow"><span>${label}</span><div class="track"><div class="fill" style="width:${100*count/n}%;background:${color}"></div></div><span class="number">${fmt(count)} (${(100*count/n).toFixed(2)}%)</span></div>`).join('')}
function update(){const s=DATA.scopes[document.getElementById('scope').value],n=s.pairs,q=s.qwen,o=s.official_chexpert.categories;
document.getElementById('cards').innerHTML=[[fmt(n),'不同检查对'],[fmt(s.patients),'不同患者'],[fmt(s.unique_images),'不同图像'],[s.gap_hours.median.toFixed(2)+'h','间隔中位数']].map(([v,l])=>`<div class="stat"><b>${v}</b><span>${l}</span></div>`).join('');
bars('official',[['可观察变化',o.observed_change||0,'var(--orange)'],['未观察到变化',o.no_observed_change||0,'var(--blue)'],['无可比较字段',o.no_comparable_field||0,'var(--gray)']],n);
bars('qwen',[['有变化',q.classes.changed||0,'var(--orange)'],['稳定',q.classes.stable||0,'var(--blue)'],['无法判断',q.classes.indeterminate||0,'var(--gray)']],n);
bars('review',[['暂定变化',q.routing.changed||0,'var(--orange)'],['暂定稳定',q.routing.stable||0,'var(--blue)'],['待复核',q.routing.needs_review||0,'var(--gray)']],n);
const stable=q.stable_subtypes_within_primary_stable;document.getElementById('stable').textContent=`稳定中：持续异常 ${fmt(stable.persistent_abnormalities||0)} 对，无急性异常 ${fmt(stable.no_acute_findings||0)} 对。稳定不等于正常。`;
document.getElementById('flags').textContent=`图像/报告判定不一致标记 ${fmt(q.review_flags.image_report_disagreement||0)} 对，可能技术干扰 ${fmt(q.review_flags.possible_technical_confound||0)} 对；两者有重叠。`;
document.getElementById('horizons').textContent=`6–24h：${fmt(s.horizon_counts['6-24h']||0)} 对；>24–72h：${fmt(s.horizon_counts['>24-72h']||0)} 对。中间50%：${s.gap_hours.p25.toFixed(2)}–${s.gap_hours.p75.toFixed(2)}h。`;
const h=s.gap_histogram,top=Math.max(...h.counts),w=430,left=35,base=172,width=380/24;
let svg=`<svg viewBox="0 0 440 210" role="img" aria-label="时间间隔直方图"><line x1="${left}" y1="${base}" x2="420" y2="${base}" stroke="#8194a0"/>`;
h.counts.forEach((v,i)=>{const height=140*v/top;svg+=`<rect x="${left+i*width}" y="${base-height}" width="${width-1}" height="${height}" fill="#477ca8"><title>${h.edges[i]}–${h.edges[i+1]}h：${v} 对</title></rect>`});
[0,12,24,36,48,60,72].forEach(x=>svg+=`<text x="${left+x/72*380}" y="194" text-anchor="middle" font-size="12" fill="#536b7b">${x}h</text>`);
const mx=left+s.gap_hours.median/72*380;svg+=`<line x1="${mx}" y1="20" x2="${mx}" y2="${base}" stroke="#bc703c" stroke-dasharray="4 4"/></svg>`;document.getElementById('hist').innerHTML=svg;}
document.getElementById('scope').addEventListener('change',update);update();</script></html>'''


def render(directory):
    data=json.loads((directory/'statistics.json').read_text())
    (directory/'index.html').write_text(PAGE.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    render(p.parse_args().out)
