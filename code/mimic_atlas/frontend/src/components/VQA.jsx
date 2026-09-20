import { useEffect, useState } from "react";
import { useDebounced, useResource } from "../api";
import { fmt } from "../format";
import "../styles/vqa.css";

const defaults = { split: "all", semantic: "all", content: "all", answers: "all", q: "" };
function Answer({ values }) {
  return values.length ? <span className="vqa-answers">{values.map((value, i) => <span key={i}>{value}</span>)}</span> : <span className="muted">∅ 空答案集合 []</span>;
}
function Detail({ row, quality, imagesReady, onClose, onOpen }) {
  const [page, setPage] = useState(1);
  const [failed, setFailed] = useState(false);
  const [zoom, setZoom] = useState(false);
  const { data, error } = useResource(`/api/vqa/questions?image_id=${encodeURIComponent(row.image_id)}&page=${page}`);
  return <section className="panel vqa-detail" aria-label="问答详情">
    <div className="panel-heading"><div><span className="eyebrow">IMAGE & QUESTION</span><h2>胸片与参考答案</h2></div><button onClick={onClose}>关闭详情 ×</button></div>
    <div className="vqa-detail-grid">
      <div>
        <button className="vqa-image" onClick={() => setZoom(true)} disabled={failed || !imagesReady} aria-label="放大胸片">
          {!imagesReady ? <span>正在读取胸片目录…</span> : failed ? <span>该胸片在本地不可用</span> : <img src={`/api/images/${encodeURIComponent(row.image_id)}?size=${quality}`} alt={`患者 ${row.subject_id} 的胸片`} onError={() => setFailed(true)} />}
        </button>
        <p className="muted">点击影像放大 · 原始胸片，不包含区域标注框</p>
        <dl className="vqa-identifiers"><dt>患者</dt><dd>{row.subject_id}</dd><dt>检查</dt><dd>{row.study_id}</dd><dt>影像 ID</dt><dd>{row.image_id}</dd><dt>来源</dt><dd>{row.split}.json · idx {row.idx}</dd></dl>
        <button onClick={() => onOpen(row.subject_id)}>查看患者时间线与临床记录 ↗</button>
      </div>
      <div className="vqa-question-detail">
        <div className="vqa-tags"><span>{row.split}</span><span>{row.semantic_type}</span><span>{row.content_type}</span></div>
        <h3>{row.question}</h3><p className="eyebrow">DATASET REFERENCE ANSWER</p><Answer values={row.answer} />
        <p className="muted">这里展示数据集原始问答与参考答案，不是模型预测。空集合保留为 []。</p>
        <h3>同一胸片的全部问答 <span className="muted">{data ? fmt(data.total) : "…"}</span></h3>
        <p className="muted">包含所有官方 split；不受问答目录筛选限制。</p>
        {error && <div className="notice" role="alert">{error}</div>}
        {!data && !error && <p role="status">读取同图问答…</p>}
        <div className="vqa-related">{data?.rows.map(item => <article key={item.id} className={item.id === row.id ? "selected" : ""}><small>{item.split} · {item.semantic_type} · {item.content_type} · idx {item.idx}</small><p>{item.question}</p><Answer values={item.answer} /></article>)}</div>
        <div className="observation-pagination"><button disabled={page === 1} onClick={() => setPage(page - 1)} aria-label="同图问答上一页">←</button><span>{page} / {Math.max(1, Math.ceil((data?.total || 0) / 25))}</span><button disabled={!data || page * 25 >= data.total} onClick={() => setPage(page + 1)} aria-label="同图问答下一页">→</button></div>
      </div>
    </div>
    {zoom && <div className="vqa-lightbox" role="dialog" aria-modal="true" aria-label="胸片放大" onClick={() => setZoom(false)} onKeyDown={e => { if (e.key === "Escape") setZoom(false); }}><button autoFocus onClick={() => setZoom(false)}>关闭 ×</button><img src={`/api/images/${encodeURIComponent(row.image_id)}?size=1800`} alt="放大胸片" /></div>}
  </section>;
}
export default function VQA({ quality, imagesReady, onOpen }) {
  const [poll, setPoll] = useState(1000);
  const { data: summary, error: summaryError } = useResource("/api/vqa/summary", poll);
  const [filters, setFilters] = useState(defaults);
  const [page, setPage] = useState(1);
  const [selected, setSelected] = useState(null);
  const search = useDebounced(filters.q);
  const params = new URLSearchParams({ ...filters, q: search, page });
  const ready = summary?.state === "ready";
  const { data, error } = useResource(ready ? `/api/vqa/questions?${params}` : null);
  // Poll only while the lazy source catalog is being prepared.
  useEffect(() => { if (ready || summary?.state === "error") setPoll(0); }, [ready, summary?.state]);
  useEffect(() => { if (selected) document.querySelector(".vqa-detail")?.scrollIntoView({ behavior: "smooth", block: "start" }); }, [selected]);
  function change(key, value) { setFilters(old => ({ ...old, [key]: value })); setPage(1); setSelected(null); }
  const groups = summary?.splits || {};
  const types = key => [...new Set(Object.values(groups).flatMap(s => Object.keys(s[key])))].sort();
  const distribution = Object.values(filters.split === "all" ? groups : { [filters.split]: groups[filters.split] }).filter(Boolean).reduce((counts, group) => { for (const [k, n] of Object.entries(group.semantic_types)) counts[k] = (counts[k] || 0) + n; return counts; }, {});
  const maximum = Math.max(1, ...Object.values(distribution));
  return <div id="vqa-content">
    <section className="page-heading"><div><div className="eyebrow">MIMIC-CXR-VQA · 1.0.0</div><h1>VQA 问答</h1><p>从问题回到胸片，浏览原始参考答案与同图问答。</p></div><span className="subtle-badge">{fmt(summary?.questions)} 道问题</span></section>
    {(summaryError || summary?.error) && <div className="notice" role="alert">{summaryError || summary.error}</div>}
    {!ready && !summary?.error && !summaryError && <div className="empty-state" role="status">正在读取 VQA 原始数据，首次打开需要数秒…</div>}
    {ready && <>
      <div className="vqa-summary">{Object.entries(groups).map(([split, stats]) => <button key={split} className={`panel vqa-stat ${filters.split === split ? "selected" : ""}`} onClick={() => change("split", filters.split === split ? "all" : split)}><span className="eyebrow">{split}</span><strong>{fmt(stats.questions)} <small>题</small></strong><span>{fmt(stats.images)} 张胸片 · {fmt(stats.patients)} 位患者</span><small className="muted">空答案 {fmt(stats.empty_answers)} · {(stats.empty_answers / stats.questions * 100).toFixed(1)}%</small></button>)}</div>
      <section className="panel vqa-overview"><div><h3>题型分布</h3><p className="muted">{filters.split === "all" ? "全部官方集合" : filters.split} · 按问题计数，点击条形筛选</p><div className="vqa-bars">{Object.entries(distribution).map(([type, n]) => <button key={type} aria-pressed={filters.semantic === type} onClick={() => change("semantic", filters.semantic === type ? "all" : type)}><span>{type}</span><span className="vqa-bar-track"><i style={{ width: `${n / maximum * 100}%` }} /></span><span>{fmt(n)}</span></button>)}</div></div><div><h3>数据范围与划分</h3><p>共 {fmt(summary.images)} 张不同胸片、{fmt(summary.patients)} 位去重患者。问题和答案保留原始英文。</p><p className="muted">患者交集：{Object.entries(summary.overlap).map(([k, v]) => `${k} ${fmt(v)} 人`).join("；")}。官方 split 与 Atlas 影像划分分别展示。</p><a href="https://physionet.org/content/mimic-ext-mimic-cxr-vqa/1.0.0/" target="_blank" rel="noreferrer">数据集说明 ↗</a></div></section>
      <section className="panel vqa-browser"><div className="vqa-filters"><label className="vqa-search">搜索<input id="vqa-search" value={filters.q} placeholder="问题、答案、患者 / 检查 / 影像 ID" onChange={e => change("q", e.target.value)} /></label>{[["split", "官方划分", Object.keys(groups)], ["semantic", "题型", types("semantic_types")], ["content", "内容类型", types("content_types")], ["answers", "答案", ["empty", "nonempty"]]].map(([key, label, values]) => <label key={key}>{label}<select aria-label={label} value={filters[key]} onChange={e => change(key, e.target.value)}><option value="all">全部</option>{values.map(value => <option key={value} value={value}>{value === "empty" ? "空答案 []" : value === "nonempty" ? "非空答案" : value}</option>)}</select></label>)}<button onClick={() => { setFilters(defaults); setPage(1); setSelected(null); }}>重置</button></div>
      <div className="vqa-results-heading"><h3>问答目录</h3><span className="muted">{data ? `${fmt(data.total)} 条匹配结果` : "加载中…"}</span></div>
      {error && <div className="notice" role="alert">{error}</div>}
      <div className="table-wrap vqa-table"><table><thead><tr><th>问题 / 参考答案</th><th>题型 / 内容</th><th>患者 / 划分</th><th>查看</th></tr></thead><tbody>{data?.rows.map(row => <tr key={row.id} className={selected?.id === row.id ? "selected" : ""}><td><p>{row.question}</p><Answer values={row.answer} /></td><td>{row.semantic_type}<small>{row.content_type}</small></td><td>{row.subject_id}<small>{row.split} · idx {row.idx}</small></td><td><button aria-label={`查看 ${row.id}`} onClick={() => setSelected(row)}>胸片与问答 ↗</button></td></tr>)}</tbody></table>{data?.total === 0 && <p className="empty-state">没有匹配的问题，请调整筛选条件。</p>}{!data && !error && <p className="empty-state" role="status">正在读取问答…</p>}</div>
      <div className="observation-pagination"><button disabled={page === 1} onClick={() => setPage(page - 1)} aria-label="问答上一页">←</button><span>{page} / {Math.max(1, Math.ceil((data?.total || 0) / 25))}</span><button disabled={!data || page * 25 >= data.total} onClick={() => setPage(page + 1)} aria-label="问答下一页">→</button></div></section>
      {selected && <Detail key={selected.id} row={selected} quality={quality} imagesReady={imagesReady} onClose={() => setSelected(null)} onOpen={onOpen} />}
    </>}
  </div>;
}
