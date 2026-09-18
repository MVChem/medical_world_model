import { fmt } from "../format";

function MiniChart({ title, distribution, unit, insight, note, state }) {
  const { bins = [], total = 0 } = distribution || {};
  const maxCount = Math.max(1, ...bins.map(bin => bin.count));
  return <section className="summary-card interval-chart" aria-label={title}>
    <div className="panel-heading"><h3>{title}</h3><span className="muted">{fmt(total)} {unit}</span></div>
    {!total ? <p className="muted" role="status">{state === "ready" ? "暂无数据" : state === "error" ? "统计暂不可用" : "正在统计…"}</p> : <>
      <p className="interval-insight">{insight}</p>
      <div className="interval-bars" role="list" aria-label={title}>
        {bins.map(bin => {
          const percent = (bin.count / total * 100).toFixed(1);
          const label = `${bin.label}：${fmt(bin.count)} ${unit}，占 ${percent}%`;
          return <div className="interval-row" key={bin.label} role="listitem" aria-label={label} title={label}>
            <span className="interval-label">{bin.label}</span>
            <span className="interval-track"><i style={{ width: `${bin.count / maxCount * 100}%` }} /></span>
            <strong>{percent}%</strong><small>{fmt(bin.count)}</small>
          </div>;
        })}
      </div>
      <p className="interval-footnote">{note}</p>
    </>}
  </section>;
}

export default function IntervalChart({ index }) {
  const all = index.interval_distribution;
  const short = index.short_interval_distribution;
  const studies = index.study_count_distribution;
  return <div className="overview-distributions">
    <MiniChart title="检查间隔分布" distribution={all} unit="对" state={index.state}
      insight={<>中位数 <b>{all?.median_days?.toFixed(2)} 天</b> · 一周内 <b>{all?.within_week_percent?.toFixed(1)}%</b></>}
      note="全部候选配对；间隔 1 小时至 365 天。" />
    <MiniChart title="24 小时内的检查间隔" distribution={short} unit="对" state={index.state}
      insight={<>中位数 <b>{short?.median_hours?.toFixed(1)} 小时</b> · 占全部配对 <b>{all?.total ? (short?.total / all.total * 100).toFixed(1) : "—"}%</b></>}
      note="仅统计 1 ≤ 间隔 < 24 小时；区间含起点、不含终点。" />
    <MiniChart title="每位患者的检查次数" distribution={studies} unit="人" state={index.state}
      insight={<>有多次检查 <b>{fmt(studies?.multiple_count)} 人</b>（{studies?.total ? (studies.multiple_count / studies.total * 100).toFixed(1) : "—"}%）</>}
      note="全部 CXR 患者；按独立检查计数，同次多张影像算一次。" />
  </div>;
}
