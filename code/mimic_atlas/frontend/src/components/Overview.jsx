import { fmt } from "../format";
import IntervalChart from "./IntervalChart";
import DatasetIntro from "./DatasetIntro";

export default function Overview({ catalog, onBrowse }) {
  const index = catalog.cohort || {}, c = index.counts || {};
  return (
    <div id="overview-content">
      <section className="page-heading">
        <div>
          <div className="eyebrow">DATA OVERVIEW</div>
          <h1>数据总览</h1>
          <p>了解 CXR 与 IV 的数据内容、患者覆盖和候选配对分布。</p>
        </div>
        <button className="button" onClick={() => onBrowse("patients")}>浏览全量患者 ↗</button>
      </section>
      <section className="metrics" aria-label="全库数据规模">
        <div>
          <span>全部患者</span>
          <strong id="metric-patients">
            {fmt(c.patients)}
          </strong>
          <small id="metric-patients-note">CXR ∪ IV · 包含仅 IV 患者</small>
        </div>
        <div>
          <span>CXR × IV 患者</span>
          <strong id="metric-matched">{fmt(c.matched_patients)}</strong>
          <small>subject_id 一致</small>
        </div>
        <div>
          <span>影像检查</span>
          <strong id="metric-studies">{fmt(catalog.counts.studies)}</strong>
          <small id="metric-images">{fmt(catalog.counts.images)} 张胸片</small>
        </div>
        <div>
          <span>同住院影像配对</span>
          <strong id="metric-pairs">{fmt(c.linked_pairs)}</strong>
          <small id="metric-pairs-note">{fmt(c.pairs)} 个可用影像配对中</small>
        </div>
      </section>
      <DatasetIntro catalog={catalog} />
      <div
        id="cohort-progress"
        className="index-progress"
        role="status"
        hidden={index.state === "ready" && !index.warnings?.length}
      >
        {index.error ||
          (index.state === "ready" ? index.warnings?.join("；") : index.stage)}
      </div>
      <section className="cohort-summary">
        <div className="summary-card">
          <div className="panel-heading">
            <h3>数据覆盖</h3>
            <span className="muted">点击筛选患者</span>
          </div>
          <div id="coverage-bars">
            {[
              ["matched", "CXR × IV", c.matched_patients],
              ["cxr_only", "仅 CXR", c.cxr_only],
              ["iv_only", "仅 IV", c.iv_only],
            ].map(([key, title, n]) => (
              <button
                className="coverage-row"
                data-coverage={key}
                key={key}
                onClick={() => onBrowse("patients", { coverage: key })}
              >
                <span>{title}</span>
                <span className="coverage-track">
                  <i
                    style={{
                      width:
                        (c.patients
                          ? Math.max(n ? 1 : 0, (n / c.patients) * 100)
                          : 0) + "%",
                    }}
                  />
                </span>
                <b>{fmt(n)}</b>
                <span className="muted">↗</span>
              </button>
            ))}
          </div>
        </div>
        <div className="summary-card">
          <div className="panel-heading">
            <h3>候选配对与数据划分</h3>
            <span className="muted">按官方 CXR split</span>
          </div>
          <div id="split-summary">
            <table className="split-mini">
              <thead>
                <tr>
                  {["Split", "患者", "影像配对", "同住院配对"].map((h) => (
                    <th key={h}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {["train", "validate", "test"].map((k) => (
                  <tr key={k}>
                    <td>
                      <button data-split={k} onClick={() => onBrowse("pairs", { split: k })}>
                        {k}
                      </button>
                    </td>
                    <td>{fmt(index.splits?.[k]?.patients)}</td>
                    <td>{fmt(index.splits?.[k]?.pairs)}</td>
                    <td>{fmt(index.splits?.[k]?.linked_pairs)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>
      <details className="dataset-audit">
        <summary>
          配对规则与数据分布 <span>标签 · 投照体位 · 间隔 · 排除原因</span>
        </summary>
        <div id="dataset-audit-body">
          <div className="audit-rules">
            {Object.values(index.rules || {}).map((v, i) => (
              <p key={i}>{v}</p>
            ))}
          </div>
          <div className="audit-distributions">
            {[
              ["时间间隔", index.horizons],
              ["投照体位 · 影像数", index.views],
              ["阳性标签 · 检查数", index.positive_labels],
              ["配对审计", index.audit],
              ["检查匹配", index.study_linkage],
              ["二元标签变化", index.label_changes],
            ].map(([title, values]) => (
              <div key={title}>
                <h4>{title}</h4>
                {Object.entries(values || {}).map(([k, v]) => (
                  <div className="distribution-row" key={k}>
                    <span>{k}</span>
                    <b>{fmt(v)}</b>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      </details>
      <IntervalChart index={index} />
    </div>
  );
}
