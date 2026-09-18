import { useState } from "react";
import { useDebounced, useResource } from "../api";
import { fmt, hours, when } from "../format";
const defaults = {
  q: "",
  coverage: "all",
  split: "all",
  paired: "all",
  linkage: "all",
  sort: "subject",
  longitudinal: false,
  limit: "25",
};
const Badge = ({ children, matched }) => (
  <span className={"data-badge " + (matched ? "matched" : "")}>{children}</span>
);
export default function Cohort({ catalog, mode, initialFilters, onOpen }) {
  const [filters, setFilters] = useState(() => ({ ...defaults, ...initialFilters })),
    [page, setPage] = useState(1);
  const [pageInput, setPageInput] = useState(null);
  const changePage = (next) => {
    setPage(next);
    setPageInput(null);
  };
  const query = useDebounced(filters.q),
    pairs = mode === "pairs";
  const set = (name, value) => {
    setFilters((f) => ({ ...f, [name]: value }));
    changePage(1);
  };
  const params = new URLSearchParams({
    q: query.trim(),
    split: pairs && filters.split === "none" ? "all" : filters.split,
    page,
    limit: filters.limit,
    ...(pairs
      ? { linkage: filters.linkage }
      : {
          coverage: filters.coverage,
          paired: filters.paired,
          sort: filters.sort,
          longitudinal: filters.longitudinal,
          featured: mode === "featured",
        }),
  });
  const { data, error } = useResource(
    `${pairs ? "/api/cohort/pairs" : "/api/patients"}?${params}`,
    catalog.cohort?.state === "ready" ? 0 : 2500,
  );
  const index = catalog.cohort || {},
    count = data?.total || 0,
    limit = Number(filters.limit),
    pages = Math.max(1, Math.ceil(count / limit));
  const jumpToPage = (event) => {
    event.preventDefault();
    if (!data || !count) return;
    const value = pageInput ?? String(page);
    const requested = Number(value);
    if (!value.trim() || !Number.isFinite(requested)) {
      setPageInput(null);
      return;
    }
    changePage(Math.max(1, Math.min(pages, Math.trunc(requested))));
  };
  const title = pairs ? "影像配对" : mode === "featured" ? "精选示例" : "全量患者";
  const description = pairs
    ? "筛选候选配对，比较同一患者的两次检查、报告与临床记录。"
    : mode === "featured"
      ? "人工挑选的病例与随访片段，用于数据审阅和展示。"
      : "浏览 CXR 与 IV 合并去重后的患者，按数据覆盖和检查情况筛选。";
  const headers = pairs
    ? [
        "患者",
        "当前 → 随访",
        "Split",
        "投照",
        "间隔",
        "标签变化",
        "住院匹配",
        "",
      ]
    : [
        "患者",
        "数据覆盖",
        "Split",
        "检查 / 影像",
        "住院 / ICU",
        "匹配检查",
        "影像配对",
        "同住院配对",
        "",
      ];
  const select = (name, label, choices, hidden = false) => (
    <select
      id={name === "limit" ? "page-size" : name + "-filter"}
      aria-label={label}
      value={filters[name]}
      onChange={(e) => set(name, e.target.value)}
      hidden={hidden}
    >
      {choices.map(([v, text]) => (
        <option key={v} value={v}>
          {text}
        </option>
      ))}
    </select>
  );
  return (
    <div id="cohort-content">
      <section className="page-heading">
        <div>
          <div className="eyebrow">DATA EXPLORER</div>
          <h1>{title}</h1>
          <p>{description}</p>
        </div>
        <span className="subtle-badge">{pairs ? "候选配对目录" : mode === "featured" ? "人工精选 · 非随机样本" : "按患者浏览"}</span>
      </section>
      <div
        id="cohort-progress"
        className="index-progress"
        role="status"
        hidden={index.state === "ready" && !index.warnings?.length}
      >
        {index.error ||
          (index.state === "ready" ? index.warnings?.join("；") : index.stage)}
      </div>
      <section className="cohort-table-panel">
        <div className="cohort-table-heading">
          <div>
            <h2 id="list-title">
              {pairs
                ? "影像配对"
                : mode === "featured"
                  ? "精选示例"
                  : "全量患者"}
            </h2>
            <span id="list-count" className="count-badge">
              {fmt(data?.total)}
            </span>
          </div>
          <a
            id="cohort-export"
            className="button"
            href={`/api/cohort/${pairs ? "pairs" : "patients"}.csv?${params}`}
            download
          >
            导出筛选清单 ↓
          </a>
        </div>
        <div className="cohort-filters">
          <label className="cohort-search">
            <span aria-hidden="true">⌕</span>
            <input
              id="patient-search"
              type="search"
              placeholder="搜索 subject_id"
              aria-label="搜索患者"
              value={filters.q}
              onChange={(e) => set("q", e.target.value)}
            />
          </label>
          {select(
            "coverage",
            "数据覆盖",
            [
              ["all", "全部数据"],
              ["matched", "CXR × IV"],
              ["cxr", "全部 CXR"],
              ["cxr_only", "仅 CXR"],
              ["iv_only", "仅 IV"],
            ],
            pairs,
          )}
          {select("split", "数据划分", [
            ["all", "全部 split"],
            ["train", "Train"],
            ["validate", "Validate"],
            ["test", "Test"],
            ["none", "无 CXR split"],
          ])}
          {select(
            "paired",
            "配对条件",
            [
              ["all", "全部配对状态"],
              ["pairs", "有可用影像配对"],
              ["linked", "有同住院配对"],
            ],
            pairs,
          )}
          {select(
            "linkage",
            "住院匹配",
            [
              ["all", "全部住院匹配"],
              ["unique", "唯一共同住院"],
              ["unmatched", "未匹配"],
              ["ambiguous", "有歧义"],
            ],
            !pairs,
          )}
          {select(
            "sort",
            "排序",
            [
              ["subject", "患者 ID ↑"],
              ["studies", "检查数量 ↓"],
              ["pairs", "配对数量 ↓"],
              ["linked_pairs", "同住院配对 ↓"],
            ],
            pairs,
          )}
          <button
            id="reset-filters"
            title="清空筛选"
            onClick={() => {
              setFilters(defaults);
              changePage(1);
            }}
          >
            重置
          </button>
        </div>
        <div className="list-context">
          <label className="check-line" hidden={pairs}>
            <input
              id="longitudinal-filter"
              type="checkbox"
              checked={filters.longitudinal}
              onChange={(e) => set("longitudinal", e.target.checked)}
            />{" "}
            仅多次检查
          </label>
          <span id="cohort-list-note">
            点击患者或配对，查看影像与完整临床记录。
          </span>
        </div>
        <div
          className="table-wrap cohort-table-wrap"
          id="patient-list"
          aria-live="polite"
        >
          {error ? (
            <div className="notice">{error}</div>
          ) : (
            <table className="cohort-table">
              <thead>
                <tr>
                  {headers.map((h, i) => (
                    <th key={i}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data?.rows.map((r) =>
                  pairs ? (
                    <tr key={r.transition_id}>
                      <td className="mono">{r.subject_id}</td>
                      <td className="mono pair-studies">
                        {r.source_study_id} → {r.target_study_id}
                        <small>{when(r.source_time).slice(0, 16)}</small>
                      </td>
                      <td>
                        <Badge>{r.split}</Badge>
                      </td>
                      <td>{r.view}</td>
                      <td>{hours(r.hours)}</td>
                      <td>{r.label_flips}</td>
                      <td>
                        <Badge matched={r.linkage === "unique"}>
                          {
                            {
                              unique: "唯一共同住院",
                              unmatched: "未匹配",
                              ambiguous: "有歧义",
                            }[r.linkage]
                          }
                        </Badge>
                        <small>{r.hadm_id}</small>
                      </td>
                      <td>
                        <button
                          className="row-open"
                          data-patient={r.subject_id}
                          data-transition={r.transition_id}
                          onClick={() => onOpen(r.subject_id, r.transition_id)}
                        >
                          查看 ↗
                        </button>
                      </td>
                    </tr>
                  ) : (
                    <tr key={r.subject_id}>
                      <td>
                        <button
                          className="patient-link mono"
                          data-patient={r.subject_id}
                          onClick={() => onOpen(r.subject_id)}
                        >
                          {r.subject_id}
                        </button>
                        {r.featured && <span className="featured-star">☆</span>}
                      </td>
                      <td>
                        <Badge matched={r.has_iv && r.has_cxr}>
                          {r.has_iv && r.has_cxr
                            ? "CXR × IV"
                            : r.has_iv
                              ? "IV"
                              : "CXR"}
                        </Badge>
                      </td>
                      <td>{r.split ? <Badge>{r.split}</Badge> : "—"}</td>
                      <td>
                        {fmt(r.studies)}{" "}
                        <span className="muted">/ {fmt(r.images)}</span>
                      </td>
                      <td>
                        {fmt(r.admissions)}{" "}
                        <span className="muted">/ {fmt(r.icu_stays)}</span>
                      </td>
                      <td>{fmt(r.matched_studies)}</td>
                      <td>{fmt(r.pairs)}</td>
                      <td>{fmt(r.linked_pairs)}</td>
                      <td>
                        <button
                          className="row-open"
                          onClick={() => onOpen(r.subject_id)}
                        >
                          查看 ↗
                        </button>
                      </td>
                    </tr>
                  ),
                )}
                {!data?.rows.length && (
                  <tr>
                    <td colSpan={headers.length} className="empty-state">
                      {!data
                        ? "正在读取目录…"
                        : "没有匹配记录，请调整筛选条件。"}
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          )}
        </div>
        <div className="pagination" id="pagination">
          <span id="range-info">
            {count ? fmt((page - 1) * limit + 1) : 0}–
            {fmt(Math.min(page * limit, count))} / {fmt(count)} 条
          </span>
          <div>
            <label>
              每页{" "}
              {select("limit", "每页条数", [
                ["25", "25"],
                ["50", "50"],
                ["100", "100"],
              ])}
            </label>
            <button
              id="page-prev"
              aria-label="上一页"
              disabled={!data || page <= 1}
              onClick={() => changePage(page - 1)}
            >
              ←
            </button>
            <form className="page-jump" onSubmit={jumpToPage} noValidate>
              <span id="page-info">
                <input
                  id="page-input"
                  type="number"
                  inputMode="numeric"
                  min="1"
                  max={pages}
                  step="1"
                  aria-label="页码"
                  title="输入页码，按 Enter 跳转"
                  value={pageInput ?? String(page)}
                  disabled={!data || !count}
                  onChange={(event) => setPageInput(event.target.value)}
                />
                <span> / {data ? pages : "—"}</span>
              </span>
              <button id="page-jump" type="submit" disabled={!data || !count}>
                跳转
              </button>
            </form>
            <button
              id="page-next"
              aria-label="下一页"
              disabled={!data || page >= pages}
              onClick={() => changePage(page + 1)}
            >
              →
            </button>
          </div>
        </div>
      </section>
      <p className="page-foot">
        影像配对为全库可用候选，未应用训练抽样和具体任务约束；不等于最终训练清单。
      </p>
    </div>
  );
}
