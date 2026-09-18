import { useEffect, useRef, useState } from "react";
import { download, saveBlob, useResource } from "../api";
import { hours, labelNames, when } from "../format";
import ImageViewer from "./ImageViewer";
import ClinicalContext from "./ClinicalContext";
import ClinicalRecords from "./ClinicalRecords";

export default function Patient(props) {
  const { data, error } = useResource(
    props.snapshot ? null : `/api/patients/${props.subject}?compact=true`,
  );
  const patient = props.snapshot?.patient || data;
  if (!patient)
    return (
      <div className="startup" role="status">
        {error ? (
          <>
            <p>{error}</p>
            <button onClick={props.onBack}>返回列表</button>
          </>
        ) : (
          <>
            <span className="spinner" />
            <p>正在读取患者 {props.subject}…</p>
          </>
        )}
      </div>
    );
  return <Workspace {...props} initial={patient} />;
}
function Workspace({ initial, initialTransition, snapshot, quality, onBack }) {
  const preferred = initialTransition || initial.preferred_transition;
  const [pairIndex, setPairIndex] = useState(() =>
    Math.max(
      initial.transitions.length ? 0 : -1,
      initial.transitions.findIndex((p) => p.transition_id === preferred),
    ),
  );
  useEffect(() => {
    if (!initialTransition) return;
    const index = initial.transitions.findIndex(
      (p) => p.transition_id === initialTransition,
    );
    if (index >= 0) setPairIndex(index);
  }, [initialTransition, initial.transitions]);
  const [studyIndex, setStudyIndex] = useState(0),
    [tab, setTab] = useState(
      initial.studies.length ? "reports" : "observations",
    );
  const [error, setError] = useState(""),
    [exporting, setExporting] = useState(false);
  const [poll, setPoll] = useState(
    initial.clinical_status.state === "loading" ? 1000 : 0,
  );
  const status = useResource(
    !snapshot && poll
      ? `/api/patients/${initial.subject_id}/clinical-status`
      : null,
    poll,
  );
  const [clinicalState, setClinicalState] = useState(
    initial.clinical_status.state,
  );
  useEffect(() => {
    if (status.data) {
      setClinicalState(status.data.state);
      if (status.data.state !== "loading") setPoll(0);
    }
  }, [status.data]);
  const compactPair = initial.transitions[pairIndex];
  const selectionKey =
    compactPair?.transition_id || initial.studies[studyIndex]?.study_id || "iv";
  const params = new URLSearchParams({
    transition: compactPair?.transition_id || "",
    study: initial.studies[studyIndex]?.study_id || "",
    clinical: clinicalState,
  });
  const response = useResource(
    !snapshot
      ? `/api/patients/${initial.subject_id}/selection?${params}`
      : null,
  );
  // Retain this selection while its clinical background finishes. A different selection never reuses it.
  const [retained, setRetained] = useState(null);
  useEffect(() => {
    if (response.data) setRetained({ key: selectionKey, data: response.data });
  }, [response.data, selectionKey]);
  const detail =
    response.data || (retained?.key === selectionKey ? retained.data : null);
  const patient = snapshot ? initial : { ...initial, ...(detail || {}) };
  const pair = snapshot ? compactPair : detail?.transition || compactPair;
  const packet = pair?.mimic_cxr_transition;
  const studies = snapshot
    ? packet
      ? [packet.current_state.study_id, packet.future_state.study_id].map(
          (id) => initial.studies.find((s) => s.study_id === id),
        )
      : [initial.studies[studyIndex]].filter(Boolean)
    : detail?.studies || [];
  const ready = Boolean(snapshot || detail);
  const selection = {
    start:
      packet?.current_state.image.acquisition_timestamp ||
      studies[0]?.timestamp ||
      null,
    end:
      packet?.future_state.image.acquisition_timestamp ||
      studies[0]?.timestamp ||
      null,
    hadm_id:
      pair?.retrospective_context_for_audit_only?.admission.hadm_id || "",
  };
  const track = useRef(null);
  useEffect(() => {
    const node = track.current?.querySelector(".is-current");
    if (node) {
      const container = track.current;
      container.scrollLeft +=
        node.getBoundingClientRect().left -
        container.getBoundingClientRect().left -
        container.clientWidth / 2 +
        node.clientWidth / 2;
    }
  }, [selectionKey, ready]);
  useEffect(() => {
    const keyboard = (e) => {
      if (
        ["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName) ||
        document.querySelector("dialog[open]")
      )
        return;
      if (e.key === "ArrowRight" && pairIndex < initial.transitions.length - 1)
        setPairIndex(pairIndex + 1);
      if (e.key === "ArrowLeft" && pairIndex > 0) setPairIndex(pairIndex - 1);
    };
    addEventListener("keydown", keyboard);
    return () => removeEventListener("keydown", keyboard);
  }, [pairIndex, initial.transitions.length]);
  async function exportData(format) {
    setExporting(true);
    setError("");
    try {
      if (snapshot)
        saveBlob(
          new Blob([JSON.stringify(patient, null, 2)], {
            type: "application/json",
          }),
          `mimic-${patient.subject_id}.json`,
        );
      else
        await download(
          `/api/export/${patient.subject_id}.${format}?transition=${encodeURIComponent(pair?.transition_id || "")}`,
          format === "html"
            ? `${pair.transition_id}.html`
            : `mimic-${patient.subject_id}.json`,
        );
    } catch (e) {
      setError(e.message);
    } finally {
      setExporting(false);
    }
  }
  const labels = Object.keys(studies[0]?.labels || {});
  const changes = labels.filter(
    (k) =>
      k !== "No Finding" &&
      ["present", "absent"].includes(studies[0].labels[k]) &&
      ["present", "absent"].includes(studies[1]?.labels[k]) &&
      studies[0].labels[k] !== studies[1].labels[k],
  );
  const pill = (value) => (
    <span className={"label-pill " + (value || "not_mentioned")}>
      {labelNames[value] || "未记录"}
    </span>
  );
  const tabs = [
    ["reports", "影像报告"],
    ["labels", "标签变化"],
    ["clinical", "MIMIC-IV 临床背景"],
    ["observations", "检验 · 生命体征 · 更多"],
    ["provenance", "数据来源"],
  ];
  return (
    <article id="patient-content">
      {!snapshot && (
        <button id="back-cohort" className="back-link" onClick={onBack}>
          ← 返回列表
        </button>
      )}
      <section className="patient-header">
        <div>
          <div className="eyebrow">PATIENT WORKSPACE</div>
          <h2>
            Subject <span id="subject-id">{patient.subject_id}</span>
            <span id="split-badge" className="badge">
              {patient.split || "IV only"}
            </span>
          </h2>
          <p id="patient-meta">
            {initial.studies.length
              ? `${initial.studies.length} 次检查 · ${initial.transitions.length} 个可用相邻片段 · ${when(initial.studies[0].timestamp).slice(0, 10)} 至 ${when(initial.studies.at(-1).timestamp).slice(0, 10)}`
              : "MIMIC-IV 患者 · 可浏览全部可用临床数据表"}
          </p>
        </div>
        <div className="export-actions">
          <button
            id="export-json"
            className="button"
            disabled={exporting}
            onClick={() => exportData("json")}
          >
            JSON ↓
          </button>
          <button
            id="export-html"
            className="button primary"
            disabled={
              Boolean(snapshot) ||
              !pair ||
              !ready ||
              patient.clinical_status.state !== "ready" ||
              exporting
            }
            onClick={() => exportData("html")}
          >
            导出当前片段 HTML ↗
          </button>
        </div>
      </section>
      {(error || response.error || status.error) && (
        <div className="notice" role="status">
          {error || response.error || status.error}
        </div>
      )}
      <div
        id="iv-only-note"
        className="notice"
        hidden={Boolean(initial.studies.length)}
      >
        此患者仅有 MIMIC-IV 记录，没有 CXR 检查。下方自动加载全部临床数据表。
      </div>
      {!!initial.studies.length && (
        <>
          <section className="panel study-panel">
            <div className="panel-heading">
              <h3>
                <span className="section-num">01</span> 影像时间线
              </h3>
              <span id="study-count" className="muted">
                {initial.studies.length} STUDIES
              </span>
            </div>
            <div
              ref={track}
              id="study-timeline"
              className="study-timeline"
              aria-label="检查时间线"
            >
              {initial.studies.map((s, i) => (
                <button
                  key={s.study_id}
                  className={
                    "study-node " +
                    (s.study_id === studies[0]?.study_id
                      ? "is-current"
                      : s.study_id === studies[1]?.study_id
                        ? "is-future"
                        : "")
                  }
                  data-study={i}
                  aria-label={"检查 " + s.study_id}
                  onClick={() => {
                    setPairIndex(-1);
                    setStudyIndex(i);
                  }}
                >
                  <div className="study-date">{when(s.timestamp)}</div>
                  <div className="study-id">{s.study_id}</div>
                  <span className="study-view">
                    {[...new Set(s.images.map((im) => im.view))].join(" / ")} ·{" "}
                    {s.images.length} 张影像
                  </span>
                </button>
              ))}
            </div>
            <div className="timeline-footer">
              <span>
                <i className="dot current" /> 当前 <i className="dot future" />{" "}
                随访
              </span>
              <span>
                点击任一检查查看单次影像；配对使用相邻、同投照的 AP / PA。
              </span>
            </div>
          </section>
          <section className="viewer-section">
            <div className="viewer-toolbar">
              <div className="pair-control">
                <button
                  id="pair-prev"
                  aria-label="上一个片段"
                  disabled={pairIndex <= 0}
                  onClick={() => setPairIndex(pairIndex - 1)}
                >
                  ←
                </button>
                <select
                  id="pair-select"
                  aria-label="选择相邻片段"
                  value={pairIndex}
                  onChange={(e) => setPairIndex(Number(e.target.value))}
                >
                  {pairIndex < 0 && (
                    <option value={-1}>
                      {initial.transitions.length
                        ? "单次检查浏览"
                        : "没有符合规则的相邻配对 · 可浏览单次检查"}
                    </option>
                  )}
                  {initial.transitions.map((p, i) => (
                    <option key={p.transition_id} value={i}>
                      {p.mimic_cxr_transition.curation ? "✧ " : ""}
                      {p.mimic_cxr_transition.current_state.study_id} →{" "}
                      {p.mimic_cxr_transition.future_state.study_id} ·{" "}
                      {hours(p.mimic_cxr_transition.interval.elapsed_hours)}
                    </option>
                  ))}
                </select>
                <button
                  id="pair-next"
                  aria-label="下一个片段"
                  disabled={pairIndex >= initial.transitions.length - 1}
                  onClick={() => setPairIndex(pairIndex + 1)}
                >
                  →
                </button>
              </div>
              <div id="interval-badge" className="interval-badge">
                {packet ? (
                  <>
                    间隔 <b>{hours(packet.interval.elapsed_hours)}</b> /{" "}
                    {packet.interval.horizon_bin} · {packet.matched_view}
                  </>
                ) : (
                  "单次检查 · 包含全部投照体位"
                )}
              </div>
            </div>
            {ready ? (
              <ImageViewer
                key={selectionKey}
                patient={patient}
                studies={studies}
                packet={packet}
                quality={quality}
                offline={Boolean(snapshot)}
              />
            ) : (
              <div className="selection-loading">正在读取所选检查…</div>
            )}
          </section>
        </>
      )}
      <section className="panel details-panel">
        <div className="tabs" role="tablist" aria-label="病例详情">
          {tabs.map(([key, title]) => (
            <button
              role="tab"
              key={key}
              id={"tab-" + key}
              aria-selected={tab === key}
              aria-controls={key + "-panel"}
              data-tab={key}
              onClick={() => setTab(key)}
            >
              {title}
              {key === "labels" && (
                <span id="change-count">{changes.length}</span>
              )}
              {key === "clinical" && (
                <i
                  id="clinical-dot"
                  className={"status-dot " + patient.clinical_status.state}
                />
              )}
            </button>
          ))}
        </div>
        <div
          id="reports-panel"
          className="tab-panel"
          role="tabpanel"
          aria-labelledby="tab-reports"
          hidden={tab !== "reports"}
        >
          {!ready ? (
            <div className="empty-state">正在读取所选检查…</div>
          ) : (
            <>
              <div className="reports-grid">
                {studies.map((s, i) => {
                  const sections = [
                    ["findings", "FINDINGS"],
                    ["impression", "IMPRESSION"],
                    ["unsectioned_report", "REPORT"],
                  ].filter(
                    ([k]) =>
                      s.report?.[k] &&
                      (k !== "unsectioned_report" ||
                        (!s.report.findings && !s.report.impression)),
                  );
                  return (
                    <div key={s.study_id}>
                      <div className={"report-head " + (i ? "future" : "")}>
                        <i className={"dot " + (i ? "future" : "current")} />
                        {i ? "随访报告 · 评估用" : "当前报告"}
                        <span className="muted">{s.study_id}</span>
                      </div>
                      {sections.length ? (
                        sections.map(([k, title]) => (
                          <div className="report-section" key={k}>
                            <h4>{title}</h4>
                            <p
                              className={k === "impression" ? "impression" : ""}
                            >
                              {s.report[k]}
                            </p>
                          </div>
                        ))
                      ) : (
                        <div className="empty-state">
                          该检查未提供可读报告。
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
              {packet?.curation && (
                <div className="audit-note">
                  <b>精选片段备注</b>
                  {packet.curation.audit_note}
                  <p>按报告和随访挑选的演示案例，非随机队列或临床裁定。</p>
                </div>
              )}
            </>
          )}
        </div>
        <div
          id="labels-panel"
          className="tab-panel"
          role="tabpanel"
          aria-labelledby="tab-labels"
          hidden={tab !== "labels"}
        >
          <div className="legend">
            {Object.keys(labelNames).map((k) => (
              <span key={k}>{pill(k)}</span>
            ))}
            <span>空值保留为“未提及”，不当作阴性。</span>
          </div>
          <div className="table-wrap">
            <table className="label-table">
              <thead>
                <tr>
                  <th>CHEXPERT FINDING</th>
                  <th>当前</th>
                  {studies[1] && (
                    <>
                      <th>随访</th>
                      <th>变化</th>
                    </>
                  )}
                </tr>
              </thead>
              <tbody>
                {labels.map((k) => (
                  <tr
                    key={k}
                    className={changes.includes(k) ? "changed-row" : ""}
                  >
                    <td>{k}</td>
                    <td>{pill(studies[0].labels[k])}</td>
                    {studies[1] && (
                      <>
                        <td>{pill(studies[1].labels[k])}</td>
                        <td>
                          {changes.includes(k)
                            ? studies[1].labels[k] === "present"
                              ? "阴性 → 阳性"
                              : "阳性 → 阴性"
                            : studies[0].labels[k] === studies[1].labels[k]
                              ? "相同"
                              : "标注状态变化"}
                        </td>
                      </>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="table-note">
            报告自动标注的变化不等同于经验证的疾病发生、消退或严重程度变化。
          </p>
        </div>
        <div
          id="clinical-panel"
          className="tab-panel"
          role="tabpanel"
          aria-labelledby="tab-clinical"
          hidden={tab !== "clinical"}
        >
          {ready && (
            <ClinicalContext
              key={selectionKey}
              patient={patient}
              pair={pair}
              studies={studies}
              offline={Boolean(snapshot)}
            />
          )}
        </div>
        <div
          id="observations-panel"
          className="tab-panel"
          role="tabpanel"
          aria-labelledby="tab-observations"
          hidden={tab !== "observations"}
        >
          {ready && (
            <ClinicalRecords
              key={selectionKey}
              patient={patient}
              selection={selection}
              snapshot={snapshot?.extended}
              active={tab === "observations"}
            />
          )}
        </div>
        <div
          id="provenance-panel"
          className="tab-panel"
          role="tabpanel"
          aria-labelledby="tab-provenance"
          hidden={tab !== "provenance"}
        >
          <div className="table-wrap">
            <table className="provenance-table">
              <thead>
                <tr>
                  <th>数据</th>
                  <th>来源与连接方式</th>
                  <th>使用范围</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>影像 / study_id / dicom_id</td>
                  <td>MIMIC-CXR-JPG 2.0.0</td>
                  <td>当前影像为输入；随访为观测目标</td>
                </tr>
                <tr>
                  <td>报告 / CheXpert / split</td>
                  <td>原始报告、官方标签和患者级划分</td>
                  <td>阳性、阴性、不确定、未提及</td>
                </tr>
                <tr>
                  <td>临床记录</td>
                  <td>MIMIC-IV 3.1 · subject_id 严格一致</td>
                  <td>回顾性背景；按患者索引读取原始 CSV</td>
                </tr>
                <tr>
                  <td>影像配对</td>
                  <td>相邻检查，同一 AP / PA 投照；1 小时至 365 天</td>
                  <td>完整保留患者所有检查；配对为可选视图</td>
                </tr>
                <tr>
                  <td>住院匹配</td>
                  <td>同一患者，采集时刻包含于住院区间</td>
                  <td>有歧义时保留候选住院</td>
                </tr>
              </tbody>
            </table>
          </div>
          <details className="audit-note">
            <summary>查看配对筛选计数</summary>
            <pre>{JSON.stringify(initial.pairing_audit, null, 2)}</pre>
          </details>
          <p className="table-note">
            study_id 和 hadm_id
            不是关联键。缺失影像、报告或不符合配对规则的检查仍保留在患者时间线上。
          </p>
        </div>
      </section>
      <p className="page-foot">
        研究数据浏览 · CheXpert 为报告衍生标签；随访和 IV
        回顾性记录不作为当前状态预测输入。
      </p>
    </article>
  );
}
