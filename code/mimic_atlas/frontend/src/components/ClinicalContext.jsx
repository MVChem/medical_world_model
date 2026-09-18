import { useState } from "react";
import { useDebounced, useResource } from "../api";
import { when } from "../format";
import { timelineHTML } from "../charts";

function RawContext({ subject }) {
  const [open, setOpen] = useState(false),
    [table, setTable] = useState("admissions"),
    [search, setSearch] = useState(""),
    [page, setPage] = useState(1);
  const q = useDebounced(search),
    url = `/api/patients/${subject}/context/${table}?${new URLSearchParams({ page, q })}`;
  const { data, error } = useResource(open ? url : null);
  return (
    <details
      className="raw-context"
      onToggle={(e) => setOpen(e.currentTarget.open)}
    >
      <summary>患者全部住院记录 · 原始表</summary>
      <p className="table-note">
        包含片段之外的原始记录；每页 25 条，点击一行展开完整字段。
      </p>
      <div className="raw-context-controls">
        <select
          id="raw-context-table"
          aria-label="原始住院表"
          value={table}
          onChange={(e) => {
            setTable(e.target.value);
            setPage(1);
          }}
        >
          {[
            ["admissions", "住院"],
            ["transfers", "病区转移"],
            ["diagnoses_icd", "诊断 ICD"],
            ["procedures_icd", "操作 ICD"],
            ["icustays", "ICU 住院"],
            ["procedureevents", "ICU 操作"],
            ["inputevents", "ICU 输入 / 用药"],
          ].map(([key, label]) => (
            <option key={key} value={key}>
              {label} · {key}
            </option>
          ))}
        </select>
        <input
          id="raw-context-search"
          type="search"
          placeholder="搜索原始字段"
          value={search}
          onChange={(e) => {
            setSearch(e.target.value);
            setPage(1);
          }}
        />
        <a
          className="button"
          id="raw-context-export"
          href={url + "&download=true"}
        >
          CSV ↓
        </a>
      </div>
      <div id="raw-context-rows">
        {error ? (
          <div className="notice">{error}</div>
        ) : !data ? (
          <div className="empty-state">读取中…</div>
        ) : data.rows.length ? (
          data.rows.map((r, i) => (
            <details key={i} className="raw-context-row">
              <summary>
                <span>{(page - 1) * 25 + i + 1}</span>
                {Object.entries(r)
                  .filter(([k]) => k !== "subject_id")
                  .slice(0, 5)
                  .map(([k, v]) => `${k}: ${v || "—"}`)
                  .join(" · ")}
              </summary>
              <pre>{JSON.stringify(r, null, 2)}</pre>
            </details>
          ))
        ) : (
          <div className="empty-state">没有匹配记录</div>
        )}
      </div>
      <div className="observation-pagination">
        <button
          id="raw-context-prev"
          disabled={page <= 1}
          onClick={() => setPage(page - 1)}
        >
          ←
        </button>
        <span id="raw-context-page">
          {page} / {Math.max(1, Math.ceil((data?.total || 0) / 25))} ·{" "}
          {data?.total ?? "—"} 条
        </span>
        <button
          id="raw-context-next"
          disabled={!data || page * 25 >= data.total}
          onClick={() => setPage(page + 1)}
        >
          →
        </button>
      </div>
    </details>
  );
}
function Codes({ rows }) {
  return rows.length ? (
    <ol>
      {rows.map((r, i) => (
        <li key={i}>
          <code>
            ICD-{r.icd_version} {r.icd_code}
          </code>
          {r.title || "无编码释义"}{" "}
          {r.chartdate && <span className="muted">{r.chartdate}</span>}
        </li>
      ))}
    </ol>
  ) : (
    <div className="muted">没有记录</div>
  );
}
function Events({ context, loaded }) {
  const [search, setSearch] = useState(""),
    [kind, setKind] = useState("all");
  const all = [
    ...context.interval_icu_procedureevents.map((e) => ({
      ...e,
      kind: "procedure",
    })),
    ...(context.interval_icu_inputevents || []).map((e) => ({
      ...e,
      kind: "input",
    })),
  ].sort((a, b) => String(a.starttime).localeCompare(String(b.starttime)));
  const rows = all.filter(
    (e) =>
      (kind === "all" || e.kind === kind) &&
      `${e.label} ${e.category} ${e.itemid}`
        .toLowerCase()
        .includes(search.toLowerCase()),
  );
  return (
    <>
      <div className="clinical-toolbar">
        <h3>片段内 ICU 事件</h3>
        <div>
          <input
            id="event-search"
            type="search"
            placeholder="搜索事件"
            aria-label="搜索临床事件"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <select
            id="event-filter"
            aria-label="事件类型"
            value={kind}
            onChange={(e) => setKind(e.target.value)}
          >
            <option value="all">全部事件</option>
            <option value="procedure">操作 / 支持</option>
            <option value="input">输入 / 用药</option>
          </select>
        </div>
      </div>
      <div id="event-table" className="event-table">
        {rows.length ? (
          <table>
            <thead>
              <tr>
                <th>时间</th>
                <th>
                  事件 · {rows.length} / {all.length}
                </th>
                <th>记录值 / 速率</th>
                <th>时间关系</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((e, i) => (
                <tr key={i}>
                  <td>
                    {when(e.starttime)}
                    <br />
                    <span className="muted">→ {when(e.endtime)}</span>
                  </td>
                  <td>
                    {e.label || e.itemid}
                    <br />
                    <span className="event-type">
                      {e.kind === "input" ? "INPUT" : "PROCEDURE"} · {e.itemid}
                    </span>
                  </td>
                  <td>
                    {(e.kind === "input"
                      ? [e.amount, e.amount_unit]
                      : [e.value, e.value_unit]
                    )
                      .filter(Boolean)
                      .join(" ") || "—"}
                    {e.rate && (
                      <>
                        <br />
                        <span className="muted">
                          {e.rate} {e.rate_unit}
                        </span>
                      </>
                    )}
                  </td>
                  <td>
                    <span className="muted">
                      {e.temporal_role}
                      <br />
                      {e.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="empty-state">已加载事件中没有匹配记录</div>
        )}
      </div>
      <p className="table-note">
        {loaded
          ? "已读取 ICU inputevents；数量和速率保留原始单位。"
          : "ICU inputevents 尚未加载。"}{" "}
        事件为同期观测，不表示治疗导致影像变化。
      </p>
      <div className="clinical-codes">
        <div>
          <h4>
            住院诊断{" "}
            <span className="muted">
              出院编码 · {context.admission_diagnoses.length}
            </span>
          </h4>
          <Codes rows={context.admission_diagnoses} />
        </div>
        <div>
          <h4>
            住院操作{" "}
            <span className="muted">
              日期级精度 · {context.admission_procedures.length}
            </span>
          </h4>
          <Codes rows={context.admission_procedures} />
        </div>
      </div>
    </>
  );
}
export default function ClinicalContext({ patient, pair, studies, offline }) {
  const status = patient.clinical_status,
    context = pair?.retrospective_context_for_audit_only,
    link = pair?.linkage_status,
    studyLink = studies[0]?.admission_link;
  if (status.state !== "ready")
    return (
      <div className="clinical-status">
        {status.state === "loading" && <span className="spinner" />}
        {status.stage}
        <p>
          {status.error ||
            "正在读取当前患者的临床记录。影像、报告和标签可先浏览。"}
        </p>
      </div>
    );
  const names = {
    unique_common_admission: "唯一共同住院",
    unmatched: "未匹配共同住院",
    ambiguous: "存在多条候选住院",
  };
  return (
    <>
      {!patient.studies.length ? (
        <>
          <h3>住院记录 · {patient.admissions.length}</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>住院号</th>
                  <th>入院</th>
                  <th>出院</th>
                  <th>类型</th>
                </tr>
              </thead>
              <tbody>
                {patient.admissions.map((a) => (
                  <tr key={a.hadm_id}>
                    <td>{a.hadm_id}</td>
                    <td>{when(a.admittime)}</td>
                    <td>{when(a.dischtime)}</td>
                    <td>{a.admission_type}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <>
          <div className="clinical-summary">
            <span className={"status-badge " + link}>
              {pair
                ? names[link] || link
                : `单次检查 · ${studyLink?.status === "unique" ? "已匹配住院" : studyLink?.status === "ambiguous" ? "住院匹配有歧义" : "未匹配住院"}`}
            </span>
            {[
              [
                "HADM_ID",
                context?.admission.hadm_id ||
                  (pair ? pair.candidate_hadm_ids : studyLink?.hadm_ids)?.join(
                    ", ",
                  ),
              ],
              ["当前 ICU STAY", context?.source_location?.stay?.stay_id],
              ["当前病区", context?.source_location?.careunit],
              ["随访病区", context?.target_location?.careunit],
            ].map(([k, v]) => (
              <div className="kv" key={k}>
                <small>{k}</small>
                {v || "—"}
              </div>
            ))}
          </div>
          <div
            className="chart-scroll"
            dangerouslySetInnerHTML={{
              __html: timelineHTML(patient, pair, studies),
            }}
          />
          <p className="table-note">
            时间轴按所选胸片窗口裁剪；悬停查看原始区间。缺少区间端点的记录不绘制。
          </p>
          {context ? (
            <Events context={context} loaded={patient.icu_inputs} />
          ) : pair ? (
            <div className="empty-state">
              {link === "ambiguous"
                ? "多个住院区间同时覆盖两次胸片，保留候选记录，不自动选取。"
                : "两次胸片不在同一个可唯一匹配的住院区间内，不拼接临床事件。"}
            </div>
          ) : (
            <p className="table-note">
              选择一个相邻片段，可查看片段内的临床事件、诊断和操作。
            </p>
          )}
        </>
      )}
      {!offline && <RawContext subject={patient.subject_id} />}
    </>
  );
}
