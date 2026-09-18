import { useEffect, useRef, useState } from "react";
import { api, download, saveBlob, useDebounced, useResource } from "../api";
import { fmt, when, time, iso } from "../format";
import { chartHTML, offlineQuery } from "../charts";

export default function ClinicalRecords({
  patient,
  selection,
  snapshot,
  active,
}) {
  const [manifest, setManifest] = useState(
    snapshot?.manifest || patient.extended_tables || [],
  );
  const [table, setTable] = useState("labevents"),
    [scope, setScope] = useState(selection.start ? "window" : "patient");
  const [padding, setPadding] = useState(24),
    [hadm, setHadm] = useState(selection.hadm_id || "");
  const [search, setSearch] = useState(""),
    [series, setSeries] = useState(""),
    [page, setPage] = useState(1);
  const [error, setError] = useState(""),
    [raw, setRaw] = useState(null),
    [retry, setRetry] = useState(0);
  const dialog = useRef(null),
    query = useDebounced(search);
  const endpoint = `/api/patients/${patient.subject_id}/tables`;
  useEffect(() => {
    if (!hadm && selection.hadm_id) setHadm(selection.hadm_id);
  }, [selection.hadm_id, hadm]);
  useEffect(() => {
    if (!active || snapshot) return;
    const controller = new AbortController();
    let timer;
    async function poll() {
      try {
        const data = await api(endpoint, { signal: controller.signal });
        if (controller.signal.aborted) return;
        setManifest(data);
        if (
          data.some((m) =>
            ["queued", "loading", "not_loaded"].includes(m.state),
          )
        )
          timer = setTimeout(poll, 750);
      } catch (e) {
        if (!controller.signal.aborted) {
          setError(e.message);
          timer = setTimeout(poll, 2500);
        }
      }
    }
    poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [endpoint, active, snapshot, retry]);
  const spec = manifest.find((m) => m.name === table);
  const bounds = () =>
    !selection.start
      ? {}
      : snapshot
        ? { start: snapshot.start, end: snapshot.end }
        : {
            start: iso(time(selection.start) - Number(padding) * 3600000),
            end: iso(time(selection.end) + Number(padding) * 3600000),
          };
  const params = new URLSearchParams({
    scope,
    ...bounds(),
    hadm_id: hadm,
    q: query,
    series_id: series,
    page,
    limit: 50,
  });
  const validScope = scope !== "admission" || hadm;
  const response = useResource(
    !snapshot && active && validScope && spec?.state === "ready"
      ? `${endpoint}/${table}?${params}`
      : null,
  );
  useEffect(() => {
    const status = response.data?.status;
    if (!snapshot && status && status.state !== "ready") {
      // The patient's server cache may have expired while this tab was open.
      setManifest((rows) =>
        rows.map((row) => (row.name === table ? { ...row, ...status } : row)),
      );
      setRetry((value) => value + 1);
    }
  }, [response.data, snapshot, table]);
  const state = { snapshot, table, scope, query, series, page, selection };
  const data = snapshot ? offlineQuery(state) : response.data;
  const reset = () => {
    setPage(1);
    setSeries("");
  };
  async function reload(names) {
    try {
      setManifest(
        await api(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ tables: names }),
        }),
      );
      setRetry((n) => n + 1);
      setError("");
    } catch (e) {
      setError(e.message);
    }
  }
  async function exportCSV() {
    try {
      const filename = `mimic-${patient.subject_id}-${table}.csv`;
      if (!snapshot)
        return await download(
          `${endpoint}/${table}/export.csv?${params}`,
          filename,
        );
      const entry = snapshot.tables[table];
      const rows = entry.events.filter(
        (r) =>
          (!series || r.series_id === series) &&
          (!query ||
            [r.label, r.item_id, r.category, ...Object.values(r.raw)]
              .join(" ")
              .toLowerCase()
              .includes(query.toLowerCase())),
      );
      const cell = (v) => '"' + String(v ?? "").replace(/"/g, '""') + '"';
      saveBlob(
        new Blob(
          [
            [
              entry.fields,
              ...rows.map((r) => entry.fields.map((k) => r.raw[k])),
            ]
              .map((r) => r.map(cell).join(","))
              .join("\r\n"),
          ],
          { type: "text/csv" },
        ),
        filename,
      );
    } catch (e) {
      setError(e.message);
    }
  }
  const statuses = {
    ready: "已加载",
    queued: "排队中",
    loading: "读取中",
    unavailable: "源表缺失",
    error: "读取失败",
    not_loaded: "等待自动加载",
  };
  return (
    <>
      <div className="observation-heading">
        <div>
          <div className="eyebrow teal">OBSERVED CLINICAL RECORDS</div>
          <h3>检验、生命体征与其他临床记录</h3>
          <p>
            打开患者后自动加载全部临床记录；按时间、项目与单位浏览，可查看全部原始字段。
          </p>
        </div>
        {!snapshot && manifest.some((m) => m.state === "error") && (
          <button
            id="load-all-tables"
            className="button"
            onClick={() =>
              reload(
                manifest.filter((m) => m.state === "error").map((m) => m.name),
              )
            }
          >
            重试失败的表
          </button>
        )}
      </div>
      <div
        id="observation-error"
        className="notice"
        role="status"
        hidden={!error && !response.error && validScope}
      >
        {error ||
          response.error ||
          "请选择住院号；不会自动推断记录的住院归属。"}
      </div>
      <div className="observation-controls">
        <label>
          数据表
          <select
            id="observation-table"
            value={table}
            onChange={(e) => {
              setTable(e.target.value);
              setSearch("");
              reset();
            }}
          >
            {manifest.map((m) => (
              <option key={m.name} value={m.name}>
                {m.title} · {m.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          记录范围
          <select
            id="observation-scope"
            value={scope}
            disabled={Boolean(snapshot)}
            onChange={(e) => {
              setScope(e.target.value);
              reset();
            }}
          >
            <option value="window" disabled={!selection.start}>
              影像时间窗
            </option>
            <option value="admission">指定住院</option>
            <option value="patient">患者全部记录</option>
          </select>
        </label>
        <label id="padding-label" hidden={scope !== "window"}>
          前后扩展
          <select
            id="observation-padding"
            value={padding}
            disabled={Boolean(snapshot)}
            onChange={(e) => {
              setPadding(Number(e.target.value));
              reset();
            }}
          >
            {[0, 6, 24, 72].map((n) => (
              <option key={n} value={n}>
                {n ? `± ${n} 小时` : "仅片段内"}
              </option>
            ))}
          </select>
        </label>
        <label id="admission-label" hidden={scope !== "admission"}>
          住院号
          <select
            id="observation-admission"
            value={hadm}
            onChange={(e) => {
              setHadm(e.target.value);
              reset();
            }}
          >
            <option value="">选择住院</option>
            {patient.admissions.map((a) => (
              <option key={a.hadm_id} value={a.hadm_id}>
                {a.hadm_id} · {when(a.admittime).slice(0, 10)}
              </option>
            ))}
          </select>
        </label>
        {spec?.state === "error" && !snapshot && (
          <button
            id="load-one-table"
            className="button"
            onClick={() => reload([table])}
          >
            重试此表
          </button>
        )}
      </div>
      <div
        id="observation-status"
        aria-live="polite"
        className="clinical-status"
      >
        {spec?.module}.{table} · {statuses[spec?.state] || "正在读取"} · 患者共{" "}
        {fmt(spec?.count)} 条记录{spec?.error ? ` · ${spec.error}` : ""}
      </div>
      <p id="observation-note" className="table-note">
        {spec?.note}
      </p>
      <div className="observation-filters">
        <label>
          数值指标
          <select
            id="observation-series"
            value={series}
            onChange={(e) => {
              setSeries(e.target.value);
              setPage(1);
            }}
          >
            <option value="">全部指标</option>
            {(data?.series || []).map((s) => (
              <option key={s.id} value={s.id}>
                {s.label} · {s.unit || "单位未记录"} · {s.count} 条 [{s.item_id}
                ]
              </option>
            ))}
          </select>
        </label>
        <label>
          搜索记录
          <input
            id="observation-search"
            type="search"
            placeholder="项目、标本、药物、原始字段…"
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(1);
            }}
          />
        </label>
        <button
          id="observation-csv"
          className="button"
          disabled={data?.status.state !== "ready"}
          onClick={exportCSV}
        >
          导出筛选 CSV ↓
        </button>
      </div>
      {data && validScope && (
        <div id="observation-results">
          <div className="observation-metrics" id="observation-metrics">
            {[
              ["当前范围记录", data.filtered_total],
              ["数值项目 × 单位", data.series.length],
              ["表内患者记录", spec?.count || 0],
            ].map(([label, n]) => (
              <div key={label}>
                <small>{label}</small>
                <strong>{fmt(n)}</strong>
              </div>
            ))}
          </div>
          <div
            id="observation-chart"
            dangerouslySetInnerHTML={{ __html: chartHTML(data, state, bounds) }}
          />
          <div className="observation-records" id="observation-records">
            {data.rows.length ? (
              <table>
                <thead>
                  <tr>
                    {[
                      "事件时刻 / 精度",
                      "项目 / 记录",
                      "原值 · 单位",
                      "住院 / ICU",
                      "标记 / 参考区间",
                      "",
                    ].map((h, i) => (
                      <th key={i}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.rows.map((r, i) => (
                    <tr key={i}>
                      <td>
                        {r.time_precision === "date"
                          ? r.time.slice(0, 10)
                          : when(r.time)}
                        {r.end_time && (
                          <>
                            <br />
                            <span className="muted">→ {when(r.end_time)}</span>
                          </>
                        )}
                        <small>
                          {r.time_precision === "date"
                            ? "仅日期"
                            : r.time_source || "无事件时刻"}
                        </small>
                      </td>
                      <td>
                        {r.label}
                        <small>
                          {[r.item_id, r.fluid, r.category]
                            .filter(Boolean)
                            .join(" · ")}
                        </small>
                        {table === "microbiologyevents" && (
                          <small>
                            {[
                              r.raw.org_name,
                              r.raw.ab_name,
                              r.raw.interpretation,
                            ]
                              .filter(Boolean)
                              .join(" · ")}
                          </small>
                        )}
                      </td>
                      <td className="record-value">
                        {r.value || "未记录"}{" "}
                        <span className="muted">{r.unit}</span>
                      </td>
                      <td>
                        {r.hadm_id || "未关联住院"}
                        <small>{r.stay_id}</small>
                      </td>
                      <td>
                        {r.flag ? (
                          <span className="record-flag">{r.flag}</span>
                        ) : (
                          "—"
                        )}
                        {(r.reference_lower || r.reference_upper) && (
                          <small>
                            {r.reference_lower || "—"} ~{" "}
                            {r.reference_upper || "—"}
                          </small>
                        )}
                      </td>
                      <td>
                        <button
                          className="raw-record"
                          data-record={i}
                          onClick={() => {
                            setRaw(r.raw);
                            dialog.current.showModal();
                          }}
                        >
                          原始 ↗
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="empty-state">
                {data.status.state !== "ready"
                  ? "此表未嵌入离线文件。"
                  : spec?.count
                    ? "当前范围没有匹配记录。可切换“患者全部记录”或清除筛选。"
                    : "此患者在该表中没有记录。"}
              </div>
            )}
          </div>
          <div className="observation-pagination">
            <button
              id="observation-prev"
              disabled={page <= 1}
              onClick={() => setPage(page - 1)}
            >
              ←
            </button>
            <span id="observation-page">
              {page} / {Math.max(1, Math.ceil(data.total / 50))} ·{" "}
              {fmt(data.total)} 条
            </span>
            <button
              id="observation-next"
              disabled={page * 50 >= data.total}
              onClick={() => setPage(page + 1)}
            >
              →
            </button>
          </div>
        </div>
      )}
      <details className="table-inventory">
        <summary id="inventory-title">
          患者 {patient.subject_id} ·{" "}
          {manifest.filter((m) => m.state === "ready").length} /{" "}
          {manifest.length} 张表已加载
        </summary>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>原始表</th>
                <th>状态</th>
                <th>当前患者记录</th>
                <th>全库源文件 · 服务器本地</th>
              </tr>
            </thead>
            <tbody id="inventory-body">
              {manifest.map((m) => (
                <tr key={m.name}>
                  <td>
                    {m.module}.{m.name}
                  </td>
                  <td>{statuses[m.state]}</td>
                  <td>{fmt(m.count)} 条</td>
                  <td>{(m.bytes / 2 ** 20).toFixed(1)} MiB</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="table-note">
          {snapshot
            ? "离线 HTML 仅包含导出时已加载的时间窗记录。"
            : "记录直接按患者索引从原始 CSV 读取；源文件大小是全库大小，不是当前患者的数据量。"}
        </p>
      </details>
      <p className="table-note observation-boundary">
        这些记录用于回顾性浏览。测量时间与录入时间分别保留；处方、实际给药和药房订单分别显示。
      </p>
      <dialog ref={dialog} id="observation-row-dialog">
        <div className="dialog-toolbar">
          <span>
            原始记录 · <span id="raw-table-name">{table}</span>
          </span>
          <button id="close-raw-row" onClick={() => dialog.current.close()}>
            关闭 ×
          </button>
        </div>
        <pre id="raw-row-content">{JSON.stringify(raw, null, 2)}</pre>
      </dialog>
    </>
  );
}
