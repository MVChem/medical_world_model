/* Separate clinical-table browser; also runs from the embedded offline snapshot. */
(() => {
  "use strict";
  const esc = (v) =>
    String(v ?? "").replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );
  const $ = (id) => document.getElementById(id);
  const fmt = (n) => Number(n).toLocaleString("en-US");
  const when = (s) =>
    s ? String(s).replace("T", " ").slice(0, 19) : "无事件时刻";
  const time = (s) =>
    s ? Date.parse(String(s).replace(" ", "T").replace(/Z$/, "") + "Z") : NaN;
  const iso = (n) => new Date(n).toISOString().replace(/Z$/, "");
  const statuses = {
    not_loaded: "未加载",
    queued: "排队中",
    loading: "服务器筛选中",
    ready: "已加载",
    unavailable: "源表缺失",
    error: "加载失败",
  };
  const state = {
    active: false,
    patient: null,
    selection: null,
    snapshot: null,
    manifest: [],
    table: "labevents",
    scope: "window",
    hadm: "",
    padding: 24,
    query: "",
    series: "",
    page: 1,
    token: 0,
    generation: 0,
    timer: null,
    debounce: null,
    data: null,
  };

  async function api(url, options = {}) {
    const response = await fetch(url, { cache: "no-store", ...options });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(
        typeof body.detail === "string"
          ? body.detail
          : `请求失败 (${response.status})`,
      );
    }
    return response.json();
  }
  const endpoint = () => `/api/patients/${state.patient.subject_id}/tables`;
  const spec = () => state.manifest.find((s) => s.name === state.table);
  function bounds() {
    if (!state.selection?.start) return {};
    return state.snapshot
      ? { start: state.snapshot.start, end: state.snapshot.end }
      : {
          start: iso(time(state.selection.start) - state.padding * 3600000),
          end: iso(time(state.selection.end) + state.padding * 3600000),
        };
  }
  function filters() {
    return {
      scope: state.scope,
      ...bounds(),
      hadm_id: state.hadm,
      q: state.query,
      series_id: state.series,
      page: state.page,
      limit: 50,
    };
  }
  function error(message) {
    $("observation-error").textContent = message;
    $("observation-error").hidden = !message;
  }
  function build() {
    const p = state.patient;
    $("observations-panel").innerHTML =
      `<div class="observation-heading"><div><div class="eyebrow teal">OBSERVED CLINICAL RECORDS</div><h3>检验、生命体征与其他临床记录</h3><p>按原始时间、项目与单位浏览；可查看每条记录的全部字段。</p></div><button id="load-all-tables" class="button">加载全部 ${state.manifest.length} 张扩展表</button></div>
      <div id="observation-error" class="notice" hidden role="status"></div>
      <div class="observation-controls"><label>数据表<select id="observation-table">${state.manifest.map((s) => `<option value="${s.name}">${esc(s.title)} · ${s.name}</option>`).join("")}</select></label><label>记录范围<select id="observation-scope"><option value="window">影像时间窗</option><option value="admission">指定住院</option><option value="patient">患者全部记录</option></select></label><label id="padding-label">前后扩展<select id="observation-padding"><option value="0">仅片段内</option><option value="6">± 6 小时</option><option value="24">± 24 小时</option><option value="72">± 72 小时</option></select></label><label id="admission-label" hidden>住院号<select id="observation-admission"><option value="">选择住院</option>${p.admissions.map((a) => `<option value="${esc(a.hadm_id)}">${esc(a.hadm_id)} · ${when(a.admittime).slice(0, 10)}</option>`).join("")}</select></label><button id="load-one-table" class="button primary">加载此表</button></div>
      <div id="observation-status" aria-live="polite"></div><p id="observation-note" class="table-note"></p>
      <div id="observation-results" hidden><div class="observation-metrics" id="observation-metrics"></div><div class="observation-filters"><label>数值指标<select id="observation-series"><option value="">全部指标</option></select></label><label>搜索记录<input id="observation-search" type="search" placeholder="项目、标本、药物、原始字段…"></label><button id="observation-csv" class="button">导出筛选 CSV ↓</button></div><div id="observation-chart"></div><div class="observation-records" id="observation-records"></div><div class="observation-pagination"><button id="observation-prev" aria-label="上一页记录">←</button><span id="observation-page"></span><button id="observation-next" aria-label="下一页记录">→</button></div></div>
      <details class="table-inventory"><summary id="inventory-title">扩展数据表覆盖与加载进度</summary><div class="table-wrap"><table><thead><tr><th>原始表</th><th>状态</th><th>${state.snapshot ? "离线窗口记录" : "当前患者记录"}</th><th>全库源文件 · 服务器本地</th></tr></thead><tbody id="inventory-body"></tbody></table></div><p class="table-note">${state.snapshot ? "此 HTML 仅包含导出时已加载的时间窗记录，不能继续读取服务器索引。" : state.manifest.some((m) => m.indexed) ? "全部患者已提前建立索引，记录数来自患者目录。点击后只读取该患者所在的数据块，无需重新扫描全库。" : "源文件包含所有患者；首次加载由服务器扫描整表、筛选当前患者。"}源文件大小不是该患者的数据量，也不是下载量；页面仅传输当前页和绘图数据。</p></details>
      <p class="table-note observation-boundary">这些记录用于回顾性浏览。测量时间与录入时间分别保留；处方、实际给药、药房订单分别显示，所有数值按原始项目和单位分组。</p>
      <dialog id="observation-row-dialog"><div class="dialog-toolbar"><span>原始记录 · <span id="raw-table-name"></span></span><button id="close-raw-row">关闭 ×</button></div><pre id="raw-row-content"></pre></dialog>`;
    $("observation-table").value = state.table;
    $("observation-scope").value = state.scope;
    $("observation-padding").value = String(state.padding);
    $("observation-admission").value = state.hadm;
    $("observation-search").value = state.query;
    $("observation-table").addEventListener("change", (e) => {
      state.table = e.target.value;
      state.series = "";
      state.page = 1;
      renderStatus();
      refresh();
    });
    $("observation-scope").addEventListener("change", (e) => {
      state.scope = e.target.value;
      state.page = 1;
      state.series = "";
      updateScope();
      refresh();
    });
    $("observation-padding").addEventListener("change", (e) => {
      state.padding = Number(e.target.value);
      state.page = 1;
      refresh();
    });
    $("observation-admission").addEventListener("change", (e) => {
      state.hadm = e.target.value;
      state.page = 1;
      refresh();
    });
    $("observation-series").addEventListener("change", (e) => {
      state.series = e.target.value;
      state.page = 1;
      refresh();
    });
    $("observation-search").addEventListener("input", (e) => {
      state.query = e.target.value;
      clearTimeout(state.debounce);
      ++state.token;
      state.debounce = setTimeout(() => {
        state.page = 1;
        refresh();
      }, 250);
    });
    $("observation-prev").addEventListener("click", () => {
      clearTimeout(state.debounce);
      state.page--;
      refresh();
    });
    $("observation-next").addEventListener("click", () => {
      clearTimeout(state.debounce);
      state.page++;
      refresh();
    });
    $("load-one-table").addEventListener("click", () => load([state.table]));
    $("load-all-tables").addEventListener("click", () => load(["all"]));
    $("observation-csv").addEventListener("click", downloadCSV);
    $("observation-records").addEventListener("click", (e) => {
      const button = e.target.closest("[data-record]");
      if (!button) return;
      const row = state.data.rows[Number(button.dataset.record)];
      $("raw-table-name").textContent = `${spec().module}.${state.table}`;
      $("raw-row-content").textContent = JSON.stringify(
        {
          derived_display: {
            time: row.time,
            time_precision: row.time_precision,
            time_source: row.time_source,
            hadm_id: row.hadm_id,
            parent_link: row.parent_link,
          },
          original: row.raw,
        },
        null,
        2,
      );
      $("observation-row-dialog").showModal();
    });
    $("close-raw-row").addEventListener("click", () =>
      $("observation-row-dialog").close(),
    );
    if (state.snapshot) {
      $("load-all-tables").hidden = true;
      $("load-one-table").hidden = true;
      $("observation-scope").disabled = true;
      $("observation-padding").disabled = true;
    }
    $("observation-scope").querySelector('option[value="window"]').disabled =
      !state.selection.start;
    updateScope();
    renderStatus();
    refresh();
    poll();
  }
  function updateScope() {
    $("padding-label").hidden = state.scope !== "window";
    $("admission-label").hidden = state.scope !== "admission";
  }
  function renderStatus() {
    const s = spec();
    if (!s) return;
    $("load-one-table").disabled =
      Boolean(state.snapshot) ||
      ["ready", "loading", "queued", "unavailable"].includes(s.state);
    $("load-one-table").textContent =
      s.state === "error" ? "重试此表" : "加载此表";
    $("observation-note").textContent =
      s.note || "保留原始记录；缺少时间的记录可在“患者全部记录”中查看。";
    const active = state.manifest.filter((m) =>
      ["queued", "loading"].includes(m.state),
    ).length;
    const ready = state.manifest.filter((m) => m.state === "ready").length;
    $("inventory-title").textContent =
      `患者 ${state.patient.subject_id} · ${state.manifest.some((m) => m.indexed) ? "全局索引已就绪 · " : ""}${ready} / ${state.manifest.length} 张表已加载${active ? ` · ${active} 张正在读取或排队` : ""}`;
    $("inventory-body").innerHTML = state.manifest
      .map(
        (m) =>
          `<tr><td>${esc(m.module)}.${esc(m.name)}</td><td>${state.snapshot ? (state.snapshot.tables[m.name] ? "已嵌入时间窗" : "未嵌入") : m.indexed && m.state === "not_loaded" ? "已建索引 · 按需读取" : m.indexed && m.state === "loading" ? "读取患者数据" : esc(statuses[m.state])}${m.state === "loading" && !state.snapshot && !m.indexed ? ` · ${m.percent}%` : ""}</td><td>${state.snapshot ? (state.snapshot.tables[m.name] ? fmt(state.snapshot.tables[m.name].events.length) : "未嵌入") : m.count == null ? "待筛选" : fmt(m.count) + " 条"}</td><td>${(m.bytes / 1024 / 1024).toFixed(1)} MiB</td></tr>`,
      )
      .join("");
    let message = "";
    if (state.snapshot)
      message = `离线时间窗：${when(state.snapshot.start)} → ${when(state.snapshot.end)}；仅含导出时已加载的表。`;
    else if (s.state === "loading")
      message = s.indexed ? `正在按索引读取患者 ${state.patient.subject_id} 的 ${s.name} 记录。` : `服务器正在从全库 ${s.name} 筛选患者 ${state.patient.subject_id} · 扫描进度 ${s.percent || 0}%（不是下载进度）`;
    else if (s.state === "queued")
      message = `${s.name} 已排队；其他影像与报告可继续浏览。`;
    else if (s.state === "not_loaded")
      message =
        s.indexed ? `索引已就绪：该患者共 ${fmt(s.count || 0)} 条记录。点击“加载此表”读取，浏览器按页显示。` : "此表尚未加载。点击“加载此表”由服务器筛选当前患者的记录；浏览器按页读取结果。";
    else if (s.state === "unavailable") message = "本地未找到此源表。";
    else if (s.state === "error") message = `加载失败：${s.error}`;
    else
      message = `${s.module}.${s.name} · 该患者 ${fmt(s.count || 0)} 条记录 · ${fmt(s.missing_hadm || 0)} 条无住院号 · ${fmt(s.undated || 0)} 条无事件时刻`;
    $("observation-status").innerHTML =
      `<div class="observation-state ${esc(s.state)}"><span>${esc(message)}</span>${!state.snapshot && s.state === "loading" ? `<progress max="100" value="${s.percent || 0}"></progress>` : ""}</div>`;
  }
  async function load(tables) {
    const generation = state.generation;
    try {
      const result = await api(endpoint(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tables }),
      });
      if (generation !== state.generation) return;
      state.manifest = result;
      error("");
      renderStatus();
      refresh();
      poll();
    } catch (e) {
      if (generation === state.generation) error(e.message);
    }
  }
  function poll() {
    clearTimeout(state.timer);
    if (
      !state.active ||
      state.snapshot ||
      !state.manifest.some((s) => ["queued", "loading"].includes(s.state))
    )
      return;
    const generation = state.generation;
    state.timer = setTimeout(async () => {
      try {
        const before = spec()?.state;
        const result = await api(endpoint());
        if (generation !== state.generation) return;
        state.manifest = result;
        renderStatus();
        if (before !== spec()?.state) refresh();
        poll();
      } catch (e) {
        if (generation === state.generation) {
          error(e.message);
          poll();
        }
      }
    }, 2500);
  }
  function offlineQuery() {
    const entry = state.snapshot.tables[state.table];
    if (!entry)
      return {
        status: { state: "not_loaded" },
        rows: [],
        series: [],
        total: 0,
        filtered_total: 0,
        chart: { points: [], total: 0, sampled: false },
        fields: [],
      };
    const all = entry.events.filter(
      (r) =>
        !state.query ||
        [r.label, r.item_id, r.category, ...Object.values(r.raw)]
          .join(" ")
          .toLowerCase()
          .includes(state.query.toLowerCase()),
    );
    const grouped = new Map();
    for (const r of all) {
      if (r.numeric_value == null || r.time_precision !== "timestamp") continue;
      const s = grouped.get(r.series_id) || {
        id: r.series_id,
        label: r.label,
        item_id: r.item_id,
        unit: r.unit,
        category: r.category,
        count: 0,
      };
      s.count++;
      grouped.set(r.series_id, s);
    }
    const series = [...grouped.values()].sort((a, b) =>
      a.label.localeCompare(b.label),
    );
    const selected = state.series || series[0]?.id || "";
    const rows = state.series
      ? all.filter((r) => r.series_id === state.series)
      : all;
    // Offline exports retain every observation. The plot uses points, never interpolation.
    const points = all.filter(
      (r) =>
        r.series_id === selected &&
        r.numeric_value != null &&
        r.time_precision === "timestamp",
    );
    return {
      status: { state: "ready" },
      rows: rows.slice((state.page - 1) * 50, state.page * 50),
      series,
      selected_series: selected,
      total: rows.length,
      filtered_total: all.length,
      chart: { points, total: points.length, sampled: false },
      fields: entry.fields,
    };
  }
  async function refresh() {
    if (!state.active) return;
    const token = ++state.token,
      generation = state.generation;
    if (state.scope === "admission" && !state.hadm) {
      error("请选择住院号；不会将没有住院号的记录自动归入住院。");
      $("observation-results").hidden = true;
      return;
    }
    if (!state.snapshot && spec()?.state !== "ready") {
      $("observation-results").hidden = true;
      return;
    }
    try {
      const data = state.snapshot
        ? offlineQuery()
        : await api(
            `${endpoint()}/${state.table}?${new URLSearchParams(filters())}`,
          );
      if (token !== state.token || generation !== state.generation) return;
      state.data = data;
      error("");
      renderData();
    } catch (e) {
      if (token === state.token && generation === state.generation) {
        error(e.message);
        $("observation-results").hidden = true;
      }
    }
  }
  function renderData() {
    const d = state.data;
    $("observation-results").hidden = false;
    $("observation-metrics").innerHTML =
      `<div><small>当前范围记录</small><strong>${fmt(d.filtered_total)}</strong></div><div><small>数值项目 × 单位</small><strong>${fmt(d.series.length)}</strong></div><div><small>表内患者记录</small><strong>${fmt(spec().count || 0)}</strong></div>`;
    $("observation-series").innerHTML =
      '<option value="">全部指标</option>' +
      d.series
        .map(
          (s) =>
            `<option value="${s.id}">${esc(s.label)} · ${esc(s.unit || "单位未记录")} · ${s.count} 条 [${esc(s.item_id)}]</option>`,
        )
        .join("");
    $("observation-series").value = state.series;
    if (state.series && !d.series.some((s) => s.id === state.series))
      $("observation-series").insertAdjacentHTML(
        "beforeend",
        `<option value="${esc(state.series)}" selected>所选指标在当前范围无数值</option>`,
      );
    drawChart(d);
    $("observation-records").innerHTML = d.rows.length
      ? `<table><thead><tr><th>事件时刻 / 精度</th><th>项目 / 记录</th><th>原值 · 单位</th><th>住院 / ICU</th><th>标记 / 参考区间</th><th></th></tr></thead><tbody>${d.rows.map((r, i) => `<tr><td>${esc(r.time_precision === "date" ? r.time.slice(0, 10) : when(r.time))}${r.end_time ? `<br><span class="muted">→ ${esc(when(r.end_time))}</span>` : ""}<small>${r.time_precision === "date" ? "仅日期" : r.time_source ? esc(r.time_source) : "无事件时刻"}</small></td><td>${esc(r.label)}<small>${esc([r.item_id, r.fluid, r.category].filter(Boolean).join(" · "))}</small>${state.table === "microbiologyevents" ? `<small>${esc([r.raw.org_name, r.raw.ab_name, r.raw.interpretation].filter(Boolean).join(" · "))}</small>` : ""}</td><td class="record-value">${esc(r.value || "未记录")} <span class="muted">${esc(r.unit)}</span></td><td>${esc(r.hadm_id || "未关联住院")}<small>${esc(r.stay_id || "")}</small></td><td>${r.flag ? `<span class="record-flag">${esc(r.flag)}</span>` : "—"}${r.reference_lower || r.reference_upper ? `<small>${esc(r.reference_lower || "—")} ~ ${esc(r.reference_upper || "—")}</small>` : ""}</td><td><button class="raw-record" data-record="${i}" aria-label="查看第 ${i + 1} 条原始记录">原始 ↗</button></td></tr>`).join("")}</tbody></table>`
      : `<div class="empty-state">${d.status.state !== "ready" ? (state.snapshot ? "此表未嵌入离线文件。" : "此表尚未读取，点击“加载此表”查看记录。") : spec().count ? "当前范围没有匹配记录。可切换“患者全部记录”或清除筛选。" : "此患者在该表中没有记录。"}</div>`;
    $("observation-page").textContent =
      `${state.page} / ${Math.max(1, Math.ceil(d.total / 50))} · ${fmt(d.total)} 条`;
    $("observation-prev").disabled = state.page <= 1;
    $("observation-next").disabled = state.page * 50 >= d.total;
    $("observation-csv").disabled = d.status.state !== "ready";
  }
  function drawChart(data) {
    const chart = data.chart,
      series = data.series.find((s) => s.id === data.selected_series),
      points = chart.points;
    if (!points.length) {
      $("observation-chart").innerHTML =
        '<div class="empty-state">当前记录没有可绘图的数值与时间。文本、范围值、日期级及无时间记录保留在明细中。</div>';
      return;
    }
    const times = points.map((p) => time(p.time)),
      values = points.map((p) => p.numeric_value);
    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity;
    times.forEach((v) => {
      minX = Math.min(minX, v);
      maxX = Math.max(maxX, v);
    });
    values.forEach((v) => {
      minY = Math.min(minY, v);
      maxY = Math.max(maxY, v);
    });
    if (state.scope === "window") {
      const b = bounds();
      minX = time(b.start);
      maxX = time(b.end);
    }
    if (maxX === minX) {
      minX -= 3600000;
      maxX += 3600000;
    }
    const padding = Math.max((maxY - minY) * 0.12, Math.abs(maxY) * 0.02, 0.1);
    minY -= padding;
    maxY += padding;
    const x = (n) => 70 + ((n - minX) / (maxX - minX)) * 790,
      y = (n) => 238 - ((n - minY) / (maxY - minY)) * 175;
    let svg = `<svg viewBox="0 0 910 290" class="observation-svg" role="img" aria-label="${esc(series?.label)}，${points.length} 个观测点"><text x="70" y="26" font-size="13" fill="#315563">${esc(series?.label)} <tspan fill="#8598a1" font-size="10">${esc(series?.unit || "单位未记录")} · ${esc(series?.item_id || "")}</tspan></text>`;
    for (let i = 0; i <= 4; i++) {
      const v = minY + ((maxY - minY) * i) / 4;
      svg += `<line x1="70" x2="860" y1="${y(v)}" y2="${y(v)}" stroke="#e8eff1"/><text x="60" y="${y(v) + 3}" text-anchor="end" font-size="9" fill="#869ba6">${Number(v.toPrecision(4))}</text>`;
      const t = minX + ((maxX - minX) * i) / 4,
        d = iso(t);
      svg += `<text x="${x(t)}" y="260" text-anchor="middle" font-size="9" fill="#8297a1">${d.slice(5, 10)}</text><text x="${x(t)}" y="275" text-anchor="middle" font-size="9" fill="#a0aeb5">${d.slice(11, 16)}</text>`;
    }
    [state.selection.start, state.selection.end].forEach((at, i) => {
      const t = time(at);
      if (
        !Number.isFinite(t) ||
        t < minX ||
        t > maxX ||
        (i && at === state.selection.start)
      )
        return;
      svg += `<line x1="${x(t)}" x2="${x(t)}" y1="43" y2="239" stroke="${i ? "#b48a53" : "#198b88"}" stroke-dasharray="4 4"/><text x="${x(t)}" y="49" font-size="9" text-anchor="middle" fill="${i ? "#b48a53" : "#198b88"}">${i ? "随访 T₁" : "当前 T₀"}</text>`;
    });
    points.forEach((p, i) => {
      const flagged = p.flag && p.flag !== "0";
      svg += `<circle data-point="${i}" cx="${x(times[i])}" cy="${y(p.numeric_value)}" r="${points.length > 800 ? 2 : 3.5}" fill="${flagged ? "#c48150" : "#218e91"}" fill-opacity=".75"><title>${esc(`${when(p.time)}\n${p.value} ${p.unit}\n录入: ${when(p.storetime)}\n标记: ${p.flag || "无"}\n参考: ${p.reference_lower || "—"} ~ ${p.reference_upper || "—"}`)}</title></circle>`;
    });
    $("observation-chart").innerHTML =
      `<div class="observation-chart-wrap">${svg}</svg></div><div class="chart-caption"><span>${fmt(chart.total)} 个原始数值点${chart.sampled ? ` · 绘图保留 ${points.length} 个分段极值与端点` : ""} · 悬停查看原始值和录入时间</span><span>仅显示观测点，不插值、不换算单位</span></div>`;
  }
  async function downloadCSV() {
    try {
      let blob;
      if (state.snapshot) {
        const entry = state.snapshot.tables[state.table];
        const rows = entry.events.filter(
          (r) =>
            (!state.series || r.series_id === state.series) &&
            (!state.query ||
              [r.label, r.item_id, r.category, ...Object.values(r.raw)]
                .join(" ")
                .toLowerCase()
                .includes(state.query.toLowerCase())),
        );
        const cell = (v) => '"' + String(v ?? "").replace(/"/g, '""') + '"';
        blob = new Blob(
          [
            [
              entry.fields,
              ...rows.map((r) => entry.fields.map((k) => r.raw[k])),
            ]
              .map((r) => r.map(cell).join(","))
              .join("\r\n"),
          ],
          { type: "text/csv" },
        );
      } else {
        const response = await fetch(
          `${endpoint()}/${state.table}/export.csv?${new URLSearchParams(filters())}`,
        );
        if (!response.ok) throw new Error("导出失败");
        blob = await response.blob();
      }
      const a = document.createElement("a"),
        url = URL.createObjectURL(blob);
      a.href = url;
      a.download = `mimic-${state.patient.subject_id}-${state.table}.csv`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
    } catch (e) {
      error(e.message);
    }
  }
  window.ClinicalExplorer = {
    pause() {
      state.patient = null;
      state.key = null;
      state.manifest = [];
      state.active = false;
      clearTimeout(state.timer);
      clearTimeout(state.debounce);
      state.token++;
      state.generation++;
    },
    setActive(active) {
      const changed = state.active !== active;
      state.active = active;
      if (!active) {
        clearTimeout(state.timer);
        return;
      }
      if (changed && state.patient) {
        refresh();
        poll();
      }
    },
    update({ patient, selection, snapshot }) {
      const key = JSON.stringify([
        patient.subject_id,
        selection.start,
        selection.end,
      ]);
      const same = key === state.key;
      const sameSubject = patient.subject_id === state.patient?.subject_id;
      const previousAdmissions = JSON.stringify(
        state.patient?.admissions || [],
      );
      state.patient = patient;
      state.selection = selection;
      state.snapshot = snapshot || null;
      if (same) {
        if (previousAdmissions !== JSON.stringify(patient.admissions)) {
          $("observation-admission").innerHTML =
            '<option value="">选择住院</option>' +
            patient.admissions
              .map(
                (a) =>
                  `<option value="${esc(a.hadm_id)}">${esc(a.hadm_id)} · ${when(a.admittime).slice(0, 10)}</option>`,
              )
              .join("");
          state.hadm ||= selection.hadm_id || "";
          $("observation-admission").value = state.hadm;
        }
        return;
      }
      state.key = key;
      state.generation++;
      state.token++;
      clearTimeout(state.timer);
      clearTimeout(state.debounce);
      state.manifest =
        snapshot?.manifest ||
        (sameSubject ? state.manifest : patient.extended_tables) ||
        [];
      if (!state.manifest.length) return;
      state.scope = selection.start ? "window" : "patient";
      state.page = 1;
      state.series = "";
      state.query = "";
      state.hadm = selection.hadm_id || "";
      build();
    },
  };
})();
