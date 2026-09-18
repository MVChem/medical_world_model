import { esc, fmt, when, time, time as epoch, iso } from "./format";
export function offlineQuery(state) {
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

export function chartHTML(data, state, bounds) {
  const chart = data.chart,
    series = data.series.find((s) => s.id === data.selected_series),
    points = chart.points;
  if (!points.length) {
    return '<div class="empty-state">当前记录没有可绘图的数值与时间。文本、范围值、日期级及无时间记录保留在明细中。</div>';
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
  return `<div class="observation-chart-wrap">${svg}</svg></div><div class="chart-caption"><span>${fmt(chart.total)} 个原始数值点${chart.sampled ? ` · 绘图保留 ${points.length} 个分段极值与端点` : ""} · 悬停查看原始值和录入时间</span><span>仅显示观测点，不插值、不换算单位</span></div>`;
}

export function timelineHTML(patient, selectedPair, studies) {
  const pair = selectedPair?.mimic_cxr_transition;
  const t0 = epoch(
    pair?.current_state.image.acquisition_timestamp ||
      studies[0]?.timestamp ||
      null,
  );
  const t1 = pair ? epoch(pair.future_state.image.acquisition_timestamp) : t0;
  const pad = Math.max((t1 - t0) * 0.12, 4 * 3600000);
  const start = t0 - pad,
    end = t1 + pad;
  const x = (time) => 106 + ((time - start) / (end - start)) * 750;
  const clamp = (time) => Math.max(start, Math.min(end, time));
  const lanes = [
    ["住院", patient.admissions, "admittime", "dischtime", "#d1e8e7", 70],
    ["ICU", patient.stays, "intime", "outtime", "#a6cbd2", 107],
    ["病区转移", patient.transfers, "intime", "outtime", "#dce3ed", 144],
  ];
  let svg = `<svg viewBox="0 0 885 212" class="timeline-chart" role="img" aria-label="所选影像前后的住院、ICU 与病区时间线"><text x="18" y="38" fill="#80949e" font-size="10">胸片</text>`;
  lanes.forEach(([name, rows, a, b, color, y]) => {
    svg += `<text x="18" y="${y + 15}" fill="#80949e" font-size="10">${name}</text><line x1="106" x2="856" y1="${y + 11}" y2="${y + 11}" stroke="#e7edef"/>`;
    rows.forEach((r) => {
      const s = epoch(r[a]),
        e = epoch(r[b]);
      if (
        !Number.isFinite(s) ||
        !Number.isFinite(e) ||
        e < start ||
        s > end ||
        e < s
      )
        return;
      const title = `${r.careunit || r.first_careunit || r.admission_type || name} · ${r.stay_id || r.hadm_id || ""}\n${when(r[a])} → ${when(r[b])}`;
      svg += `<rect x="${x(clamp(s)).toFixed(2)}" y="${y}" width="${Math.max(2, x(clamp(e)) - x(clamp(s))).toFixed(2)}" height="22" rx="3" fill="${color}"><title>${esc(title)}</title></rect>`;
    });
  });
  [t0, ...(pair ? [t1] : [])].forEach((t, i) => {
    const color = i ? "#b18b50" : "#208b83";
    svg += `<line x1="${x(t)}" x2="${x(t)}" y1="40" y2="171" stroke="${color}" stroke-dasharray="3 4" opacity=".65"/><circle cx="${x(t)}" cy="34" r="5" fill="${color}"/><text x="${x(t)}" y="20" text-anchor="middle" fill="${color}" font-size="9">${i ? "随访 T₁" : "当前 T₀"}</text>`;
  });
  for (let i = 0; i < 5; i++) {
    const t = start + ((end - start) * i) / 4;
    const d = new Date(t).toISOString();
    svg += `<text x="${x(t)}" y="191" text-anchor="middle" fill="#91a1aa" font-size="9">${d.slice(5, 10)}</text><text x="${x(t)}" y="204" text-anchor="middle" fill="#a3afb6" font-size="8">${d.slice(11, 16)}</text>`;
  }
  return svg + "</svg>";
}
