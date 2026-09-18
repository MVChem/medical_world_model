/* Native HTML/SVG/Canvas: no external scripts, no data uploads. */
const $ = (q) => document.querySelector(q);
const $$ = (q) => [...document.querySelectorAll(q)];
const escapeHTML = (v) =>
  String(v ?? "未记录").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const fmt = (v, digits = 0) =>
  v == null
    ? "—"
    : Number(v).toLocaleString("en-US", { maximumFractionDigits: digits });
const sexLabel = (v) => ({ Female: "女", Male: "男" })[v] || v || "未记录";
const icon = (name) => `<svg><use href="#i-${name}"/></svg>`;
const sequences = {
  t1ce: ["T1 CE", "增强 T1 加权"],
  flair: ["FLAIR", "液体衰减反转恢复"],
  t1: ["T1", "T1 加权"],
  t2: ["T2", "T2 加权"],
};
const planeAxis = { axial: 2, coronal: 1, sagittal: 0 };
const state = {
  dataset: "ucsf",
  view: "explore",
  patient: null,
  sequence: "t1ce",
  plane: "axial",
  index: 75,
  overlay: true,
  opacity: 0.45,
  page: 0,
  search: "",
  generation: 0,
  imageGeneration: 0,
};
let datasets,
  sliceTimer,
  imageAbort,
  imageURLs = [];
const fullOffline = () => window.__ATLAS_SNAPSHOT__?.mode === "volumes";
const sampleOffline = () =>
  Boolean(window.__ATLAS_SNAPSHOT__) && !fullOffline();

async function json(url) {
  if (window.__ATLAS_SNAPSHOT__) {
    const snap = window.__ATLAS_SNAPSHOT__;
    if (url === "/api/catalog") return snap.catalog;
    const [dataset, id] = url.split("?")[0].split("/").slice(-2);
    if (fullOffline())
      return AtlasOffline.patient(
        dataset,
        id,
        new URLSearchParams(url.split("?")[1] || ""),
      );
    if (snap.patients[id]) return snap.patients[id];
    if (dataset === "mu")
      return snap.catalog.mu.patients.find((p) => p.id === id);
    throw new Error(
      "该病例未包含在离线 HTML 中。请启动完整服务查看全部 UCSF MRI。",
    );
  }
  const r = await fetch(url);
  if (!r.ok)
    throw new Error(
      `读取失败 (${r.status})：${(await r.text()).slice(0, 160)}`,
    );
  return r.json();
}

function notify(message) {
  $("#toast").textContent = message;
  $("#toast").hidden = false;
  clearTimeout(notify.timer);
  notify.timer = setTimeout(() => ($("#toast").hidden = true), 4500);
}

function metric(label, value, unit, note, name) {
  return `<article class="metric"><div class="metric-top">${label}<span class="metric-icon">${icon(name)}</span></div><div class="metric-value">${value}<small>${unit}</small></div><div class="metric-note">${note}</div></article>`;
}

function renderMetrics() {
  const d = datasets[state.dataset],
    u = state.dataset === "ucsf";
  $("#metrics").innerHTML =
    metric(
      "患者",
      fmt(d.patient_count),
      "人",
      u ? "独立患者 · 两次纵向检查" : "独立患者 · 临床表有效记录",
      "users",
    ) +
    metric(
      u ? "MRI 检查" : "临床 MRI 时间项",
      fmt(d.visits),
      "次",
      u
        ? "<em>4 序列</em> + 分区标注"
        : `${d.image_visits} 组本地 MRI · ${d.mask_visits} 份标注`,
      "layers",
    ) +
    metric(
      u ? "纵向时间对" : "候选相邻时间对",
      fmt(d.pairs),
      "对",
      u ? "每人一对 · T₁ → T₂" : "按天数排序、去重后计算",
      "grid",
    ) +
    metric(
      "随访间隔中位数",
      fmt(d.median_gap, 1),
      "天",
      u ? "实际扫描间隔" : "候选相邻时间对的间隔",
      "chart",
    );
}

function barChart(target, rows, total, colors) {
  const max = Math.max(1, ...rows.map((x) => x.count));
  $(target).innerHTML = rows
    .map(
      (r, i) =>
        `<div class="bar-row" title="${escapeHTML(r.name)}: ${r.count}${total ? " / " + total : ""}"><span class="bar-label">${escapeHTML(r.name)}</span><div class="bar-track"><div class="bar-fill" style="width:${(r.count / max) * 100}%;background:${colors?.[i % colors.length] || "#91a582"}"></div></div><span class="bar-number">${r.count}</span></div>`,
    )
    .join("");
}

function histogram(target, bins, color = "#91a582") {
  const w = 480,
    h = 154,
    left = 26,
    right = 8,
    top = 12,
    bottom = 30,
    pw = w - left - right,
    ph = h - top - bottom;
  const max = Math.max(1, ...bins.map((b) => b.count)),
    step = pw / bins.length;
  let svg = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${escapeHTML(bins.map((b) => b.label + ": " + b.count).join("；"))}">`;
  [0, 0.5, 1].forEach((f) => {
    const y = top + ph * (1 - f);
    svg += `<path d="M${left} ${y}H${w - right}" stroke="#edf0e6" stroke-width="1" ${f ? 'stroke-dasharray="3 4"' : ""}/><text x="${left - 7}" y="${y + 3}" text-anchor="end" fill="#a7b198" font-size="8">${Math.round(max * f)}</text>`;
  });
  bins.forEach((b, i) => {
    const height = (b.count / max) * ph,
      x = left + step * i + step * 0.2;
    svg += `<rect x="${x}" y="${top + ph - height}" width="${step * 0.6}" height="${Math.max(height, b.count ? 2 : 0)}" rx="3" fill="${color}" opacity="${0.52 + (i / bins.length) * 0.4}"><title>${escapeHTML(b.label)}: ${b.count}</title></rect><text x="${x + step * 0.3}" y="${h - 11}" text-anchor="middle" fill="#9fa98f" font-size="8">${escapeHTML(b.label)}</text>`;
  });
  $(target).innerHTML = svg + "</svg>";
}

function renderCharts() {
  const d = datasets[state.dataset],
    u = state.dataset === "ucsf";
  histogram("#gap-chart", d.gap_histogram);
  $("#gap-subtitle").textContent =
    `${fmt(d.pairs)} ${u ? "个时间对" : "个候选时间对"}`;
  $("#gap-caption").textContent =
    `间隔 / 天 · ${u ? "原始临床表的两次扫描日期差" : "临床时间项；含尚无对应 MRI 的记录"}`;
  const diagnoses =
    d.diagnosis.length > 6
      ? [
          ...d.diagnosis.slice(0, 5),
          {
            name: "其他诊断（合并）",
            count: d.diagnosis.slice(5).reduce((n, r) => n + r.count, 0),
          },
        ]
      : d.diagnosis;
  barChart("#diagnosis-chart", diagnoses, d.patient_count, [
    "#8ca47d",
    "#b4c2a4",
    "#d3ccb2",
    "#a1b7b0",
    "#c8b3b3",
  ]);
  histogram("#age-chart", d.age_histogram, "#b1a386");
  $("#age-caption").textContent = d.age_label + " / 岁";
  const ageRecorded = d.age_histogram.reduce((sum, b) => sum + b.count, 0);
  $("#age-chart").insertAdjacentHTML(
    "beforeend",
    `<p class="chart-caption">有效年龄 ${ageRecorded} / ${d.patient_count} 人 · 缺失值不计入分布</p>`,
  );
  barChart(
    "#timepoint-chart",
    d.timepoint_counts
      .map((r) => ({ ...r, name: r.name + " 个时间点" }))
      .sort((a, b) => parseInt(a.name) - parseInt(b.name)),
    d.patient_count,
  );
  barChart(
    "#sex-chart",
    d.sex.map((r) => ({ ...r, name: sexLabel(r.name) })),
    d.patient_count,
    ["#bba88f", "#96afa4", "#c4cbb9"],
  );
  $("#fourth-chart-title").textContent = u ? "肿瘤分级" : "扫描仪厂商";
  $("#fourth-chart-note").textContent = u
    ? "WHO · 原始记录"
    : d.scanner_rows + " 条扫描仪记录";
  barChart(
    "#grade-chart",
    u
      ? d.grades.map((r) => ({
          ...r,
          name: r.name === "未记录" ? "未记录" : "Grade " + fmt(r.name),
        }))
      : d.scanner_vendors,
    u ? d.patient_count : d.scanner_rows,
  );
  $("#cohort-badge").textContent = d.name + " · " + d.patient_count + " 人";
  $("#mu-volume-card").hidden = u;
  if (!u)
    $("#mu-volumes").innerHTML = d.volume_summary
      .map(
        (v) =>
          `<div class="mu-volume-summary"><small>${v.label} · ${fmt(v.rows)} 条记录</small><b>${fmt(v.median_ml, 2)} <small>mL</small></b><p>中位数 · IQR ${fmt(v.q1_ml, 2)}–${fmt(v.q3_ml, 2)}</p></div>`,
      )
      .join("");
}

function renderTable() {
  const d = datasets[state.dataset],
    u = state.dataset === "ucsf",
    term = state.search.toLowerCase();
  const rows = d.patients.filter((p) =>
    `${p.id} ${p.diagnosis || ""}`.toLowerCase().includes(term),
  );
  const size = 8,
    pages = Math.max(1, Math.ceil(rows.length / size));
  state.page = Math.min(state.page, pages - 1);
  const start = state.page * size;
  $("#patient-count").textContent = rows.length;
  $("#age-column").textContent = u ? "首次扫描年龄" : "诊断年龄";
  $("#gap-column").textContent = u ? "随访间隔" : "相邻间隔中位数";
  $("#patient-rows").innerHTML =
    rows
      .slice(start, start + size)
      .map(
        (p) =>
          `<tr data-id="${escapeHTML(p.id)}" class="${state.patient?.id === p.id ? "selected" : ""}"><td>${escapeHTML(p.id)}</td><td title="${escapeHTML(p.diagnosis)}">${escapeHTML(p.diagnosis)}</td><td>${fmt(p.age)} 岁</td><td>${escapeHTML(sexLabel(p.sex))}</td><td>${p.timepoints}</td><td>${fmt(p.gap_days, 1)} ${p.gap_days == null ? "" : "天"}</td><td><span class="badge ${p.image_available ? "green-badge" : "amber-badge"}">${p.image_available ? "MRI 已匹配" : "仅表格"}</span></td><td><button class="row-open" aria-label="查看患者 ${escapeHTML(p.id)}">查看 ↗</button></td></tr>`,
      )
      .join("") ||
    '<tr><td colspan="8" style="text-align:center;padding:30px">没有匹配的患者</td></tr>';
  $("#page-info").textContent = rows.length
    ? `显示 ${start + 1}–${Math.min(start + size, rows.length)} / ${rows.length} 位患者`
    : "0 位患者";
  $("#previous-page").disabled = state.page === 0;
  $("#next-page").disabled = state.page >= pages - 1;
  $$("#patient-rows tr[data-id]").forEach((row) =>
    row.addEventListener("click", () => {
      setView("explore");
      setPatient(row.dataset.id);
      $(".viewer-card").scrollIntoView({ block: "start", behavior: "smooth" });
    }),
  );
}

function setView(view) {
  state.view = view;
  $("#explore-view").hidden = view !== "explore";
  $("#cohort-view").hidden = view !== "cohort";
  $("#sources-view").hidden = view !== "sources";
  $("#tables-view").hidden = view !== "tables";
  $("#patient-records").hidden = view !== "explore";
  $("#patients-section").hidden = ["sources", "tables"].includes(view);
  $("#metrics").hidden = ["sources", "tables"].includes(view);
  $$(".nav-item").forEach((b) =>
    b.classList.toggle("active", b.dataset.view === view),
  );
  $("#breadcrumb-current").textContent = {
    explore: "数据探索",
    cohort: "队列概览",
    sources: "来源与说明",
    tables: "原始表格",
  }[view];
  if (view === "tables") loadWorksheets();
}

function setDataset(dataset) {
  state.dataset = dataset;
  state.page = 0;
  state.search = "";
  $("#patient-search").value = "";
  const d = datasets[dataset],
    u = dataset === "ucsf";
  $$(".dataset-button").forEach((b) =>
    b.classList.toggle("active", b.dataset.dataset === dataset),
  );
  $("#export-link").href = `/api/export/${dataset}.csv`;
  $("#viewer-status").textContent = u
    ? "UCSF · MRI 已就绪"
    : `MU · ${d.image_visits} 组 MRI`;
  $("#viewer-status").className =
    "badge " + (d.image_available ? "green-badge" : "amber-badge");
  $("#ucsf-viewer").hidden = !d.image_available;
  $("#mu-viewer").hidden = u;
  $("#visit-controls").hidden = u;
  $("#sequence-controls").hidden = !d.image_available;
  $("#sequence-note").hidden = !d.image_available;
  $("#viewer-method").textContent =
    "RAS+ 神经学显示 · 同步切片 · 各扫描独立显示窗";
  if (sampleOffline())
    $("#viewer-method").textContent =
      "离线示例 · 轴位采样切片 · 固定叠加 45% · 完整服务可浏览所有病例";
  if (fullOffline())
    $("#viewer-method").textContent =
      "完整三维影像 · 全切片 · 首个本地检查网格 · 按需加载";
  $("#patient-select").innerHTML = d.patients
    .map(
      (p) =>
        `<option value="${escapeHTML(p.id)}">${escapeHTML(p.id)}${sampleOffline() && u && !window.__ATLAS_SNAPSHOT__.patients[p.id] ? " · 完整服务可用" : ""}</option>`,
    )
    .join("");
  renderMetrics();
  renderCharts();
  const initial = u
    ? window.__ATLAS_SNAPSHOT__?.default_patient || "100004"
    : d.patients[0].id;
  setPatient(initial);
  if (state.view === "tables") loadWorksheets();
}

function renderClinical(p) {
  const fields = Object.entries(p.clinical || {});
  $("#clinical-count").textContent = `${p.id} · ${fields.length} 个字段`;
  $("#clinical-fields").innerHTML = fields
    .map(
      ([key, value]) =>
        `<tr><th>${escapeHTML(key)}</th><td>${escapeHTML(value)}</td></tr>`,
    )
    .join("");
}

let filePatient,
  fileGeneration = 0,
  fileTimer;
async function setupFileViewer(p) {
  const generation = ++fileGeneration;
  filePatient = null;
  $("#file-record").hidden = true;
  $("#file-image").removeAttribute("src");
  if (sampleOffline()) return;
  try {
    const record = p.files ? p : await AtlasOffline.record(state.dataset, p.id);
    if (generation !== fileGeneration) return;
    filePatient = record;
    $("#file-record").hidden = false;
    $("#file-count").textContent = `${p.id} · ${record.files.length} 个文件`;
    $("#file-select").innerHTML = record.files
      .map((f, i) => `<option value="${i}">${escapeHTML(f.name)}</option>`)
      .join("");
    $("#file-description").textContent = "展开后可查看该患者全部原始影像文件。";
    if ($("#file-record").open) updateFileViewer(true);
  } catch (e) {
    if (generation === fileGeneration) console.warn(e.message);
  }
}

async function updateFileViewer(reset = false) {
  if (!filePatient) return;
  const generation = ++fileGeneration,
    f = filePatient.files[Number($("#file-select").value)];
  if (!f) return;
  const plane = $("#file-plane").value,
    maximum = f.shape[planeAxis[plane]] - 1;
  $("#file-slice").max = maximum;
  if (reset) $("#file-slice").value = Math.floor(maximum / 2);
  const index = Math.min(maximum, Number($("#file-slice").value));
  $("#file-slice-value").textContent = `${index + 1} / ${maximum + 1}`;
  const display =
    f.kind === "label"
      ? "分割标签：" + f.labels.join(" / ")
      : f.kind === "signed"
        ? "有符号差分 · 对称显示窗"
        : "MRI · 各体积独立显示窗";
  const directions = {
    axial: "上 A / 下 P · 左 L / 右 R",
    coronal: "上 S / 下 I · 左 L / 右 R",
    sagittal: "上 S / 下 I · 左 P / 右 A",
  };
  $("#file-description").textContent =
    `${f.shape.join(" × ")} · ${f.spacing.join(" × ")} mm · ${display} · ${directions[plane]}${f.source_spatial_unit === "unknown" ? " · 原头未填写单位；已核对同病例毫米网格" : ""}`;
  $("#file-loading").hidden = false;
  $("#file-loading").textContent = "正在载入三维影像…";
  try {
    const url = await AtlasOffline.renderFile(f, plane, index);
    if (generation !== fileGeneration) return;
    $("#file-image").src = url;
    $("#file-loading").hidden = true;
  } catch (e) {
    if (generation === fileGeneration)
      $("#file-loading").textContent = e.message;
  }
}
$("#file-record").addEventListener("toggle", () => {
  if ($("#file-record").open) updateFileViewer(true);
});
$("#file-select").addEventListener("change", () => updateFileViewer(true));
$("#file-plane").addEventListener("change", () => updateFileViewer(true));
$("#file-slice").addEventListener("input", () => {
  clearTimeout(fileTimer);
  fileTimer = setTimeout(() => updateFileViewer(), 80);
});

let worksheets = [],
  worksheetPage = 0,
  tableGeneration = 0;
async function loadWorksheets() {
  const generation = ++tableGeneration;
  $("#worksheet-description").textContent = "正在读取原始工作表…";
  try {
    const result = await AtlasOffline.rawTables(state.dataset);
    if (generation !== tableGeneration) return;
    worksheets = result;
    worksheetPage = 0;
    $("#table-search").value = "";
    $("#worksheet-select").innerHTML = result
      .map(
        (t, i) =>
          `<option value="${i}">${escapeHTML(t.workbook)} / ${escapeHTML(t.sheet)}</option>`,
      )
      .join("");
    renderWorksheet();
  } catch (e) {
    $("#worksheet-description").textContent = e.message;
  }
}
function columnName(i) {
  let s = "";
  for (i++; i > 0; i = Math.floor((i - 1) / 26))
    s = String.fromCharCode(65 + ((i - 1) % 26)) + s;
  return s;
}
function renderWorksheet() {
  const table = worksheets[Number($("#worksheet-select").value)];
  if (!table) return;
  const term = $("#table-search").value.toLowerCase(),
    rows = table.rows
      .map((r, i) => ({ r, number: i + 1 }))
      .filter((x) =>
        x.r.some((v) =>
          String(v ?? "")
            .toLowerCase()
            .includes(term),
        ),
      );
  const pages = Math.max(1, Math.ceil(rows.length / 30));
  worksheetPage = Math.max(0, Math.min(pages - 1, worksheetPage));
  const columns = Math.max(0, ...table.rows.map((r) => r.length));
  $("#worksheet-description").textContent =
    `${table.workbook} / ${table.sheet} · ${table.rows.length} 行 × ${columns} 列（含原始标题行）`;
  $("#worksheet-head").innerHTML =
    "<tr><th>原始行</th>" +
    Array.from({ length: columns }, (_, i) => `<th>${columnName(i)}</th>`).join(
      "",
    ) +
    "</tr>";
  $("#worksheet-body").innerHTML = rows
    .slice(worksheetPage * 30, (worksheetPage + 1) * 30)
    .map(
      (x) =>
        `<tr><th>${x.number}</th>${x.r.map((v) => `<td>${escapeHTML(v ?? "")}</td>`).join("")}</tr>`,
    )
    .join("");
  $("#worksheet-page").textContent =
    `${rows.length} 行 · 第 ${worksheetPage + 1} / ${pages} 页`;
  $("#table-previous").disabled = worksheetPage === 0;
  $("#table-next").disabled = worksheetPage === pages - 1;
}
$("#worksheet-select").addEventListener("change", () => {
  worksheetPage = 0;
  renderWorksheet();
});
$("#table-search").addEventListener("input", () => {
  worksheetPage = 0;
  renderWorksheet();
});
$("#table-previous").addEventListener("click", () => {
  worksheetPage--;
  renderWorksheet();
});
$("#table-next").addEventListener("click", () => {
  worksheetPage++;
  renderWorksheet();
});
$("#table-export").addEventListener("click", () => {
  const table = worksheets[Number($("#worksheet-select").value)];
  if (!table) return;
  const csv = table.rows
    .map((row) =>
      row
        .map((v) => {
          let text = String(v ?? "");
          if (typeof v === "string" && /^[=+@-]/.test(text)) text = "'" + text;
          return '"' + text.replaceAll('"', '""') + '"';
        })
        .join(","),
    )
    .join("\n");
  const a = document.createElement("a"),
    url = URL.createObjectURL(
      new Blob(["\ufeff" + csv], { type: "text/csv;charset=utf-8" }),
    );
  a.href = url;
  a.download = table.sheet + ".csv";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

async function setPatient(pid, first = null, second = null) {
  const generation = ++state.generation;
  state.imageGeneration++;
  imageAbort?.abort();
  state.patient = null;
  $("#patient-select").value = pid;
  const u = state.dataset === "ucsf";
  $("#image-loading").textContent = "正在读取 MRI…";
  $("#image-loading").hidden = false;
  {
    $("#scan-1").removeAttribute("src");
    $("#scan-2").removeAttribute("src");
    $("#detail-id").textContent = pid;
    $("#detail-diagnosis").textContent = "正在读取病例…";
    $("#detail-facts").innerHTML = "";
    $("#volume-chart").innerHTML = "";
  }
  $("#slice-slider").disabled = true;
  try {
    const query = new URLSearchParams();
    if (first !== null) query.set("first", first);
    if (second !== null && second !== first) query.set("second", second);
    const p = await json(
      `/api/patient/${state.dataset}/${pid}${query.size ? "?" + query : ""}`,
    );
    if (generation !== state.generation) return;
    state.patient = p;
    renderClinical(p);
    setupFileViewer(p);
    renderTable();
    $("#ucsf-viewer").hidden = !p.image;
    $("#sequence-controls").hidden = !p.image;
    $("#sequence-note").hidden = !p.image;
    $("#visit-controls").hidden = u || !p.image;
    $(".mu-placeholder").hidden = Boolean(p.image);
    $(".mu-content").style.gridTemplateColumns = p.image ? "1fr" : "";
    if (!u) renderMUPatient(p);
    if (p.image) {
      $("#detail-id").textContent = p.id;
      $("#detail-diagnosis").textContent = p.diagnosis || "未记录";
      const facts = [
        ["年龄 / 性别", `${fmt(p.age)} 岁 / ${sexLabel(p.sex)}`],
        ["WHO 分级", p.grade == null ? "未记录" : "Grade " + p.grade],
        [
          u ? "随访间隔" : "本地 MRI",
          u ? fmt(p.gap_days, 1) + " 天" : p.image_visits.length + " 个时间点",
        ],
        [
          u ? "IDH 状态" : "分割标注",
          u
            ? p.idh || "未记录"
            : p.image.mask_available.filter(Boolean).length +
              " / " +
              p.image.timepoints.length,
        ],
      ];
      $("#detail-facts").innerHTML = facts
        .map(
          ([k, v]) =>
            `<div class="fact"><small>${k}</small><b>${escapeHTML(v)}</b></div>`,
        )
        .join("");
      const ts = p.image.timepoints;
      const visitDay = (t) =>
        u
          ? t === 1
            ? 0
            : p.gap_days
          : p.image_visits.find((v) => v.timepoint === t)?.day;
      $("#scan-time-1").textContent = "T" + ts[0];
      $("#scan-time-2").textContent = ts.length > 1 ? "T" + ts[1] : "";
      $("#reference-day").textContent = "DAY " + fmt(visitDay(ts[0]));
      $("#followup-day").textContent =
        ts.length > 1 ? "DAY " + fmt(visitDay(ts[1])) : "";
      $("#followup-panel").hidden = ts.length < 2;
      $("#scan-pair").style.gridTemplateColumns = ts.length < 2 ? "1fr" : "";
      $("#volume-unit").textContent =
        "mL · " + ts.map((t) => "T" + t).join(" → ");
      $("#mask-note").textContent = p.image.mask_available.every(Boolean)
        ? "由原始分割掩膜计算。体积变化不直接等同于临床进展。"
        : "部分检查缺少分割标注，显示为 —；不能视为零体积。";
      if (!u) {
        const options = p.image_visits
          .map(
            (v) =>
              `<option value="${v.timepoint}">T${v.timepoint} · ${v.day == null ? "日期缺失" : "诊断后 " + fmt(v.day) + " 天"}${v.mask_available ? "" : " · 无标注"}</option>`,
          )
          .join("");
        for (const [id, selected] of [
          ["#visit-first", ts[0]],
          ["#visit-second", ts[1] ?? ts[0]],
        ]) {
          $(id).innerHTML = options;
          $(id).value = selected;
          $(id).disabled = sampleOffline() || p.image_visits.length < 2;
        }
        $("#visit-note").textContent =
          ts.length < 2 ? "单次检查" : "保留原始时间点编号";
      }
      const max = Math.max(
        1,
        ...p.image.volumes_ml.flatMap((v) => (v ? Object.values(v) : [])),
      );
      $("#volume-chart").innerHTML = p.image.labels
        .map((l) => {
          const a = p.image.volumes_ml[0]?.[l.id],
            b = p.image.volumes_ml[1]?.[l.id];
          return `<div class="volume-item"><div class="volume-label"><i style="background:${l.color}"></i>${l.name}<span>${fmt(a, 2)}${ts.length > 1 ? " → " + fmt(b, 2) : ""}</span></div><div class="volume-bars"><div style="width:${((a || 0) / max) * 100}%;background:${l.color}"></div>${ts.length > 1 ? `<div style="width:${((b || 0) / max) * 100}%;background:${l.color}"></div>` : ""}</div></div>`;
        })
        .join("");
      $("#legend").innerHTML = p.image.labels
        .map(
          (l) =>
            `<span title="${l.id} · ${l.description}"><i style="background:${l.color}"></i>${l.name} ${l.description}</span>`,
        )
        .join("");
      state.index = p.image.default_slices[state.plane];
      if (sampleOffline()) {
        state.plane = "axial";
        state.overlay = true;
        state.opacity = 0.45;
        $("#overlay-toggle").checked = true;
        $("#opacity").value = 45;
        state.index =
          window.__ATLAS_SNAPSHOT__.examples?.[pid]?.default_index ??
          window.__ATLAS_SNAPSHOT__.default_index;
      }
      $("#slice-slider").disabled = false;
      updateSliceControls();
      await requestImages();
    }
  } catch (e) {
    if (generation !== state.generation) return;
    $("#image-loading").textContent = e.message;
    $("#image-loading").hidden = false;
    notify(e.message);
    renderTable();
  }
}

function renderMUPatient(p) {
  $("#mu-id").textContent = p.id;
  $("#mu-profile").textContent =
    `${p.diagnosis || "未记录"} · ${fmt(p.age)} 岁 · ${sexLabel(p.sex)}`;
  const ts = p.timeline;
  if (!ts.length) {
    $("#mu-timeline").innerHTML =
      '<p class="notice">没有数值型 MRI 时间记录</p>';
    return;
  }
  const lo = Math.min(0, ...ts.map((t) => t.day)),
    hi = Math.max(lo + 1, ...ts.map((t) => t.day)),
    w = 440;
  let svg = `<div class="timeline"><svg viewBox="0 0 ${w} 116" role="img" aria-label="患者随访时间线"><path d="M20 48H420" stroke="#dfe7d3" stroke-width="2"/><text x="20" y="100" font-size="9" fill="#a7b694">诊断后天数</text>`;
  ts.forEach((t, i) => {
    const x = 20 + ((t.day - lo) / (hi - lo)) * 400;
    svg += `<circle cx="${x}" cy="48" r="${i ? 5 : 7}" fill="${i ? "#a3b38b" : "#68845c"}" stroke="#fff" stroke-width="2"><title>Timepoint_${t.timepoint}: ${t.day} 天</title></circle><text x="${x}" y="${i % 2 ? 79 : 23}" text-anchor="middle" font-size="9" fill="#91a279">T${t.timepoint}</text>`;
  });
  $("#mu-timeline").innerHTML =
    svg +
    '</svg></div><div class="timeline-entries">' +
    ts
      .map(
        (t) =>
          `<span class="timeline-entry">T${t.timepoint} · ${fmt(t.day)} 天 · ${t.image_available ? "MRI ✓" : "本地无 MRI"}</span>`,
      )
      .join("") +
    "</div>";
}

function updateSliceControls() {
  if (!state.patient?.image) return;
  const max = state.patient.image.shape[planeAxis[state.plane]] - 1;
  const snap = sampleOffline()
    ? (window.__ATLAS_SNAPSHOT__.examples?.[state.patient.id] ??
      window.__ATLAS_SNAPSHOT__)
    : null;
  if (snap) {
    $("#slice-slider").min = 0;
    $("#slice-slider").max = snap.indices.length - 1;
    $("#slice-slider").value = Math.max(0, snap.indices.indexOf(state.index));
  } else {
    state.index = Math.max(0, Math.min(max, state.index));
    $("#slice-slider").min = 0;
    $("#slice-slider").max = max;
    $("#slice-slider").value = state.index;
  }
  $("#slice-value").textContent = `${state.index + 1} / ${max + 1}`;
  $$(".slice-caption").forEach(
    (n) => (n.textContent = `SLICE ${state.index + 1}`),
  );
  $$(".plane-select button").forEach((b) =>
    b.classList.toggle("active", b.dataset.plane === state.plane),
  );
  const captions = `${sequences[state.sequence][0]} · ${state.plane.toUpperCase()}`;
  $("#caption-1").textContent = captions;
  $("#caption-2").textContent = captions;
  const [north, south, west, east] =
    state.plane === "axial"
      ? ["A", "P", "L", "R"]
      : state.plane === "coronal"
        ? ["S", "I", "L", "R"]
        : ["S", "I", "P", "A"];
  for (const [cls, text] of Object.entries({ north, south, west, east }))
    $$(".orientation." + cls).forEach((n) => (n.textContent = text));
}

async function requestImages() {
  if (!state.patient?.image) return;
  const generation = ++state.imageGeneration;
  imageAbort?.abort();
  imageAbort = new AbortController();
  const signal = imageAbort.signal;
  $("#scan-pair").classList.add("loading");
  $("#image-loading").textContent = "正在读取切片…";
  $("#image-loading").hidden = false;
  const p = state.patient,
    snap = window.__ATLAS_SNAPSHOT__;
  let urls = [];
  const allocated = [];
  try {
    urls = await Promise.all(
      p.image.timepoints.map(async (t) => {
        if (fullOffline())
          return AtlasOffline.render(
            p,
            state.sequence,
            state.plane,
            state.index,
            t,
            state.overlay,
            state.opacity,
          );
        if (snap) {
          const key = `${p.id}:${state.sequence}:${state.plane}:${state.index}:${t}`;
          if (!snap.slices[key]) throw new Error("此切片未包含在离线 HTML 中");
          return snap.slices[key];
        }
        const params = new URLSearchParams({
          sequence: state.sequence,
          plane: state.plane,
          index: state.index,
          timepoint: t,
          overlay: state.overlay,
          opacity: state.opacity,
          first: p.image.timepoints[0],
        });
        if (p.image.timepoints.length > 1)
          params.set("second", p.image.timepoints[1]);
        const r = await fetch(`/api/slice/${p.id}.png?${params}`, { signal });
        if (!r.ok) throw new Error("MRI 切片读取失败：" + r.status);
        const url = URL.createObjectURL(await r.blob());
        allocated.push(url);
        return url;
      }),
    );
    await Promise.all(
      urls.map(
        (url) =>
          new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = resolve;
            img.onerror = () => reject(new Error("切片图像解码失败"));
            img.src = url;
          }),
      ),
    );
    if (generation !== state.imageGeneration) {
      urls.filter((u) => u.startsWith("blob:")).forEach(URL.revokeObjectURL);
      return;
    }
    $("#scan-1").src = urls[0];
    if (urls[1]) $("#scan-2").src = urls[1];
    else $("#scan-2").removeAttribute("src");
    imageURLs.filter((u) => u.startsWith("blob:")).forEach(URL.revokeObjectURL);
    imageURLs = urls;
    $("#image-loading").hidden = true;
    $("#scan-pair").classList.remove("loading");
  } catch (e) {
    allocated.forEach(URL.revokeObjectURL);
    if (e.name === "AbortError" || generation !== state.imageGeneration) return;
    $("#image-loading").textContent = e.message;
    $("#scan-pair").classList.remove("loading");
  }
}

function setSlice(value) {
  if (!state.patient?.image) return;
  const snap = sampleOffline()
    ? (window.__ATLAS_SNAPSHOT__.examples?.[state.patient.id] ??
      window.__ATLAS_SNAPSHOT__)
    : null;
  state.index = snap
    ? snap.indices[Math.max(0, Math.min(snap.indices.length - 1, value))]
    : value;
  updateSliceControls();
  clearTimeout(sliceTimer);
  sliceTimer = setTimeout(requestImages, 110);
}

$$(".nav-item").forEach((b) =>
  b.addEventListener("click", () => setView(b.dataset.view)),
);
$$("[data-go]").forEach((b) =>
  b.addEventListener("click", () => setView(b.dataset.go)),
);
$$(".dataset-button").forEach((b) =>
  b.addEventListener("click", () => setDataset(b.dataset.dataset)),
);
$("#patient-select").addEventListener("change", (e) =>
  setPatient(e.target.value),
);
["#visit-first", "#visit-second"].forEach((id) =>
  $(id).addEventListener("change", () =>
    setPatient(
      $("#patient-select").value,
      Number($("#visit-first").value),
      Number($("#visit-second").value),
    ),
  ),
);
$$("[data-sequence]").forEach((b) =>
  b.addEventListener("click", () => {
    state.sequence = b.dataset.sequence;
    $$("[data-sequence]").forEach((x) => x.classList.toggle("active", x === b));
    $("#sequence-note").textContent = sequences[state.sequence][1];
    updateSliceControls();
    requestImages();
  }),
);
$$("[data-plane]").forEach((b) =>
  b.addEventListener("click", () => {
    if (!state.patient?.image) return;
    state.plane = b.dataset.plane;
    state.index = state.patient.image.default_slices[state.plane];
    updateSliceControls();
    requestImages();
  }),
);
$("#slice-slider").addEventListener("input", (e) =>
  setSlice(Number(e.target.value)),
);
$("#slice-prev").addEventListener("click", () =>
  setSlice(Number($("#slice-slider").value) - 1),
);
$("#slice-next").addEventListener("click", () =>
  setSlice(Number($("#slice-slider").value) + 1),
);
$("#scan-pair").addEventListener(
  "wheel",
  (e) => {
    if (!state.patient?.image) return;
    e.preventDefault();
    setSlice(Number($("#slice-slider").value) + (e.deltaY > 0 ? 1 : -1));
  },
  { passive: false },
);
$("#overlay-toggle").addEventListener("change", (e) => {
  state.overlay = e.target.checked;
  requestImages();
});
$("#opacity").addEventListener("input", (e) => {
  state.opacity = Number(e.target.value) / 100;
  clearTimeout(sliceTimer);
  sliceTimer = setTimeout(requestImages, 150);
});
$("#patient-search").addEventListener("input", (e) => {
  state.search = e.target.value;
  state.page = 0;
  renderTable();
});
$("#previous-page").addEventListener("click", () => {
  state.page--;
  renderTable();
});
$("#next-page").addEventListener("click", () => {
  state.page++;
  renderTable();
});
$("#export-link").addEventListener("click", (e) => {
  if (!window.__ATLAS_SNAPSHOT__) return;
  e.preventDefault();
  const fields = [
    "id",
    "age",
    "sex",
    "diagnosis",
    "grade",
    "timepoints",
    "gap_days",
    "image_available",
  ];
  const csv =
    fields.join(",") +
    "\n" +
    datasets[state.dataset].patients
      .map((p) =>
        fields
          .map((k) => '"' + String(p[k] ?? "").replaceAll('"', '""') + '"')
          .join(","),
      )
      .join("\n");
  const url = URL.createObjectURL(
    new Blob(["\ufeff" + csv], { type: "text/csv;charset=utf-8" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = state.dataset + "_patients.csv";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

(async () => {
  try {
    datasets = await json("/api/catalog");
    $("#mu-dataset-status").innerHTML =
      `<i class="dot ${datasets.mu.image_available ? "green" : "amber"}"></i>${datasets.mu.image_visits} 组 MRI · 203 人`;
    setDataset("ucsf");
    if (sampleOffline()) {
      $$(".top-note").forEach((n) => (n.textContent = "离线 HTML · 内置示例"));
      $$("[data-plane]").forEach(
        (b) => (b.disabled = b.dataset.plane !== "axial"),
      );
      $("#overlay-toggle").disabled = true;
      $("#opacity").disabled = true;
      $("#viewer-method").textContent =
        "离线示例 · 轴位采样切片 · 固定叠加 45% · 完整服务可浏览所有病例";
    }
    if (fullOffline())
      $$(".top-note").forEach((n) => (n.textContent = "完整数据 · 501 位患者"));
  } catch (e) {
    $("#global-error").hidden = false;
    $("#global-error").textContent = e.message + "。请按 README 启动本地服务。";
  }
})();
