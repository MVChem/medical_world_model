/* Paginated metadata browser. Never requests patient details or image pixels. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const esc = (value) =>
    String(value ?? "").replace(
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
  const fmt = (n) => (n == null ? "—" : Number(n).toLocaleString("en-US"));
  const s = {
    mode: "patients",
    page: 1,
    token: 0,
    catalog: null,
    index: null,
    timer: null,
  };
  const controls = [
    "patient-search",
    "coverage-filter",
    "split-filter",
    "paired-filter",
    "linkage-filter",
    "sort-filter",
    "longitudinal-filter",
    "page-size",
  ];
  async function get(url) {
    const r = await fetch(url);
    if (!r.ok) throw Error(`读取失败 (${r.status})`);
    return r.json();
  }
  function params() {
    const out = new URLSearchParams({
      q: $("patient-search").value.trim(),
      split: $("split-filter").value,
      page: s.page,
      limit: $("page-size").value,
    });
    if (s.mode === "pairs") {
      if (out.get("split") === "none") out.set("split", "all");
      out.set("linkage", $("linkage-filter").value);
    } else {
      out.set("coverage", $("coverage-filter").value);
      out.set("paired", $("paired-filter").value);
      out.set("sort", $("sort-filter").value);
      out.set("longitudinal", $("longitudinal-filter").checked);
      out.set("featured", s.mode === "featured");
    }
    return out;
  }
  function metrics() {
    const index = s.index || {},
      c = index.counts || {},
      initial = s.catalog.counts;
    $("metric-patients").textContent = fmt(c.patients ?? initial.patients);
    $("metric-patients-note").textContent =
      index.state === "ready"
        ? "CXR ∪ IV · 包含仅 IV 患者"
        : "CXR 已就绪 · IV 汇总中";
    $("metric-matched").textContent = fmt(c.matched_patients);
    $("metric-studies").textContent = fmt(initial.studies);
    $("metric-images").textContent = `${fmt(initial.images)} 张胸片`;
    $("metric-pairs").textContent = fmt(c.linked_pairs);
    $("metric-pairs-note").textContent = `${fmt(c.pairs)} 个可用影像配对中`;
    $("nav-patient-count").textContent = fmt(c.patients ?? initial.patients);
    $("nav-pair-count").textContent = fmt(c.pairs);
    $("cohort-progress").hidden =
      index.state === "ready" && !index.warnings?.length;
    $("cohort-progress").textContent =
      index.state === "ready"
        ? (index.warnings || []).join("；")
        : index.state === "error"
          ? `全库汇总失败：${index.error}`
          : `${index.stage || "正在汇总 IV"}${index.total ? ` · ${fmt(index.processed)} / ${fmt(index.total)}` : ""}。已有 CXR 可先浏览。`;
    $("coverage-bars").innerHTML = [
      ["matched", "CXR × IV", c.matched_patients],
      ["cxr_only", "仅 CXR", c.cxr_only],
      ["iv_only", "仅 IV", c.iv_only],
    ]
      .map(
        ([key, name, n]) =>
          `<button class="coverage-row" data-coverage="${key}"><span>${name}</span><span class="coverage-track"><i style="width:${c.patients ? Math.max(n ? 1 : 0, (n / c.patients) * 100) : 0}%"></i></span><b>${fmt(n)}</b><span class="muted">↗</span></button>`,
      )
      .join("");
    $("split-summary").innerHTML =
      `<table class="split-mini"><thead><tr><th>Split</th><th>患者</th><th>影像配对</th><th>同住院配对</th></tr></thead><tbody>${["train", "validate", "test"].map((k) => `<tr><td><button data-split="${k}">${k}</button></td><td>${fmt(index.splits?.[k]?.patients)}</td><td>${fmt(index.splits?.[k]?.pairs ?? (index.state === "ready" ? 0 : null))}</td><td>${fmt(index.splits?.[k]?.linked_pairs ?? (index.state === "ready" ? 0 : null))}</td></tr>`).join("")}</tbody></table>`;
    if (index.state === "ready") {
      const distribution = (title, data) =>
        `<div><h4>${title}</h4>${Object.entries(data || {})
          .map(
            ([k, v]) =>
              `<div class="distribution-row"><span>${esc(k)}</span><b>${fmt(v)}</b></div>`,
          )
          .join("")}</div>`;
      $("dataset-audit-body").innerHTML =
        `<div class="audit-rules">${Object.values(index.rules || {})
          .map((x) => `<p>${esc(x)}</p>`)
          .join(
            "",
          )}</div><div class="audit-distributions">${distribution("时间间隔", index.horizons)}${distribution("投照体位 · 影像数", index.views)}${distribution("阳性标签 · 检查数", index.positive_labels)}${distribution("配对审计", index.audit)}${distribution("检查匹配", index.study_linkage)}${distribution("二元标签变化", index.label_changes)}</div>`;
    }
  }
  const badge = (text, cls = "") =>
    `<span class="data-badge ${cls}">${esc(text)}</span>`;
  const coverage = (r) =>
    r.has_cxr && r.has_iv
      ? badge("CXR × IV", "matched")
      : r.has_iv
        ? badge("IV")
        : badge("CXR");
  const status = (k) =>
    badge(
      { unique: "唯一共同住院", unmatched: "未匹配", ambiguous: "有歧义" }[k] ||
        k,
      k === "unique" ? "matched" : "",
    );
  async function list() {
    const token = ++s.token,
      pairs = s.mode === "pairs";
    $("list-title").textContent = pairs
      ? "影像配对"
      : s.mode === "featured"
        ? "精选示例"
        : "全量患者";
    ["coverage-filter", "paired-filter", "sort-filter"].forEach(
      (id) => ($(id).hidden = pairs),
    );
    $("longitudinal-filter").parentElement.hidden = pairs;
    $("linkage-filter").hidden = !pairs;
    $("cohort-list-note").textContent = pairs
      ? "全库可用候选 · 点击配对检查输入、随访与关联住院"
      : "当前页仅加载摘要 · 点击患者查看影像与临床记录";
    document
      .querySelectorAll(".workspace-nav button")
      .forEach((b) => b.classList.toggle("active", b.id === `nav-${s.mode}`));
    const query = params();
    $("cohort-export").href =
      `/api/cohort/${pairs ? "pairs" : "patients"}.csv?${query}`;
    $("cohort-export").setAttribute(
      "aria-disabled",
      s.index?.state !== "ready",
    );
    try {
      const data = await get(
        `${pairs ? "/api/cohort/pairs" : "/api/patients"}?${query}`,
      );
      if (token !== s.token) return;
      const limit = Number($("page-size").value),
        pages = Math.max(1, Math.ceil(data.total / limit));
      if (s.page > pages) {
        s.page = pages;
        return list();
      }
      $("list-count").textContent = fmt(data.total);
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
      const rows = data.rows
        .map((r) =>
          pairs
            ? `<tr><td class="mono">${esc(r.subject_id)}</td><td class="mono pair-studies">${esc(r.source_study_id)} → ${esc(r.target_study_id)}<small>${esc(r.source_time.replace("T", " ").slice(0, 16))}</small></td><td>${badge(r.split)}</td><td>${esc(r.view)}</td><td>${r.hours < 72 ? r.hours.toFixed(1) + " h" : (r.hours / 24).toFixed(1) + " d"}</td><td>${r.label_flips}</td><td>${status(r.linkage)}<small>${esc(r.hadm_id)}</small></td><td><button class="row-open" data-patient="${r.subject_id}" data-transition="${r.transition_id}" aria-label="打开配对 ${r.transition_id}">查看 ↗</button></td></tr>`
            : `<tr><td><button class="patient-link mono" data-patient="${r.subject_id}">${esc(r.subject_id)}</button>${r.featured ? '<span class="featured-star" title="精选示例">☆</span>' : ""}</td><td>${coverage(r)}</td><td>${r.split ? badge(r.split) : '<span class="muted">—</span>'}</td><td>${fmt(r.studies)} <span class="muted">/ ${fmt(r.images)}</span></td><td>${fmt(r.admissions)} <span class="muted">/ ${fmt(r.icu_stays)}</span></td><td>${fmt(r.matched_studies)}</td><td>${fmt(r.pairs)}</td><td>${fmt(r.linked_pairs)}</td><td><button class="row-open" data-patient="${r.subject_id}" aria-label="查看患者 ${r.subject_id}">查看 ↗</button></td></tr>`,
        )
        .join("");
      $("patient-list").innerHTML =
        `<table class="cohort-table"><thead><tr>${headers.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows || `<tr><td colspan="${headers.length}" class="empty-state">${data.index_state === "loading" ? "全库匹配进行中，稍后自动更新。" : "没有匹配记录，请调整筛选条件。"}</td></tr>`}</tbody></table>`;
      $("range-info").textContent =
        `${data.total ? fmt((s.page - 1) * limit + 1) : 0}–${fmt(Math.min(s.page * limit, data.total))} / ${fmt(data.total)} 条`;
      $("page-info").textContent = `${s.page} / ${pages}`;
      $("page-prev").disabled = s.page <= 1;
      $("page-next").disabled = s.page >= pages;
    } catch (e) {
      if (token === s.token) $("patient-list").textContent = e.message;
    }
  }
  async function poll() {
    clearTimeout(s.timer);
    if (s.index?.state === "ready" || s.index?.state === "error") return;
    s.timer = setTimeout(async () => {
      try {
        s.index = await get("/api/cohort");
        metrics();
        if (s.index.state === "ready") list();
      } catch (e) {
        $("cohort-progress").textContent = e.message;
      }
      poll();
    }, 4000);
  }
  function show(mode = s.mode) {
    s.mode = mode;
    s.open = false;
    window.Atlas?.leavePatient();
    $("patient-content").hidden = true;
    $("cohort-content").hidden = false;
    $("startup").hidden = true;
    history.replaceState(null, "", "#" + (mode === "patients" ? "" : mode));
    list();
  }
  function reset() {
    for (const id of controls) {
      const el = $(id);
      if (el.type === "checkbox") el.checked = false;
      else
        el.value =
          id === "page-size"
            ? "25"
            : id === "sort-filter"
              ? "subject"
              : id === "patient-search"
                ? ""
                : "all";
    }
    s.page = 1;
  }
  window.CohortExplorer = {
    show,
    init(catalog) {
      s.catalog = catalog;
      s.index = catalog.cohort;
      if (window.MIMIC_SNAPSHOT) {
        $("cohort-content").hidden = true;
        return;
      }
      metrics();
      list();
      poll();
      for (const mode of ["patients", "pairs", "featured"])
        $(`nav-${mode}`).addEventListener("click", () => {
          reset();
          show(mode);
        });
      $("home-brand").addEventListener("click", (e) => {
        e.preventDefault();
        reset();
        show("patients");
      });
      $("back-cohort").addEventListener("click", () => show());
      let debounce;
      controls.forEach((id) =>
        $(id).addEventListener(
          id === "patient-search" ? "input" : "change",
          () => {
            clearTimeout(debounce);
            s.page = 1;
            ++s.token;
            debounce = setTimeout(list, id === "patient-search" ? 220 : 0);
          },
        ),
      );
      $("reset-filters").addEventListener("click", () => {
        reset();
        list();
      });
      $("page-prev").addEventListener("click", () => {
        s.page--;
        list();
      });
      $("page-next").addEventListener("click", () => {
        s.page++;
        list();
      });
      $("patient-list").addEventListener("click", (e) => {
        const b = e.target.closest("[data-patient]");
        if (b)
          window.Atlas.loadPatient(b.dataset.patient, b.dataset.transition);
      });
      $("coverage-bars").addEventListener("click", (e) => {
        const b = e.target.closest("[data-coverage]");
        if (b) {
          reset();
          $("coverage-filter").value = b.dataset.coverage;
          show("patients");
        }
      });
      $("split-summary").addEventListener("click", (e) => {
        const b = e.target.closest("[data-split]");
        if (b) {
          reset();
          $("split-filter").value = b.dataset.split;
          show("pairs");
        }
      });
      $("cohort-export").addEventListener("click", (e) => {
        if (s.index?.state !== "ready") e.preventDefault();
      });
      $("cohort-content").hidden = false;
      $("startup").hidden = true;
    },
  };
})();
