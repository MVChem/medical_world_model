/* No CDN dependencies. The same UI runs against the local API or an HTML snapshot. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
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
  const fmt = (n) => Number(n).toLocaleString("en-US");
  const when = (s) => (s ? String(s).replace("T", " ").slice(0, 16) : "未记录");
  const epoch = (s) =>
    s ? Date.parse(String(s).replace(" ", "T").replace(/Z$/, "") + "Z") : NaN;
  const hours = (n) =>
    n >= 72 ? `${(n / 24).toFixed(1)} 天` : `${n.toFixed(1)} 小时`;
  const snapshot = window.MIMIC_SNAPSHOT;
  const state = {
    catalog: null,
    patient: null,
    pair: 0,
    study: 0,
    tab: "reports",
    all: false,
    page: 1,
    searchToken: 0,
    patientToken: 0,
    timer: null,
    inverted: false,
    offsets: [
      { x: 0, y: 0 },
      { x: 0, y: 0 },
    ],
    singleImage: 0,
  };
  const labelNames = {
    present: "阳性",
    absent: "阴性",
    uncertain: "不确定",
    not_mentioned: "未提及",
  };
  const label = (value) =>
    `<span class="label-pill ${esc(value || "not_mentioned")}">${labelNames[value] || "未记录"}</span>`;

  async function api(url, options = {}) {
    const response = await fetch(url, { cache: "no-store", ...options });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(
        typeof data.detail === "string"
          ? data.detail
          : `请求失败 (${response.status})`,
      );
    }
    return response.json();
  }
  function notice(message) {
    $("notice").hidden = !message;
    $("notice").textContent = message;
  }
  const currentPair = () => state.patient?.transitions[state.pair] || null;
  function selectedStudies() {
    const packet = currentPair()?.mimic_cxr_transition;
    return packet
      ? ["current_state", "future_state"].map((k) =>
          state.patient.studies.find((s) => s.study_id === packet[k].study_id),
        )
      : [state.patient.studies[state.study]].filter(Boolean);
  }
  function selectedImage(study, side) {
    const packet = currentPair()?.mimic_cxr_transition;
    const id =
      packet?.[side === 0 ? "current_state" : "future_state"].image.dicom_id;
    return id
      ? study.images.find((im) => im.dicom_id === id)
      : study.images[state.singleImage] || study.images[0];
  }
  async function loadPatient(pid, transition = "") {
    window.ClinicalExplorer?.pause();
    $("cohort-content").hidden = true;
    state.selectionAbort?.abort();
    const token = ++state.patientToken;
    clearTimeout(state.timer);
    $("patient-content").hidden = true;
    $("startup").hidden = false;
    $("startup-stage").textContent = `正在读取患者 ${pid} 的影像和报告…`;
    notice(
      snapshot
        ? "离线 HTML · 仅包含导出的当前片段；顶部计数为导出时的原始数据目录规模。"
        : "",
    );
    try {
      const patient = snapshot
        ? snapshot.patient
        : await api(
            "/api/patients/" + encodeURIComponent(pid) + "?compact=true",
          );
      if (token !== state.patientToken) return;
      state.patient = patient;
      state.pair = patient.transitions.findIndex(
        (p) => p.transition_id === (transition || patient.preferred_transition),
      );
      if (state.pair < 0 && patient.transitions.length) state.pair = 0;
      state.study = 0;
      state.singleImage = 0;
      $("startup").hidden = true;
      $("patient-content").hidden = false;
      renderPatient();
      if (!snapshot) {
        history.replaceState(
          null,
          "",
          "#subject=" +
            pid +
            (transition ? "&transition=" + encodeURIComponent(transition) : ""),
        );
        pollClinical(pid, token);
      }
    } catch (error) {
      if (token !== state.patientToken) return;
      $("startup").hidden = true;
      notice(`读取失败：${error.message}。可重新选择患者或刷新页面重试。`);
    }
  }
  function pollClinical(pid, token) {
    if (state.patient.clinical_status.state !== "loading") return;
    state.timer = setTimeout(async () => {
      try {
        const status = await api(
          "/api/patients/" + encodeURIComponent(pid) + "/clinical-status",
        );
        if (token !== state.patientToken) return;
        state.patient.clinical_status = status;
        if (status.state === "ready") await renderSelection(true);
        else renderClinical();
        updateExports();
        pollClinical(pid, token);
      } catch (error) {
        if (token !== state.patientToken) return;
        notice(`临床背景加载中断：${error.message}。正在重试…`);
        pollClinical(pid, token);
      }
    }, 4000);
  }
  function renderPatient() {
    const p = state.patient;
    $("subject-id").textContent = p.subject_id;
    $("split-badge").textContent = p.split || "IV only";
    $("iv-only-note").hidden = Boolean(p.studies.length);
    document.querySelector(".study-panel").hidden = !p.studies.length;
    document.querySelector(".viewer-section").hidden = !p.studies.length;
    if (!p.studies.length) state.tab = "observations";
    $("patient-meta").textContent = p.studies.length
      ? `${p.studies.length} 次检查 · ${p.transitions.length} 个可用相邻片段 · ${when(p.studies[0].timestamp).slice(0, 10)} 至 ${when(p.studies.at(-1).timestamp).slice(0, 10)}`
      : "MIMIC-IV 患者 · 可浏览全部可用临床数据表";
    $("study-count").textContent = `${p.studies.length} STUDIES`;
    $("pair-select").innerHTML = p.transitions.length
      ? p.transitions
          .map((r, i) => {
            const t = r.mimic_cxr_transition;
            return `<option value="${i}">${t.curation ? "✧ " : ""}${esc(t.current_state.study_id)} → ${esc(t.future_state.study_id)} · ${hours(t.interval.elapsed_hours)}</option>`;
          })
          .join("")
      : '<option value="-1">没有符合规则的相邻配对 · 可浏览单次检查</option>';
    if (state.pair < 0 && p.transitions.length)
      $("pair-select").insertAdjacentHTML(
        "afterbegin",
        '<option value="-1">单次检查浏览</option>',
      );
    renderSelection();
  }
  async function renderSelection(detailsOnly = false) {
    detailsOnly = detailsOnly && !state.selecting;
    state.selecting = !detailsOnly;
    const token = (state.selectionToken || 0) + 1;
    state.selectionToken = token;
    state.selectionAbort?.abort();
    state.selectionAbort = new AbortController();
    if (!detailsOnly) {
      window.ClinicalExplorer?.pause();
      $("image-grid").innerHTML =
        '<div class="selection-loading" role="status">正在读取所选检查…</div>';
      $("export-html").disabled = true;
      for (const id of [
        "reports",
        "labels",
        "clinical",
        "observations",
        "provenance",
      ])
        $(id + "-panel").innerHTML =
          '<div class="empty-state">正在读取所选检查…</div>';
    }
    if (!snapshot) {
      const pid = state.patient.subject_id;
      const params = new URLSearchParams({
        transition: currentPair()?.transition_id || "",
        study: selectedStudies()[0]?.study_id || "",
      });
      try {
        const data = await api(`/api/patients/${pid}/selection?${params}`, {
          signal: state.selectionAbort.signal,
        });
        if (token !== state.selectionToken || pid !== state.patient?.subject_id)
          return;
        for (const study of data.studies) {
          const index = state.patient.studies.findIndex(
            (s) => s.study_id === study.study_id,
          );
          if (index >= 0) state.patient.studies[index] = study;
        }
        if (data.transition && state.pair >= 0)
          state.patient.transitions[state.pair] = data.transition;
        for (const key of [
          "admissions",
          "stays",
          "transfers",
          "clinical_status",
          "icu_inputs",
        ])
          state.patient[key] = data[key];
      } catch (e) {
        if (e.name !== "AbortError") notice(e.message);
        return;
      }
    }
    state.selecting = false;
    if (detailsOnly) {
      renderDetails();
      updateExports();
      return;
    }

    $("pair-select").value = String(state.pair);
    $("pair-prev").disabled = state.pair <= 0;
    $("pair-next").disabled =
      state.pair >= state.patient.transitions.length - 1 ||
      !state.patient.transitions.length;
    const studies = selectedStudies();
    $("study-timeline").innerHTML = state.patient.studies
      .map(
        (s, i) =>
          `<button class="study-node ${s.study_id === studies[0]?.study_id ? "is-current" : s.study_id === studies[1]?.study_id ? "is-future" : ""}" data-study="${i}" aria-label="检查 ${esc(s.study_id)}"><div class="study-date">${esc(when(s.timestamp))}</div><div class="study-id">${esc(s.study_id)}</div><span class="study-view">${esc([...new Set(s.images.map((im) => im.view))].join(" / "))} · ${s.images.length} 张影像</span></button>`,
      )
      .join("");
    const packet = currentPair()?.mimic_cxr_transition;
    $("interval-badge").innerHTML = packet
      ? `间隔 <b>${hours(packet.interval.elapsed_hours)}</b> &nbsp; / &nbsp; ${esc(packet.interval.horizon_bin)} · ${esc(packet.matched_view)}`
      : "单次检查 · 包含全部投照体位";
    resetTools();
    renderImages();
    renderDetails();
    updateExports();
    const track = $("study-timeline"),
      node = track.querySelector(".is-current");
    if (node)
      track.scrollLeft +=
        node.getBoundingClientRect().left -
        track.getBoundingClientRect().left -
        track.clientWidth / 2 +
        node.clientWidth / 2;
  }
  function imageURL(im, large = false) {
    if (snapshot || im.url.startsWith("data:")) return im.url;
    if (large)
      return im.full_url || `/api/images/${im.dicom_id}?size=1800&quality=88`;
    const size = $("image-quality").value;
    return `/api/images/${im.dicom_id}?size=${size}&format=webp&quality=${size === "512" ? 60 : 78}`;
  }
  function renderImages() {
    const studies = selectedStudies();
    $("image-grid").classList.toggle("single", studies.length === 1);
    $("image-grid").innerHTML = studies
      .map((study, side) => {
        const im = selectedImage(study, side);
        const available = im?.available;
        return `<div class="image-card ${side ? "future" : "current"}"><div class="image-caption"><span class="role">${side ? "T₁ / OBSERVED FOLLOW-UP" : studies.length > 1 ? "T₀ / CURRENT STATE" : "SELECTED STUDY"}</span><time>${esc(when(im?.timestamp || study.timestamp))}</time></div><div class="image-stage" data-side="${side}"><div class="image-tags">${esc(study.study_id)}<br>${esc(im?.view || "无影像")}</div>${available ? `<img class="scan" data-side="${side}" src="${esc(imageURL(im))}" decoding="async" alt="患者 ${esc(state.patient.subject_id)}，检查 ${esc(study.study_id)}，${esc(im.view)} 胸片" draggable="false"><span class="image-loading">读取影像…</span><button class="expand-image" data-expand="${side}" aria-label="放大${side ? "随访" : "当前"}胸片">⛶</button>` : '<div class="image-unavailable">此影像文件在本地不可用</div>'}</div><div class="image-bottom"><span>${available ? `${im.columns ?? "—"} × ${im.rows ?? "—"} px` : "IMAGE UNAVAILABLE"}</span>${studies.length === 1 && study.images.length > 1 ? `<select id="single-image" aria-label="选择投照影像">${study.images.map((m, i) => `<option value="${i}" ${i === state.singleImage ? "selected" : ""}>${esc(m.view)} · ${esc(m.dicom_id.slice(0, 8))}${m.available ? "" : " · 缺失"}</option>`).join("")}</select>` : `<span>${side ? "随访观测 / 评估用" : "当前影像"} · MIMIC-CXR</span>`}</div></div>`;
      })
      .join("");
    document.querySelectorAll(".scan").forEach((img) => {
      const loaded = () => {
        if (img.naturalWidth) img.classList.add("loaded");
      };
      img.addEventListener("load", loaded);
      img.addEventListener("error", () => {
        img.hidden = true;
        img.nextElementSibling.textContent = "影像读取失败";
        img.closest(".image-stage").querySelector(".expand-image")?.remove();
      });
      if (img.complete) loaded();
      let drag = null;
      img.addEventListener("pointerdown", (e) => {
        if (Number($("zoom").value) <= 1) return;
        e.preventDefault();
        img.setPointerCapture(e.pointerId);
        const offset = state.offsets[Number(img.dataset.side)];
        drag = { x: e.clientX - offset.x, y: e.clientY - offset.y };
      });
      img.addEventListener("pointermove", (e) => {
        if (!drag) return;
        const offset = { x: e.clientX - drag.x, y: e.clientY - drag.y };
        state.offsets = [{ ...offset }, { ...offset }];
        applyTools();
      });
      img.addEventListener("pointerup", () => {
        drag = null;
      });
      img.addEventListener("pointercancel", () => {
        drag = null;
      });
    });
    $("single-image")?.addEventListener("change", (e) => {
      state.singleImage = Number(e.target.value);
      resetTools();
      renderImages();
    });
    applyTools();
  }
  function resetTools() {
    for (const key of ["zoom", "brightness", "contrast"]) $(key).value = "1";
    state.inverted = false;
    state.offsets = [
      { x: 0, y: 0 },
      { x: 0, y: 0 },
    ];
    applyTools();
  }
  function applyTools() {
    const zoom = Number($("zoom").value);
    if (zoom === 1)
      state.offsets = [
        { x: 0, y: 0 },
        { x: 0, y: 0 },
      ];
    $("zoom-value").textContent = zoom.toFixed(1) + "×";
    $("invert").setAttribute("aria-pressed", state.inverted);
    document.querySelectorAll(".scan").forEach((img, i) => {
      const offset = state.offsets[i];
      img.style.transform = `translate(${offset.x}px,${offset.y}px) scale(${zoom})`;
      img.style.filter = `brightness(${$("brightness").value}) contrast(${$("contrast").value}) invert(${state.inverted ? 1 : 0})`;
      img.closest(".image-stage").style.touchAction =
        zoom > 1 ? "none" : "pan-y";
    });
  }
  function renderDetails() {
    const p = state.patient,
      pair = currentPair(),
      packet = pair?.mimic_cxr_transition;
    const studies = selectedStudies();
    $("reports-panel").innerHTML = `<div class="reports-grid">${studies
      .map((s, i) => {
        const report = s.report || {};
        const sections = [
          ["findings", "FINDINGS"],
          ["impression", "IMPRESSION"],
          ["unsectioned_report", "REPORT"],
        ].filter(
          ([key]) =>
            report[key] &&
            (key !== "unsectioned_report" ||
              (!report.findings && !report.impression)),
        );
        return `<div><div class="report-head ${i ? "future" : ""}"><i class="dot ${i ? "future" : "current"}"></i>${i ? "随访报告 · 评估用" : "当前报告"}<span class="muted">${esc(s.study_id)}</span></div>${sections.map(([key, title]) => `<div class="report-section"><h4>${title}</h4><p class="${key === "impression" ? "impression" : ""}">${esc(report[key])}</p></div>`).join("") || '<div class="empty-state">该检查未提供可读报告。</div>'}</div>`;
      })
      .join(
        "",
      )}</div>${packet?.curation ? `<div class="audit-note"><b>精选片段备注</b>${esc(packet.curation.audit_note)}<p>按报告和随访挑选的演示案例，非随机队列或临床裁定。</p></div>` : ""}`;
    const changed =
      packet?.state_delta_for_evaluation_only?.num_binary_chexpert_label_flips;
    const labels = Object.keys(studies[0]?.labels || {});
    const changes = labels.filter(
      (k) =>
        ["present", "absent"].includes(studies[0].labels[k]) &&
        ["present", "absent"].includes(studies[1]?.labels[k]) &&
        studies[0].labels[k] !== studies[1].labels[k] &&
        k !== "No Finding",
    );
    $("change-count").textContent = changed ?? changes.length;
    $("labels-panel").innerHTML = `<div class="legend">${Object.keys(labelNames)
      .map((k) => label(k))
      .join(
        "",
      )}<span>空值保留为“未提及”，不当作阴性。</span></div><div class="table-wrap"><table class="label-table"><thead><tr><th>CHEXPERT FINDING</th><th>当前</th>${studies[1] ? "<th>随访</th><th>变化</th>" : ""}</tr></thead><tbody>${labels.map((k) => `<tr class="${changes.includes(k) ? "changed-row" : ""}"><td>${esc(k)}</td><td>${label(studies[0].labels[k])}</td>${studies[1] ? `<td>${label(studies[1].labels[k])}</td><td>${changes.includes(k) ? (studies[1].labels[k] === "present" ? "阴性 → 阳性" : "阳性 → 阴性") : studies[0].labels[k] === studies[1].labels[k] ? '<span class="muted">相同</span>' : '<span class="muted">标注状态变化</span>'}</td>` : ""}</tr>`).join("")}</tbody></table></div><p class="table-note">报告自动标注的变化不等同于经验证的疾病发生、消退或严重程度变化。</p>`;
    renderClinical();
    window.ClinicalExplorer?.update({
      patient: p,
      selection: {
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
      },
      snapshot: snapshot?.extended,
    });
    renderProvenance();
    activateTab(state.tab);
  }
  function clinicalTimeline() {
    const pair = currentPair()?.mimic_cxr_transition;
    const studies = selectedStudies();
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
      [
        "住院",
        state.patient.admissions,
        "admittime",
        "dischtime",
        "#d1e8e7",
        70,
      ],
      ["ICU", state.patient.stays, "intime", "outtime", "#a6cbd2", 107],
      [
        "病区转移",
        state.patient.transfers,
        "intime",
        "outtime",
        "#dce3ed",
        144,
      ],
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
  function renderClinical() {
    const status = state.patient.clinical_status;
    $("clinical-dot").className = "status-dot " + status.state;
    if (status.state !== "ready") {
      $("clinical-panel").innerHTML =
        `<div class="clinical-status">${status.state === "loading" ? '<span class="spinner"></span>' : ""}${esc(status.stage)}${status.error ? `<p>${esc(status.error)}</p>` : '<p class="table-note">正在读取当前患者的临床记录。影像、报告和标签可先浏览。</p>'}</div>`;
      return;
    }
    const pair = currentPair();
    const context = pair?.retrospective_context_for_audit_only;
    const link = pair?.linkage_status;
    const studyLink = selectedStudies()[0]?.admission_link;
    if (!state.patient.studies.length) {
      $("clinical-panel").innerHTML =
        `<h3>住院记录 · ${state.patient.admissions.length}</h3><div class="table-wrap"><table><thead><tr><th>住院号</th><th>入院</th><th>出院</th><th>类型</th></tr></thead><tbody>${state.patient.admissions.map((a) => `<tr><td>${esc(a.hadm_id)}</td><td>${esc(when(a.admittime))}</td><td>${esc(when(a.dischtime))}</td><td>${esc(a.admission_type)}</td></tr>`).join("")}</tbody></table></div><p class="table-note">该患者没有 CXR。检验及其他记录可在“检验 · 生命体征 · 更多”中按患者或住院浏览。</p>`;
      renderRawContext();
      return;
    }
    const names = {
      unique_common_admission: "唯一共同住院",
      unmatched: "未匹配共同住院",
      ambiguous: "存在多条候选住院",
    };
    const title = pair
      ? names[link] || link
      : `单次检查 · ${studyLink?.status === "unique" ? "已匹配住院" : studyLink?.status === "ambiguous" ? "住院匹配有歧义" : "未匹配住院"}`;
    $("clinical-panel").innerHTML =
      `<div class="clinical-summary"><span class="status-badge ${esc(link)}">${esc(title)}</span><div class="kv"><small>HADM_ID</small>${esc(context?.admission.hadm_id || (pair ? pair.candidate_hadm_ids?.join(", ") : studyLink?.hadm_ids.join(", ")) || "—")}</div><div class="kv"><small>当前 ICU STAY</small>${esc(context?.source_location?.stay?.stay_id || "—")}</div><div class="kv"><small>当前病区</small>${esc(context?.source_location?.careunit || "—")}</div><div class="kv"><small>随访病区</small>${esc(context?.target_location?.careunit || "—")}</div></div><div class="chart-scroll">${clinicalTimeline()}</div><p class="table-note">时间轴按所选胸片窗口裁剪；悬停查看原始区间。住院和 ICU 记录来自该患者全部记录，缺少区间端点的记录不绘制。</p>${pair && !context ? `<div class="empty-state">${link === "ambiguous" ? "多个住院区间同时覆盖两次胸片，保留候选记录，不自动选取。" : "两次胸片不在同一个可唯一匹配的住院区间内，不拼接临床事件。"}</div>` : !pair ? '<p class="table-note">选择一个相邻片段，可查看片段内的临床事件、诊断和操作。</p>' : `<div class="clinical-toolbar"><h3>片段内 ICU 事件</h3><div><input id="event-search" type="search" placeholder="搜索事件" aria-label="搜索临床事件"><select id="event-filter" aria-label="事件类型"><option value="all">全部事件</option><option value="procedure">操作 / 支持</option><option value="input">输入 / 用药</option></select></div></div><div id="event-table" class="event-table"></div><p class="table-note">${state.patient.icu_inputs ? "已读取 ICU inputevents；数量和速率保留原始单位。" : "ICU inputevents 尚未加载；需以 --include-icu-inputs 启动。未加载不代表无用药。"} 事件为同期观测，不表示治疗导致影像变化。</p><div class="clinical-codes"><div><h4>住院诊断 <span class="muted">出院编码 · ${context.admission_diagnoses.length}</span></h4>${codeList(context.admission_diagnoses)}</div><div><h4>住院操作 <span class="muted">日期级精度 · ${context.admission_procedures.length}</span></h4>${codeList(context.admission_procedures)}</div></div>`}`;
    $("event-search")?.addEventListener("input", renderEvents);
    $("event-filter")?.addEventListener("change", renderEvents);
    if (context && !state.patient.icu_inputs && !snapshot) {
      $("event-filter").insertAdjacentHTML(
        "afterend",
        '<button id="load-inputs" class="button">加载输入 / 用药</button>',
      );
      $("load-inputs").addEventListener("click", async () => {
        const pid = state.patient.subject_id,
          token = state.patientToken;
        $("load-inputs").disabled = true;
        try {
          const response = await fetch(`/api/patients/${pid}/icu-inputs`, {
            method: "POST",
          });
          if (!response.ok) throw new Error("输入事件加载请求失败");
          const patient = await response.json();
          if (token !== state.patientToken) return;
          state.patient = patient;
          renderDetails();
          updateExports();
          pollClinical(pid, token);
        } catch (error) {
          if (token === state.patientToken) {
            notice(error.message);
            renderClinical();
          }
        }
      });
    }
    if (context) renderEvents();
    renderRawContext();
  }
  function renderRawContext() {
    if (snapshot) return;
    const pid = state.patient.subject_id;
    $("clinical-panel").insertAdjacentHTML(
      "beforeend",
      `<details class="raw-context"><summary>患者全部住院记录 · 原始表</summary><p class="table-note">包含片段之外的原始记录；每页 25 条，点击一行展开完整字段。</p><div class="raw-context-controls"><select id="raw-context-table" aria-label="原始住院表">${[
        ["admissions", "住院"],
        ["transfers", "病区转移"],
        ["diagnoses_icd", "诊断 ICD"],
        ["procedures_icd", "操作 ICD"],
        ["icustays", "ICU 住院"],
        ["procedureevents", "ICU 操作"],
        ["inputevents", "ICU 输入 / 用药"],
      ]
        .map(([k, v]) => `<option value="${k}">${v} · ${k}</option>`)
        .join(
          "",
        )}</select><input id="raw-context-search" type="search" placeholder="搜索原始字段" aria-label="搜索住院原始记录"><a class="button" id="raw-context-export">CSV ↓</a></div><div id="raw-context-rows"></div><div class="observation-pagination"><button id="raw-context-prev">←</button><span id="raw-context-page"></span><button id="raw-context-next">→</button></div></details>`,
    );
    let page = 1,
      token = 0,
      debounce;
    const refresh = async () => {
      const current = ++token;
      const name = $("raw-context-table")?.value;
      if (!name) return;
      const q = $("raw-context-search").value;
      const url = `/api/patients/${pid}/context/${name}?${new URLSearchParams({ page, q })}`;
      $("raw-context-export").href = url + "&download=true";
      try {
        const data = await api(url);
        if (
          current !== token ||
          pid !== state.patient?.subject_id ||
          !$("raw-context-rows")
        )
          return;
        $("raw-context-rows").innerHTML =
          data.rows
            .map(
              (row, i) =>
                `<details class="raw-context-row"><summary><span>${(page - 1) * 25 + i + 1}</span>${esc(
                  Object.entries(row)
                    .filter(([k]) => k !== "subject_id")
                    .slice(0, 5)
                    .map(([k, v]) => `${k}: ${v || "—"}`)
                    .join(" · "),
                )}</summary><pre>${esc(JSON.stringify(row, null, 2))}</pre></details>`,
            )
            .join("") || '<p class="empty-state">此表没有匹配记录。</p>';
        $("raw-context-page").textContent =
          `${page} / ${Math.max(1, Math.ceil(data.total / 25))} · ${fmt(data.total)} 条`;
        $("raw-context-prev").disabled = page <= 1;
        $("raw-context-next").disabled = page * 25 >= data.total;
      } catch (e) {
        if (
          current !== token ||
          pid !== state.patient?.subject_id ||
          !$("raw-context-rows")
        )
          return;
        $("raw-context-rows").textContent = e.message;
        $("raw-context-page").textContent = "";
        $("raw-context-prev").disabled = $("raw-context-next").disabled = true;
        if (name === "inputevents" && !state.patient.icu_inputs) {
          $("raw-context-rows").insertAdjacentHTML(
            "beforeend",
            '<button id="raw-load-inputs" class="button">加载输入事件</button>',
          );
          $("raw-load-inputs").onclick = async () => {
            try {
              await api(`/api/patients/${pid}/icu-inputs`, { method: "POST" });
              if (pid !== state.patient?.subject_id) return;
              state.patient.clinical_status = {
                state: "loading",
                stage: "加载 ICU 输入事件",
              };
              renderClinical();
              updateExports();
              pollClinical(pid, state.patientToken);
            } catch (error) {
              notice(error.message);
            }
          };
        }
      }
    };
    document.querySelector(".raw-context").addEventListener("toggle", (e) => {
      if (e.target.open) refresh();
    });
    $("raw-context-table").onchange = () => {
      page = 1;
      refresh();
    };
    $("raw-context-search").oninput = () => {
      clearTimeout(debounce);
      ++token;
      debounce = setTimeout(() => {
        page = 1;
        refresh();
      }, 250);
    };
    $("raw-context-prev").onclick = () => {
      page--;
      refresh();
    };
    $("raw-context-next").onclick = () => {
      page++;
      refresh();
    };
  }
  function codeList(rows) {
    return rows.length
      ? `<ol>${rows.map((r) => `<li><code>ICD-${esc(r.icd_version)} ${esc(r.icd_code)}</code>${esc(r.title || "无编码释义")}${r.chartdate ? ` <span class="muted">${esc(r.chartdate)}</span>` : ""}</li>`).join("")}</ol>`
      : '<div class="muted">没有记录</div>';
  }
  function renderEvents() {
    const ctx = currentPair()?.retrospective_context_for_audit_only;
    if (!ctx || !$("event-table")) return;
    const filter = $("event-filter").value,
      query = $("event-search").value.toLowerCase();
    const all = [
      ...ctx.interval_icu_procedureevents.map((e) => ({
        ...e,
        kind: "procedure",
      })),
      ...(ctx.interval_icu_inputevents || []).map((e) => ({
        ...e,
        kind: "input",
      })),
    ].sort((a, b) => String(a.starttime).localeCompare(String(b.starttime)));
    const rows = all.filter(
      (e) =>
        (filter === "all" || e.kind === filter) &&
        `${e.label} ${e.category} ${e.itemid}`.toLowerCase().includes(query),
    );
    $("event-table").innerHTML = rows.length
      ? `<table><thead><tr><th>时间</th><th>事件 · ${rows.length} / ${all.length}</th><th>记录值 / 速率</th><th>时间关系</th></tr></thead><tbody>${rows.map((e) => `<tr><td>${esc(when(e.starttime))}<br><span class="muted">→ ${esc(when(e.endtime))}</span></td><td>${esc(e.label || e.itemid)}<br><span class="event-type">${e.kind === "input" ? "INPUT" : "PROCEDURE"} · ${esc(e.itemid)}</span></td><td>${esc(e.kind === "input" ? [e.amount, e.amount_unit].filter(Boolean).join(" ") || "—" : [e.value, e.value_unit].filter(Boolean).join(" ") || "—")}${e.rate ? `<br><span class="muted">${esc(e.rate)} ${esc(e.rate_unit)}</span>` : ""}</td><td><span class="muted">${esc(e.temporal_role)}<br>${esc(e.status || "")}</span></td></tr>`).join("")}</tbody></table>`
      : `<div class="empty-state">${filter === "input" && !state.patient.icu_inputs ? "ICU inputevents 未加载" : "已加载事件中没有匹配记录"}</div>`;
  }
  function renderProvenance() {
    const p = state.patient;
    $("provenance-panel").innerHTML =
      `<div class="table-wrap"><table class="provenance-table"><thead><tr><th>数据</th><th>来源与连接方式</th><th>使用范围</th></tr></thead><tbody><tr><td>影像 / study_id / dicom_id</td><td><a href="https://physionet.org/content/mimic-cxr-jpg/2.0.0/" target="_blank" rel="noreferrer">MIMIC-CXR-JPG 2.0.0 ↗</a></td><td>当前影像为输入；随访为观测目标</td></tr><tr><td>报告 / CheXpert / split</td><td>原始报告、官方标签和患者级划分</td><td>标签区分阳性、阴性、不确定、未提及</td></tr><tr><td>住院 / ICU / 操作 / 输入事件</td><td><a href="https://physionet.org/content/mimiciv/3.1/" target="_blank" rel="noreferrer">MIMIC-IV 3.1 ↗</a><br>subject_id 相等 + 采集时刻包含于住院区间</td><td>回顾性背景，非当前状态预测输入</td></tr><tr><td>影像配对</td><td>完整患者时间线相邻检查；同一 AP / PA 投照</td><td>1 小时至 365 天；不跨过中间检查</td></tr><tr><td>临床表覆盖</td><td>admissions · transfers · diagnoses_icd · procedures_icd · icustays · procedureevents${p.icu_inputs ? " · inputevents" : ""}</td><td>检验、生命体征及其他 15 张扩展表在“检验 · 生命体征 · 更多”页按需浏览</td></tr></tbody></table></div><details class="audit-note"><summary>查看配对筛选计数</summary><pre>${esc(JSON.stringify(p.pairing_audit, null, 2))}</pre></details><p class="table-note">study_id 和 hadm_id 不是关联键。所有检查均保留在时间线上；缺失影像、报告或不符合规则的检查不会被偷偷跨过。</p>`;
  }
  function activateTab(tab) {
    state.tab = tab;
    window.ClinicalExplorer?.setActive(tab === "observations");
    document.querySelectorAll("[data-tab]").forEach((b) => {
      const selected = b.dataset.tab === tab;
      b.setAttribute("aria-selected", selected);
      $(b.dataset.tab + "-panel").hidden = !selected;
    });
  }
  function updateExports() {
    $("export-html").disabled =
      Boolean(snapshot) ||
      !currentPair() ||
      state.patient.clinical_status.state !== "ready";
    $("export-html").title =
      state.patient.clinical_status.state !== "ready"
        ? "等待 MIMIC-IV 加载完成"
        : "内嵌当前配对的影像，可离线打开";
  }
  async function download(format) {
    const pid = state.patient.subject_id;
    try {
      let blob, filename;
      if (snapshot && format === "json") {
        blob = new Blob([JSON.stringify(state.patient, null, 2)], {
          type: "application/json",
        });
        filename = `mimic-${pid}.json`;
      } else {
        const tid = currentPair()?.transition_id || "";
        const response = await fetch(
          `/api/export/${pid}.${format}?transition=${encodeURIComponent(tid)}`,
        );
        if (!response.ok)
          throw new Error((await response.json()).detail || "导出失败");
        blob = await response.blob();
        filename = format === "html" ? `${tid}.html` : `mimic-${pid}.json`;
      }
      const url = URL.createObjectURL(blob),
        a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
    } catch (error) {
      notice(error.message);
    }
  }
  function selectPair(index) {
    state.pair = index;
    renderSelection();
  }
  function bind() {
    $("image-quality").addEventListener("change", () => {
      if (state.patient) renderImages();
    });
    $("pair-prev").addEventListener("click", () => selectPair(state.pair - 1));
    $("pair-next").addEventListener("click", () => selectPair(state.pair + 1));
    $("study-timeline").addEventListener("click", (e) => {
      const node = e.target.closest("[data-study]");
      if (!node) return;
      state.study = Number(node.dataset.study);
      state.pair = -1;
      state.singleImage = 0;
      if (!$("pair-select").querySelector('option[value="-1"]'))
        $("pair-select").insertAdjacentHTML(
          "afterbegin",
          '<option value="-1">单次检查浏览</option>',
        );
      renderSelection();
    });
    document.querySelectorAll("[data-tab]").forEach((b) => {
      b.addEventListener("click", () => activateTab(b.dataset.tab));
      b.addEventListener("keydown", (e) => {
        if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
        e.preventDefault();
        const tabs = [...document.querySelectorAll("[data-tab]")];
        const next =
          tabs[
            (tabs.indexOf(b) + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) %
              tabs.length
          ];
        next.focus();
        activateTab(next.dataset.tab);
      });
    });
    for (const id of ["zoom", "brightness", "contrast"])
      $(id).addEventListener("input", applyTools);
    $("invert").addEventListener("click", () => {
      state.inverted = !state.inverted;
      applyTools();
    });
    $("reset-image").addEventListener("click", resetTools);
    $("image-grid").addEventListener("click", (e) => {
      const button = e.target.closest("[data-expand]");
      if (!button) return;
      const index = Number(button.dataset.expand),
        study = selectedStudies()[index],
        im = selectedImage(study, index);
      $("dialog-image").src = imageURL(im, true);
      $("dialog-image").style.filter =
        document.querySelectorAll(".scan")[index]?.style.filter || "";
      $("dialog-caption").textContent =
        `${study.study_id} · ${im.view} · ${when(im.timestamp)}`;
      $("image-dialog").showModal();
    });
    $("close-dialog").addEventListener("click", () =>
      $("image-dialog").close(),
    );
    $("image-dialog").addEventListener("click", (e) => {
      if (e.target === $("image-dialog")) $("image-dialog").close();
    });
    $("export-json").addEventListener("click", () => download("json"));
    $("export-html").addEventListener("click", () => download("html"));
    window.addEventListener("keydown", (e) => {
      if (
        !state.patient ||
        /INPUT|SELECT|TEXTAREA|BUTTON/.test(e.target.tagName) ||
        $("image-dialog").open
      )
        return;
      if (e.key === "ArrowLeft" && state.pair > 0) {
        e.preventDefault();
        selectPair(state.pair - 1);
      }
      if (
        e.key === "ArrowRight" &&
        state.pair < state.patient.transitions.length - 1
      ) {
        e.preventDefault();
        selectPair(state.pair + 1);
      }
    });
  }
  async function init() {
    bind();
    if (snapshot) {
      $("connection-label").textContent = "离线 HTML 快照";
      $("mode-badge").textContent = "OFFLINE SNAPSHOT";
      $("image-quality").disabled = true;
      $("back-cohort").hidden = true;
      for (const id of ["nav-patients", "nav-pairs", "nav-featured"])
        $(id).disabled = true;
    }
    try {
      const catalog = snapshot ? snapshot.catalog : await api("/api/catalog");
      if (catalog.state === "error") {
        $("startup-stage").textContent = catalog.error;
        return;
      }
      if (catalog.state !== "ready") {
        $("startup-stage").textContent = catalog.stage;
        setTimeout(initCatalog, 900);
        return;
      }
      readyCatalog(catalog);
    } catch (error) {
      $("startup-stage").textContent = `${error.message}，正在重试…`;
      setTimeout(initCatalog, 1800);
    }
  }
  async function initCatalog() {
    try {
      const data = await api("/api/catalog");
      if (data.state === "error") {
        $("startup-stage").textContent = data.error;
        return;
      }
      if (data.state !== "ready") {
        $("startup-stage").textContent = data.stage;
        setTimeout(initCatalog, 900);
        return;
      }
      readyCatalog(data);
    } catch (error) {
      $("startup-stage").textContent = `${error.message}，正在重试…`;
      setTimeout(initCatalog, 1800);
    }
  }
  function readyCatalog(catalog) {
    state.catalog = catalog;
    window.CohortExplorer.init(catalog);
    const hash = new URLSearchParams(location.hash.slice(1));
    const pid = snapshot ? snapshot.patient.subject_id : hash.get("subject");
    if (pid) loadPatient(pid, hash.get("transition") || "");
    else if (["pairs", "featured"].includes(location.hash.slice(1)))
      window.CohortExplorer.show(location.hash.slice(1));
  }
  window.Atlas = {
    loadPatient,
    leavePatient() {
      ++state.patientToken;
      ++state.selectionToken;
      clearTimeout(state.timer);
      state.selectionAbort?.abort();
      window.ClinicalExplorer?.pause();
      state.patient = null;
      notice("");
    },
  };
  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", init);
  else init();
})();
