import { useEffect, useState } from "react";
import { api } from "./api";
import { fmt } from "./format";
import Cohort from "./components/Cohort";
import Patient from "./components/Patient";
import Exports from "./components/Exports";
import DatasetIntro from "./components/DatasetIntro";
import Overview from "./components/Overview";

const snapshot = window.MIMIC_SNAPSHOT;
function currentRoute() {
  if (snapshot)
    return {
      subject: snapshot.patient.subject_id,
      transition: snapshot.patient.preferred_transition,
    };
  const hash = location.hash.slice(1),
    params = new URLSearchParams(hash),
    page = hash.split("&")[0];
  const filters = {};
  for (const [name, values] of Object.entries({
    coverage: ["all", "cxr", "matched", "cxr_only", "iv_only"],
    split: ["all", "train", "validate", "test", "none"],
  })) {
    if (values.includes(params.get(name))) filters[name] = params.get(name);
  }
  return params.has("subject")
    ? {
        subject: params.get("subject"),
        transition: params.get("transition") || "",
      }
    : {
        mode: ["overview", "patients", "pairs", "featured", "exports"].includes(page)
          ? page
          : "overview",
        filters,
      };
}
export default function App() {
  const [catalog, setCatalog] = useState(snapshot?.catalog || null);
  const [error, setError] = useState("");
  const [route, setRoute] = useState(currentRoute);
  const [browseRoute, setBrowseRoute] = useState(
    route.mode ? route : { mode: "patients" },
  );
  const mode = browseRoute.mode;
  const [quality, setQuality] = useState("512");
  useEffect(() => {
    if (snapshot) return;
    let disposed = false,
      timer;
    const controller = new AbortController();
    async function read() {
      try {
        const next = await api("/api/catalog", { signal: controller.signal });
        if (disposed) return;
        setCatalog(next);
        setError("");
        if (next.state !== "error" && next.cohort?.state !== "ready")
          timer = setTimeout(read, 1000);
      } catch (e) {
        if (!disposed) {
          setError(e.message);
          timer = setTimeout(read, 2500);
        }
      }
    }
    read();
    return () => {
      disposed = true;
      controller.abort();
      clearTimeout(timer);
    };
  }, []);
  useEffect(() => {
    const changed = () => {
      const next = currentRoute();
      setRoute(next);
      if (next.mode) setBrowseRoute(next);
    };
    addEventListener("hashchange", changed);
    addEventListener("popstate", changed);
    return () => {
      removeEventListener("hashchange", changed);
      removeEventListener("popstate", changed);
    };
  }, []);
  function navigate(next) {
    if (snapshot) return;
    if (next.mode) setBrowseRoute(next);
    const filterQuery = new URLSearchParams(next.filters || {}).toString();
    const hash = next.subject
      ? new URLSearchParams({
          subject: next.subject,
          ...(next.transition ? { transition: next.transition } : {}),
        }).toString()
      : next.mode + (filterQuery ? "&" + filterQuery : "");
    history.pushState(null, "", "#" + hash);
    setRoute(next);
    window.scrollTo(0, 0);
  }
  const counts = catalog?.cohort?.counts || {};
  return (
    <>
      <aside className="sidebar">
        <a
          className="brand"
          href="#"
          id="home-brand"
          onClick={(e) => {
            e.preventDefault();
            navigate({ mode: "overview" });
          }}
        >
          <span className="brand-icon">
            m<span>+</span>
          </span>
          <span>
            MIMIC Atlas<small>DATA WORKSPACE</small>
          </span>
        </a>
        <div className="nav-label">浏览数据</div>
        <nav className="workspace-nav" aria-label="主导航">
          {[
            ["overview", "◫", "数据总览", null],
            [
              "patients",
              "▦",
              "全量患者",
              counts.patients,
            ],
            ["pairs", "⇄", "影像配对", counts.pairs],
            ["featured", "☆", "精选示例", catalog?.featured?.length],
            ["exports", "↓", "导出记录", null],
          ].map(([key, icon, title, n]) => (
            <button
              id={"nav-" + key}
              key={key}
              className={mode === key ? "active" : ""}
              aria-current={mode === key ? "page" : undefined}
              title={key === "patients" ? "本地 MIMIC-CXR 与 MIMIC-IV 按 subject_id 合并去重的患者并集" : undefined}
              disabled={Boolean(snapshot)}
              onClick={() => navigate({ mode: key })}
            >
              <span>{icon}</span> {title}
              <small
                id={
                  key === "patients"
                    ? "nav-patient-count"
                    : key === "pairs"
                      ? "nav-pair-count"
                      : undefined
                }
              >
                {n == null ? "" : fmt(n)}
              </small>
            </button>
          ))}
        </nav>
        <div className="sidebar-card">
          <span className="eyebrow">影像浏览</span>
          <label htmlFor="image-quality">影像清晰度</label>
          <select
            id="image-quality"
            value={quality}
            onChange={(e) => setQuality(e.target.value)}
            disabled={Boolean(snapshot)}
          >
            <option value="512">预览 · 512 px</option>
            <option value="960">标准 · 960 px</option>
            <option value="1400">清晰 · 1400 px</option>
          </select>
          <p>点击患者，自动加载相关临床信息。点击影像可放大查看。</p>
        </div>
        <div className="sidebar-foot">
          <div>
            <span className="live-dot" />
            <span id="connection-label">
              {snapshot ? "离线 HTML" : "本地数据连接"}
            </span>
          </div>
          <p>
            MIMIC-CXR-JPG 2.0.0
            <br />
            MIMIC-IV 3.1
          </p>
          <small>脱敏日期 · 仅比较患者内时间间隔</small>
        </div>
      </aside>
      <main>
        <header className="topbar">
          <div>
            <span className="breadcrumb">Medical world model</span>
            <span className="slash">/</span>Data explorer
          </div>
          <span className="local-badge" id="mode-badge">
            <i />
            {snapshot ? "OFFLINE SNAPSHOT" : "LOCAL WORKSPACE"}
          </span>
        </header>
        {error && (
          <div id="notice" className="notice" role="status">
            {error}
          </div>
        )}
        {snapshot && (
          <div className="notice">
            离线 HTML · 仅包含导出的当前片段及已嵌入的临床记录。
          </div>
        )}
        {catalog?.state === "ready" && route.subject && (
          <DatasetIntro catalog={catalog} compact offline={Boolean(snapshot)} />
        )}
        {catalog?.state !== "ready" ? (
          <div id="startup" className="startup" role="status">
            <span className="spinner" />
            <h2>连接本地 MIMIC 数据</h2>
            <p id="startup-stage">
              {catalog?.error || catalog?.stage || "正在读取数据目录…"}
            </p>
          </div>
        ) : route.subject ? (
          <Patient
            key={route.subject}
            subject={route.subject}
            initialTransition={route.transition}
            snapshot={snapshot}
            quality={quality}
            onBack={() => navigate(browseRoute)}
          />
        ) : route.mode === "exports" ? (
          <Exports
            onOpen={(subject, transition) => navigate({ subject, transition })}
          />
        ) : route.mode === "overview" ? (
          <Overview
            catalog={catalog}
            onBrowse={(mode, filters) => navigate({ mode, filters })}
          />
        ) : (
          <Cohort
            key={route.mode + JSON.stringify(route.filters || {})}
            catalog={catalog}
            mode={route.mode || "patients"}
            initialFilters={route.filters}
            onOpen={(subject, transition = "") =>
              navigate({ subject, transition })
            }
          />
        )}
      </main>
    </>
  );
}
