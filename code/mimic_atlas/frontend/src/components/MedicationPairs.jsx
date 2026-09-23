import { useEffect, useRef, useState } from "react";
import { useResource } from "../api";
import { fmt } from "../format";
import "../styles/medications.css";

const PAGE_SIZE = 25;
const SPLITS = ["train", "validate", "test", "unassigned"];
const defaults = { q: "", subject_id: "", split: "all" };
const stamp = value => value ? String(value).replace("T", " ") : "Not recorded";
const text = value => value == null || value === "" ? "—" : String(value);
const json = value => JSON.stringify(value ?? null, null, 2);
const relative = value => value == null || !Number.isFinite(Number(value))
  ? "Not recorded"
  : `${Number(value) >= 0 ? "+" : ""}${Number(value).toFixed(2)} h`;
const elapsed = value => value == null ? "—"
  : `${Number(value).toLocaleString("en-US", { maximumFractionDigits: 2 })} h · ${(Number(value) / 24).toLocaleString("en-US", { maximumFractionDigits: 2 })} d`;

function Pagination({ page, total, limit = PAGE_SIZE, loading, onChange, name }) {
  const pages = Math.max(1, Math.ceil((total || 0) / limit));
  return <div className="observation-pagination medication-pagination">
    <span>{loading ? "Loading records…" : `${fmt(total)} results`}</span>
    <div>
      <button disabled={page <= 1 || loading} onClick={() => onChange(page - 1)} aria-label={`${name} previous page`}>← Previous</button>
      <span>Page {fmt(page)} / {fmt(pages)}</span>
      <button disabled={loading || page >= pages} onClick={() => onChange(page + 1)} aria-label={`${name} next page`}>Next →</button>
    </div>
  </div>;
}

function SourceCounts({ counts }) {
  if (!counts) return <span className="muted">See source records</span>;
  return <span className="medication-source-counts">
    {[["records", "records"], ["emar", "eMAR"], ["inputevents", "ICU"]].map(([key, label]) => typeof counts[key] === "number"
      ? <span key={key}><b>{fmt(counts[key])}</b> {label}</span>
      : null)}
  </span>;
}

function imageUrl(endpoint, size) {
  const url = new URL(endpoint.image_url || `/api/images/${encodeURIComponent(endpoint.dicom_id)}`, window.location.origin);
  url.searchParams.set("size", size);
  url.searchParams.set("quality", size === "512" ? "65" : "88");
  return url.pathname + url.search;
}

function Endpoint({ endpoint, side, quality, imagesReady, onExpand }) {
  const [failed, setFailed] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const src = imageUrl(endpoint, quality);
  useEffect(() => { setFailed(false); setLoaded(false); }, [src]);
  return <article className={`medication-endpoint ${side}`} data-endpoint={side}>
    <div className="medication-endpoint-heading">
      <span className="eyebrow">{side === "source" ? "T₀ · SOURCE" : "T₁ · FOLLOW-UP"}</span>
      <time>{stamp(endpoint.time)}</time>
    </div>
    <button className="medication-image" onClick={() => onExpand(endpoint)} disabled={!imagesReady || failed} aria-label={`Enlarge ${side} chest X-ray`}>
      {!imagesReady ? <span>Loading the image directory…</span>
        : failed ? <span>Image could not be read.</span>
          : <><img src={src} alt={`${side === "source" ? "Source" : "Follow-up"} chest X-ray, study ${endpoint.study_id}, ${endpoint.view}`} onLoad={() => setLoaded(true)} onError={() => setFailed(true)} />{!loaded && <span className="medication-image-loading">Loading image…</span>}<span className="medication-expand">⛶ Enlarge</span></>}
    </button>
    <div className="medication-image-caption"><span>Study {endpoint.study_id}</span><span>{endpoint.view} · {quality} px preview</span></div>
    <details className="medication-image-id"><summary>Image identifier</summary><code>{endpoint.dicom_id}</code></details>
    <div className="medication-report-heading"><h3>Original radiology report</h3><span>{endpoint.report_status || "Source text"}</span></div>
    <pre className="medication-report" data-report={side}>{endpoint.report || "No report text is available."}</pre>
  </article>;
}

function RawDetails({ row }) {
  return <details className="medication-raw">
    <summary>Original record & source</summary>
    <dl><dt>Source table</dt><dd>{text(row.source_reference?.table || row.source_table)}</dd><dt>Patient row ordinal (zero-based)</dt><dd>{text(row.source_reference?.patient_row_ordinal)}</dd></dl>
    <p className="muted">Source records and product details are shown in full, with original field names and units.</p>
    <h4>Original record</h4><pre>{json(row.raw_record)}</pre>
    {row.raw_details != null && <><h4>Original detail rows</h4><pre>{json(row.raw_details)}</pre></>}
    {row.detail_references?.length > 0 && <><h4>Detail source references (zero-based ordinals)</h4><pre>{json(row.detail_references)}</pre></>}
    {row.dose_details != null && <><h4>Dose details</h4><pre>{json(row.dose_details)}</pre></>}
  </details>;
}

function MedicationRecords({ pair }) {
  const [source, setSource] = useState("all");
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const params = new URLSearchParams({ source, q: search, page, limit: PAGE_SIZE });
  const { data, error } = useResource(`/api/medication-cohort/pairs/${encodeURIComponent(pair.id)}/medications?${params}`);
  return <section className="medication-records" aria-label="Interval medication records">
    <div className="medication-section-heading"><div><span className="eyebrow">RECORDED CARE BETWEEN IMAGES</span><h3>Medication administrations & active infusions</h3></div><span className="subtle-badge">{data ? fmt(data.unfiltered_total) : "…"} source records</span></div>
    <p className="medication-explanation">Times below are original event times. Relative times are measured from T₀; a negative start means the infusion began before the first image and remained active during the interval.</p>
    <p className="medication-explanation">Amounts and rates belong to the original records. An ICU amount may cover a segment extending outside T₀–T₁; it is not an interval dose total. eMAR and ICU records may document the same administration and are not deduplicated across sources.</p>
    {data?.counts_match === false && <div className="notice" role="alert">The re-read source count does not match the saved selection evidence. Expected: {typeof data.expected_records === "object" ? json(data.expected_records) : fmt(data.expected_records)}. Review the original records below.</div>}
    <form className="medication-filters medication-event-filters" onSubmit={event => { event.preventDefault(); setSearch(query.trim()); setPage(1); }}>
      <label>Record source<select id="medication-source" value={source} onChange={event => { setSource(event.target.value); setPage(1); }}><option value="all">All sources</option><option value="hosp.emar">eMAR administrations</option><option value="icu.inputevents">ICU medication inputevents</option></select></label>
      <label className="medication-search-field">Search medication records<input id="medication-record-search" value={query} onChange={event => setQuery(event.target.value)} placeholder="Medication name" /></label>
      <button type="submit">Search records</button>
      <button type="button" onClick={() => { setSource("all"); setQuery(""); setSearch(""); setPage(1); }}>Reset</button>
    </form>
    {error && <div className="notice" role="alert">{error}</div>}
    <div className="table-wrap medication-records-scroll" tabIndex={0} role="region" aria-label="Medication records table, scroll horizontally for all columns">
      <table id="medication-records-table"><thead><tr><th>Medication / record status</th><th>Original event time</th><th>Relative to T₀</th><th>Original amount / dose</th><th>Original rate</th><th>Route</th><th>Source evidence</th></tr></thead><tbody>
        {data?.rows.map((row, index) => <tr key={`${row.source_table}-${row.source_reference?.patient_row_ordinal ?? index}`}>
          <td><strong>{row.name || "Unnamed medication"}</strong><small>{row.source_table}</small><span className="medication-status">{text(row.record_status)}</span></td>
          <td><time>{stamp(row.start)}</time>{row.end && row.end !== row.start && <small>to {stamp(row.end)}</small>}{row.source_end && row.source_end !== row.end && <small>Original segment ends {stamp(row.source_end)}</small>}</td>
          <td><span>{relative(row.relative_start_hours)}</span>{row.relative_end_hours != null && row.relative_end_hours !== row.relative_start_hours && <small>to {relative(row.relative_end_hours)}</small>}{Number(row.relative_start_hours) < 0 && <small className="medication-prior">Started before T₀</small>}</td>
          <td><span>{text(row.amount)} {row.amount_unit || ""}</span>{row.dose_basis && <small>{row.dose_basis}</small>}{row.dose_details != null && <details className="medication-dose-details"><summary>Dose detail</summary><pre>{json(row.dose_details)}</pre></details>}</td>
          <td>{text(row.rate)} {row.rate_unit || ""}</td><td>{text(row.route)}</td><td><RawDetails row={row} /></td>
        </tr>)}
      </tbody></table>
      {!data && !error && <p className="empty-state" role="status">Reading original medication records…</p>}
      {data?.total === 0 && <p className="empty-state">No records match these filters.</p>}
    </div>
    <Pagination name="Medication records" page={page} total={data?.total} limit={data?.limit || PAGE_SIZE} loading={!data} onChange={setPage} />
  </section>;
}

function PairDetail({ pairId, quality, imagesReady, onClose, onOpenPatient }) {
  const { data: pair, error } = useResource(`/api/medication-cohort/pairs/${encodeURIComponent(pairId)}`);
  const dialog = useRef(null);
  const [large, setLarge] = useState(null);
  const [largeFailed, setLargeFailed] = useState(false);
  const [zoom, setZoom] = useState(1);
  function expand(endpoint) {
    setLarge(endpoint); setLargeFailed(false); setZoom(1);
    dialog.current.showModal();
  }
  return <section className="panel medication-detail" id="medication-pair-detail" aria-label="Medication pair detail">
    <div className="panel-heading medication-detail-heading"><div><span className="eyebrow">PAIRED IMAGES · REPORTS · RECORDED TREATMENT</span><h2>{pair ? `Patient ${pair.patient}` : "Loading pair…"}</h2><p className="muted medication-pair-id">{pairId}</p></div><div className="medication-actions">{pair && <button onClick={() => onOpenPatient(pair.patient)}>Full patient timeline ↗</button>}<button onClick={onClose}>Close pair ×</button></div></div>
    {error && <div className="notice" role="alert">{error}</div>}
    {!pair && !error && <p className="empty-state" role="status">Reading the image pair and original reports…</p>}
    {pair && <>
      <div className="medication-interval"><div><span className="eyebrow">T₀ · SOURCE ACQUISITION</span><strong>{stamp(pair.source_time)}</strong></div><div className="medication-interval-middle"><span aria-hidden="true">⟶</span><strong>{elapsed(pair.hours)}</strong><span>{pair.full_timeline_adjacent ? "Adjacent studies" : "Nonadjacent studies"} · {pair.split}</span></div><div><span className="eyebrow">T₁ · FOLLOW-UP ACQUISITION</span><strong>{stamp(pair.target_time)}</strong></div></div>
      <div className="medication-pair-meta"><SourceCounts counts={pair.medication?.counts || pair.medication_counts || pair.medication} /><span className="muted">Recorded events; not a count of unique doses.</span></div>
      <div className="medication-endpoints">{["source", "target"].map(side => pair.endpoints?.[side] && <Endpoint key={`${pair.id}-${side}`} endpoint={pair.endpoints[side]} side={side} quality={quality} imagesReady={imagesReady} onExpand={expand} />)}</div>
      <MedicationRecords key={pair.id} pair={pair} />
    </>}
    <dialog ref={dialog} id="medication-image-dialog" aria-label="Enlarged chest X-ray" onClick={event => { if (event.target === dialog.current) dialog.current.close(); }} onClose={() => setLarge(null)}>
      <div className="medication-dialog-toolbar"><span>{large?.view} · Study {large?.study_id}</span><div><label>Zoom <input aria-label="Enlarged image zoom" type="range" min="1" max="3" step="0.25" value={zoom} onChange={event => setZoom(Number(event.target.value))} /></label>{large && <a href={imageUrl(large, "1800")} target="_blank" rel="noreferrer">Open 1800 px image ↗</a>}<button autoFocus onClick={() => dialog.current.close()}>Close image ×</button></div></div>
      <div className="medication-dialog-image">{large && (largeFailed ? <p role="alert">The enlarged image could not be read.</p> : <img src={imageUrl(large, "1800")} alt={`Enlarged chest X-ray, study ${large.study_id}`} style={{ width: `${zoom * 100}%`, maxWidth: "none" }} onError={() => setLargeFailed(true)} />)}</div>
    </dialog>
  </section>;
}

export default function MedicationPairs({ route, quality, imagesReady, onNavigate }) {
  const [poll, setPoll] = useState(1000);
  const { data: summary, error: summaryError } = useResource("/api/medication-cohort", poll);
  const filters = { ...defaults, ...route.filters };
  const [draft, setDraft] = useState(filters);
  const filterKey = JSON.stringify(filters);
  useEffect(() => { setDraft(JSON.parse(filterKey)); }, [filterKey]);
  const page = route.page || 1;
  const ready = summary?.state === "ready";
  useEffect(() => { if (ready || summary?.state === "error") setPoll(0); }, [ready, summary?.state]);
  const params = new URLSearchParams({ ...filters, page, limit: PAGE_SIZE });
  const { data, error } = useResource(ready ? `/api/medication-cohort/pairs?${params}` : null);
  const browse = (next = {}) => onNavigate({ mode: "medications", filters, page, pair: route.pair || "", ...next });
  function apply(next = draft) { browse({ filters: { ...defaults, ...next, q: next.q?.trim() || "", subject_id: next.subject_id?.trim() || "" }, page: 1, pair: "" }); }
  const splits = summary?.splits || {};
  return <div id="medications-content">
    <section className="page-heading"><div><div className="eyebrow">MIMIC-CXR × MIMIC-IV · RECORDED TREATMENT</div><h1>Medication pairs</h1><p>Compare two chest X-rays, their original reports, and the medication records between them.</p></div><span className="subtle-badge">{ready ? `${fmt(summary.pairs)} selected pairs` : "Preparing directory"}</span></section>
    {(summaryError || summary?.error) && <div className="notice" role="alert">{summaryError || summary.error}</div>}
    {!ready && !summaryError && !summary?.error && <div className="panel empty-state" role="status"><span className="spinner" /><p>Preparing the medication pair directory…</p>{summary?.loaded_pairs != null && <p className="muted">{fmt(summary.loaded_pairs)} pairs indexed</p>}</div>}
    {ready && <>
      <div className="medication-summary"><article className="panel medication-stat"><span className="eyebrow">PAIRS WITH RECORDED MEDICATION</span><strong>{fmt(summary.pairs)}</strong><span>Two usable images with reports</span></article><article className="panel medication-stat"><span className="eyebrow">DISTINCT PATIENTS</span><strong>{fmt(summary.patients)}</strong><span>Both images belong to the same patient</span></article><article className="panel medication-stat medication-rule-card"><span className="eyebrow">SELECTION</span><h3>Actual administration or active infusion</h3><p>No common-admission requirement or time-gap limit.</p></article></div>
      <div className="medication-splits" aria-label="Patient split filters"><button aria-pressed={filters.split === "all"} onClick={() => apply({ ...draft, split: "all" })}>All splits <b>{fmt(summary.pairs)}</b></button>{SPLITS.filter(split => splits[split] != null).map(split => <button key={split} aria-pressed={filters.split === split} onClick={() => apply({ ...draft, split })}>{split} <b>{fmt(typeof splits[split] === "number" ? splits[split] : splits[split].retained || 0)}</b></button>)}</div>
      <details className="medication-rules"><summary>What qualifies a pair?</summary><p>Both endpoints have a usable frontal image and a nonempty report. At least one accepted eMAR administration occurs within the image interval, or an accepted ICU medication infusion overlaps it. A prescription alone does not qualify. All chronological combinations are eligible, including nonadjacent studies and different frontal views.</p><p>This page uses the medication selection. The existing Image Pairs page retains its separate matching rules.</p></details>
      {route.pair && <PairDetail key={route.pair} pairId={route.pair} quality={quality} imagesReady={imagesReady} onClose={() => browse({ pair: "" })} onOpenPatient={subject => onNavigate({ subject })} />}
      <section className="panel medication-browser" aria-label="Medication pair directory">
        <div className="medication-section-heading"><div><span className="eyebrow">BROWSE THE SELECTION</span><h2>Image pair directory</h2></div><span className="muted">{data ? `${fmt(data.total)} matching pairs` : "Loading…"}</span></div>
        <form className="medication-filters" onSubmit={event => { event.preventDefault(); apply(); }}>
          <label className="medication-search-field">Search pairs<input id="medication-search" value={draft.q} onChange={event => setDraft({ ...draft, q: event.target.value })} placeholder="Patient, study, or pair ID" /></label>
          <label>Exact patient ID<input id="medication-subject" inputMode="numeric" value={draft.subject_id} onChange={event => setDraft({ ...draft, subject_id: event.target.value })} placeholder="e.g. 10000032" /></label>
          <label>Patient split<select id="medication-split" value={draft.split} onChange={event => { const next = { ...draft, split: event.target.value }; setDraft(next); apply(next); }}><option value="all">All splits</option>{SPLITS.map(split => <option key={split} value={split}>{split}</option>)}</select></label>
          <button type="submit">Search pairs</button><button type="button" onClick={() => apply(defaults)}>Reset</button>
        </form>
        {error && <div className="notice" role="alert">{error}</div>}
        <div className="table-wrap medication-pairs-scroll" tabIndex={0} role="region" aria-label="Image pair directory table, scroll horizontally for all columns">
          <table id="medication-pairs-table"><thead><tr><th>Patient / split</th><th>Source image · T₀</th><th>Follow-up image · T₁</th><th>Interval / adjacency</th><th>Medication source records</th><th>Inspect pair</th></tr></thead><tbody>{data?.rows.map(row => <tr key={row.id} className={route.pair === row.id ? "selected" : ""}>
            <td><strong>{row.patient}</strong><small>{row.split}</small></td><td><time>{stamp(row.source_time)}</time><small>{row.source_study_id} · {row.source_view}</small></td><td><time>{stamp(row.target_time)}</time><small>{row.target_study_id} · {row.target_view}</small></td><td>{elapsed(row.hours)}<small>{row.full_timeline_adjacent ? "Adjacent" : "Nonadjacent"}</small></td><td><SourceCounts counts={row.medication?.counts || row.medication_counts || row.medication} /></td><td><button data-medication-pair={row.id} aria-label={`Inspect medication pair ${row.id}`} onClick={() => browse({ pair: row.id })}>Images, reports & records ↗</button></td>
          </tr>)}</tbody></table>
          {!data && !error && <p className="empty-state" role="status">Loading image pairs…</p>}{data?.total === 0 && <p className="empty-state">No pairs match these filters. Try another patient, identifier, or split.</p>}
        </div>
        <Pagination name="Medication pairs" page={page} total={data?.total} limit={data?.limit || PAGE_SIZE} loading={!data} onChange={next => browse({ page: next, pair: "" })} />
      </section>
    </>}
  </div>;
}
