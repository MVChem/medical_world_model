# MIMIC Atlas

A local browser for MIMIC-CXR × MIMIC-IV, built with a **React + Vite frontend, FastAPI backend, and byte-offset indexes into the original CSV files**. The clean, light interface provides a full patient directory, complete study timelines, image comparisons, reports, original clinical records, trend charts, and offline HTML exports.

The reviewed MedWorld task dataset is built by [`data_processing`](data_processing/README.md) at
`code/data/medworld_0922`. It reuses the full-cohort CXR/IV matching rules, supports
random nonadjacent timepoint pairs, and links to the original images, 200 chest
X-rays with human-reviewed heart/lung segmentation, and 1,190 annotated brain MRI
volumes. The new dataset does not use CXAS pseudo labels. Run
`PYTHONPATH=code python -m mimic_atlas.data_processing` from the repository root.

For the newer medication-positive CXR pair selection, see
[`data_preprocessing`](../data_preprocessing/README.md). It requires two
chronological images from the same patient, a nonempty report for each endpoint,
and an actual administration between them or an overlapping active infusion.
It has no common-admission requirement or time-gap limit. This is separate from
the prepared-v2 builder above and the Atlas pair-view rules documented below.
Its current data directory is `code/data/medworld_0923`. MedWorld can use these
pairs for time-conditioned latent prediction without loading medication records;
the evidence remains available here for visual inspection.
Open [Medication pairs](http://127.0.0.1:8767/#medications) to inspect this selection
in Atlas, including both original images, their reports, and interval medication
records. See [Medication-pair review](#medication-pair-review) for its scope.

## Setup and development

Install dependencies and build the frontend:

```bash
cd /home/data2/chk/workspace/2026/08/04/medical_world_model/code/mimic_atlas/frontend
npm ci --include=dev
npm run build
```

Start the backend from `code/`; FastAPI also serves the built frontend:

```bash
cd /home/data2/chk/workspace/2026/08/04/medical_world_model/code
# Install only for a new Python environment; this machine uses the shared .venv.
.venv/bin/python -m pip install -r mimic_atlas/requirements.txt
.venv/bin/python mimic_atlas/setup_data.py --source /home/data2/chk/data/MIMIC
.venv/bin/python -m mimic_atlas --host 127.0.0.1 --port 8767
```

App: <http://127.0.0.1:8767>; API documentation: <http://127.0.0.1:8767/api/docs>. For frontend development, run `npm run dev` in `frontend/` in another terminal and open the URL printed by Vite. Requests to `/api` are automatically proxied to backend port 8767. The production build and offline HTML do not need a Node server or CDN access.

For remote access, use `ssh -L 8767:127.0.0.1:8767 your-server`. The local production instance is managed by the user service `mimic-atlas.service`; use `systemctl --user status mimic-atlas.service` to inspect it and `systemctl --user restart mimic-atlas.service` to restart it. Python logs go to `runs/atlas_YYYYMMDD/server.log`.

By default, the CLI selects the completed, version-compatible `runs/patient_index_*/manifest.json` with the latest directory name. Use `--index-root` to select one explicitly. When an index is configured, changed source files or incomplete indexes produce explicit errors instead of silently falling back to a scan. The older source-table reader without an index remains available for small-data tests; prepare the index below for full-cohort browsing.

## Global index: offsets without copies of clinical records

The local MIMIC-IV installation contains 31 decompressed CSV files totaling 90.52 GiB. The largest, `icu/chartevents.csv`, is 41.94 GB (39.06 GiB). At the user's request, the corresponding `.gz` files were deleted after complete decompression verification. Uncompressed CSV files allow direct reads of patient records by byte offset.

```bash
# Run from code/; building requires g++, but no GPU.
.venv/bin/python -m mimic_atlas.prepare_index \
  --output mimic_atlas/runs/patient_index_20260918_offsets --workers 2
.venv/bin/python -m mimic_atlas.verify_index \
  --index-root mimic_atlas/runs/patient_index_20260918_offsets
.venv/bin/python -m mimic_atlas \
  --index-root mimic_atlas/runs/patient_index_20260918_offsets --port 8767
```

The build scans each original CSV once. A streaming C++ scanner recognizes complete CSV records, including quotes, commas, quoted newlines, CRLF, and a final line without a newline. It emits only patient IDs, byte ranges, and row counts. Patients whose records occur in several places retain multiple ranges. Python builds the patient lookup indexes in SQLite without rewriting source files.

The index directory contains:

| File | Contents |
|---|---|
| `tables/<table_name>/subjects.sqlite` | `subject_id → start_byte, end_byte, row_count`; no event fields |
| `tables/<table_name>/metadata.json` | Field names, source-file fingerprint, whole-table counts, and sampled verification values |
| `directory.sqlite` | Tables containing each patient and their row counts |
| `cxr.sqlite` | Patient / study / image IDs, relative image and report paths, and file availability |
| `manifest.json` | Source roots, version, table directory, grouping rules, and build results |
| `progress.json`, `build.log` | Build progress and runtime logs |

**The index stores no CSV/Parquet copies of clinical records, report text, or image pixels.** When a patient is opened, the service uses `seek()` to read the relevant byte ranges from the original CSVs, parses every field, and verifies patient ownership row by row. Reports and images are read from their original paths; small CXR metadata tables and dictionaries are loaded into memory at startup. Parsed data for opened patients are reused in a bounded memory cache and reread after eviction, idle expiration, or restart.

The current full-cohort index is approximately **354 MiB (0.35 GiB)**, compared with 11.47 GiB for the previous Parquet copy cache, a reduction of about 97%. Reading the original records from 24 clinical tables for each of three patients, across three interleaved rounds, produced the following sums of per-table median times:

| Patient | CSV offset index | Previous Parquet cache |
|---|---:|---:|
| 10004606 | 206 ms | 1,019 ms |
| 12137189 | 224 ms | 1,038 ms |
| 10000032 | 68 ms | 334 ms |

This measures Python reading and parsing with a **warm operating-system file cache**, not first-access cold-disk latency or full-page load time. Each round compared original fields, ordering, and duplicate records for every table. Detailed results are retained in the [index run directory](runs/patient_index_20260918_offsets/README.md).

Builds can resume from completed tables; `--workers 1` reduces concurrent disk reads. The resolved source CSV path, byte size, and nanosecond modification time must match the index. Rebuild into a new directory when source tables change. This fingerprint is not a cryptographic checksum of the entire file. Report text is not indexed and is read from the current source file; newly added studies or images require rebuilding the directory and restarting the service.

## Memory management

Patient data share an LRU cache with idle expiration: defaults are at most **4 patients**, an estimated **512 MiB** budget, and automatic release **300 seconds** after the last access. A cleanup thread runs every 10 seconds, including when there are no new requests. Base clinical tables, reports and pair details, raw extended-table rows, normalized events, field/status metadata, and task references are removed together. Revisiting a patient rereads all records through the offset index.

Each load has its own validity token. Eviction cancels queued tasks; previously running tasks cannot write back after eviction, even if they finish later. Reopening the same patient cannot receive results from an earlier load. Data in use by HTTP responses or exports are temporarily protected. When the patient limit is reached and every slot is in use, the service returns a retryable 503. Dictionaries are shared across patients to avoid retaining duplicate ICD dictionaries.

The image cache stores only encoded preview bytes, up to **64 MiB / 512 images**, with the same 300-second idle expiration. Cache keys include the source image fingerprint, dimensions, quality, and format, so a changed source image cannot reuse an old preview. Original or decoded images are not retained, and at most two images are decoded concurrently. All caches reside in memory; clinical data and thumbnail copies are not cached on disk.

The patient budget estimates Python object sizes; **it is not a limit on total service RSS**. The full-cohort directory and shared dictionaries add fixed overhead, while reads and exports create temporary objects. A very large patient or an active response may temporarily exceed the budget to preserve complete records; patient data are never truncated. Released space can be reused by Python, but operating-system RSS may not immediately decrease by the same amount.

```bash
.venv/bin/python -m mimic_atlas --patient-cache-count 4 \
  --patient-cache-mib 512 --image-cache-mib 64 --cache-idle-seconds 300
```

`GET /api/memory` reports cached patient counts, estimated bytes, actual image-cache bytes, hits, and evictions. When the service cache expires, the browser automatically waits for the affected table to reload.

## Exports and project rename

The project directory and Python package are now consistently named `code/mimic_atlas`; start the service with `python -m mimic_atlas`. Downstream training, case-review scripts, and documentation references were updated accordingly.

The former root-level `example_output`, `linked_output`, `representative_output_10`, and `representative_linked_output_10` directories were moved in full to:

```text
runs/exports/legacy_20260918/
├── example_output/
├── linked_output/
├── representative_output_10/
└── representative_linked_output_10/
```

File contents and symlinks were verified individually during migration. Original JSONL/CSV files, input/target contracts, and archived HTML were preserved. Paths in historical summaries remain unchanged as provenance from the time of generation. The React Export Records page lists batches and files; episode links open the current image/clinical workspace for the same patient, and original files remain downloadable.

The CLI now defaults to `runs/exports/generated_YYYYMMDD/cxr` and `linked`. Export batches containing `summary.json` one or two levels below `runs/exports/` appear automatically on that page. Export manifests are streamed with pagination rather than retaining complete training datasets in memory. Explicitly generated export files are managed separately from automatically evicted browsing caches.

## Patient grouping and pairing rules

The rules are recorded in `prepare_index.py` and the index manifest's `cxr.rules`:

- Group patients only by a complete, exact `subject_id`; never combine patients based on nearby timestamps or similar images.
- Identify studies by `(subject_id, study_id)` and images by `dicom_id`; conflicting ownership blocks index publication. Retain every study and projection, including those without a split, labels, report, or local image.
- Locate reports as `s{study_id}.txt` under the patient's directory. Missing and empty reports are displayed separately.
- Preserve original IV fields. Records without `hadm_id` still belong to the patient, but admission membership is not guessed. eMAR/POE detail rows inherit time and admission context only from a unique parent record belonging to the same patient.
- **The index finds all of A, B, and C; it does not choose AB, BC, ABC, or AC.** The existing Image Pairs page is a separate view of adjacent studies and does not restrict the full dataset.

The existing pair view selects strictly adjacent studies along the complete timeline, with matching AP/PA projections and intervals from 1 hour to 365 days. It does not skip intervening studies. An admission is linked only when both actual acquisition timestamps fall within one unique admission interval, `[min(edregtime, admittime), dischtime]`; unmatched and ambiguous cases are retained. `study_id` is not `hadm_id`.

## Data setup and full-cohort scope

`setup_data.py` validates sources and creates reusable symlinks, stopping on path conflicts:

```text
code/data/
├── MIMIC        -> /home/data1/data/MIMIC
├── MIMIC_CXR    -> /home/data1/data/MIMIC/MIMIC_CXR
└── mimic-iv-3.1 -> /home/data1/data/MIMIC/mimic-iv-3.1
```

On this machine, `/home/data2/chk/data/MIMIC` points to `/home/data1/data/MIMIC`. The ordinary reader supports both `.csv` and `.csv.gz`; the new offset index requires `.csv`. Override source paths and the curated entrypoint with `--cxr-root`, `--iv-root`, and `--pairs`.

| Scope | Count |
|---|---:|
| CXR ∪ IV patients | 368,138 |
| CXR / IV patients | 65,379 / 364,627 |
| Patients with both CXR and IV | 61,868 |
| CXR-only / IV-only patients | 3,511 / 302,759 |
| CXR studies / chest X-rays | 227,835 / 377,110 |
| Adjacent image candidate pairs / pairs with one shared admission | 110,729 / 70,281 |
| train / validate / test pairs with a shared admission | 68,529 / 593 / 1,159 |

The overview summarizes the full cohort. Patient and pair directories provide search, filters, pagination, and complete directory CSV exports; no patient is selected automatically. These counts describe available full-cohort candidates before task-specific sampling and **are not the final training manifest**. Curated examples separately retain 7 patients and 10 annotated episodes.

Opening a patient automatically loads **24 patient-level clinical tables**: 7 admission/ICU context tables and 17 extended tables, without requiring individual load actions. Source-file sizes in the table list refer to the server's full-cohort CSVs, not the patient's records. IV-only patients can also access all relevant records.

| Group | Tables |
|---|---|
| Admission and ICU context | `admissions`, `transfers`, `diagnoses_icd`, `procedures_icd`, `icustays`, `procedureevents`, `inputevents` |
| Laboratory results and vital signs | `labevents`, `chartevents`, `outputevents`, `ingredientevents` |
| Microbiology and routine measurements | `microbiologyevents`, `omr` |
| Prescriptions, administration, and pharmacy | `prescriptions`, `emar`, `emar_detail`, `pharmacy` |
| Orders and other records | `poe`, `poe_detail`, `datetimeevents`, `services`, `hcpcsevents`, `drgcodes`, `patients` |

Original dictionaries supply item and code descriptions. `provider` and `caregiver` are staff directories, not patient record tables.

## Browsing and exporting

- The standalone Data Overview page (`#overview`) presents the introduction, full-cohort size, coverage, splits, and pair audit. All Patients (`#patients`), Image Pairs (`#pairs`), and Featured Examples (`#featured`) show their respective directories and filters. Clicking a coverage group or split in the overview opens the corresponding filtered directory; refreshing preserves that entrypoint's filters.
- The dataset introduction and counting-methods panel explains CXR/IV contents, patient-deduplicated unions and intersections, patient/admission/study/image identifiers, and the distinction between browsable candidates and actual training data, with links to official documentation. It is expanded by default on the standalone overview and collapsed on patient pages. Statistics come from the current full-cohort directory; newly exported offline HTML retains export-time statistics and explicitly states the scope of its embedded content.
- The patient timeline includes all studies. Open a study to inspect all projections, or select a pair to compare both images, reports, and four-state CheXpert labels.
- Images default to 512 px, with 960 / 1400 px options; opening the enlarged view loads 1800 px. Synchronized zoom, brightness, contrast, inversion, and panning do not modify source images.
- Clinical context displays admissions, ICU stays, ward transfers, procedure/input events, and ICD codes, with pagination, search, complete fields, and CSV exports for the 7 original tables.
- Extended tables support all patient records, a selected admission, or windows extending 0 / 6 / 24 / 72 hours before and after imaging. Items are grouped by `itemid + original unit`; scatter plots show observed values without interpolation or unit conversion. Date-only records, text, threshold values, and records without timestamps remain in the detail tables.
- Raw-record CSV exports contain every filtered result, independent of pagination and plot sampling. Plots explicitly indicate sampling when more than 1,500 points are available.
- JSON exports contain patient browsing data and load status. Offline HTML embeds the selected pair, available images from both studies (up to 1400 px), reports, labels, clinical context, and complete records within the window extending 24 hours before and after the pair from tables loaded at export time. It can be opened without a network connection. Tables not embedded are explicitly marked. Single-study views do not offer episode HTML exports.

The interface uses cancellable API requests. After switching patients, stale responses cannot overwrite the new patient's state. When clinical context arrives later, admission options update while preserving the selected table and scope. Non-image data no longer have overly restrictive transfer limits; directories and raw records remain paginated for browsing.

Follow-up images, reports, and retrospective IV records are not inputs for current-state prediction. ICD diagnoses are discharge codes, and procedure codes have only date-level precision. Concurrent treatment cannot be interpreted as the cause of an image change. The original `forecast_inputs.jsonl` input boundary is unchanged.

## Project structure

```text
mimic_atlas/
├── frontend/
│   ├── src/App.jsx                 # App, navigation, and directory loading
│   ├── src/components/             # Patient directories, viewer, reports, clinical records
│   ├── src/api.js                  # API calls, cancellation, and exports
│   ├── src/charts.js               # Local SVG timelines and trend charts
│   ├── src/styles/                 # Light interface styles
│   ├── package.json, package-lock.json
│   └── vite.config.js              # Development proxy and production build
├── backend/
│   ├── app.py                      # FastAPI lifecycle, CLI, and page serving
│   ├── api.py                      # Patient, table, image, and export APIs
│   ├── schemas.py                  # Request and query-parameter validation
│   ├── images.py                   # Original image reading and resizing
│   └── frontend.py                 # React assets and offline HTML embedding
├── app.py, __main__.py              # FastAPI factory and module entrypoint
├── data.py, cohort.py               # Patient data, cohort statistics, and linkage
├── memory.py                       # Shared patient LRU/TTL and bounded image cache
├── exports.py                      # Explicit export batches, paginated episodes, downloads
├── clinical_tables.py              # Extended-table parsing, filtering, and numeric series
├── patient_index.py                # SQLite lookup and original CSV reads by offset
├── csv_spans.cpp, prepare_index.py  # Build offset indexes in one scan
├── verify_index.py                 # Independent consistency and API validation
├── setup_data.py                   # Data symlinks
├── build_mimic_transitions.py       # Original training pairs and explicit exports
├── link_mimic_iv_context.py         # IV reading and clinical linkage
├── tests/, browser_check*.py        # Synthetic-data and real-browser checks
└── runs/                           # Git-ignored indexes, logs, and explicit exports
```

Training-data preparation and static exports remain available. Commands and imports consistently use `mimic_atlas`; see [Pairing and Training](docs/transition_pipeline.md). `runs/`, source data, symlinks, `node_modules/`, and `frontend/dist/` are excluded from Git.

## Validation

Build the frontend first, then run from `code/`:

```bash
.venv/bin/python -m pytest mimic_atlas/tests -q
.venv/bin/python -m mimic_atlas.browser_check --include-inputs
.venv/bin/python -m mimic_atlas.browser_check_cohort
.venv/bin/python -m mimic_atlas.browser_check_clinical
.venv/bin/python -m mimic_atlas.browser_check_react
.venv/bin/python -m mimic_atlas.browser_check_memory
.venv/bin/python -m mimic_atlas.verify_index \
  --index-root mimic_atlas/runs/patient_index_20260918_offsets \
  --url http://127.0.0.1:8767
```

Test dependencies are listed in `requirements-dev.txt`. Browser checks use system Chrome/Chromium or Playwright Chromium and write screenshots and results to `runs/`. Coverage includes out-of-order and fragmented patient records, multiline CSV, duplicate and missing fields, patient isolation, source-file changes, automatic loading, paginated exports, delayed responses, late clinical context, offline HTML, and a 390px mobile layout. Memory checks additionally cover patient/byte budgets, idle cleanup without requests, consistency of original records after eviction, prevention of stale task writes, automatic recovery of open pages, image reuse, and navigation from older exports.

## Design and technical references

The interface follows the light Figma design references: [shadcn/ui design system](https://www.figma.com/community/file/1203061493325953101), [official Figma directory](https://ui.shadcn.com/docs/figma), and [light Dashboard](https://ui.shadcn.com/examples/dashboard). The Figma community page previously returned 403 on this machine, so visual checks used the same system's official previews; no nodes were retrieved through the Figma API. The interface is implemented in React and does not depend on paid design assets.

Architecture references: [Build a React App from Scratch](https://react.dev/learn/build-a-react-app-from-scratch), [Vite](https://vite.dev/guide/), and [FastAPI: Bigger Applications](https://fastapi.tiangolo.com/tutorial/bigger-applications/). Data and fields: [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/), [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/), [Laboratory Events](https://mimic.mit.edu/docs/iv/modules/hosp/labevents.html), and [ICU Chart Events](https://mimic.mit.edu/docs/iv/modules/icu/chartevents.html).

The overview's study-interval histogram follows the simple cards, direct value labels, and low-contrast grid of [shadcn/ui Bar Charts](https://ui.shadcn.com/charts/bar). It uses native CSS and aggregates six interval bins from the full candidate-pair cohort at runtime. Bins include the lower bound and exclude the upper bound, except that the final bin includes its upper bound. Counts are per pair, without patient deduplication.

The bottom of the overview contains three compact bar charts: intervals for all pairs, six interval bins for pairs with 1 ≤ interval < 24 hours, and the number of distinct studies per CXR patient (multiple images from one study count once). The first two count pairs; the third counts patients. Each card displays its percentage denominator in the upper-right corner.

## Medication-pair review

Open <http://127.0.0.1:8767/#medications> or select **Medication pairs** in the
sidebar. This page directly reads the complete selection at
`code/data/medworld_0923`; it does not recompute pairs using the
older Image Pairs page's adjacent-study or admission-linkage rules.

| 2026-09-23 medication selection | Count |
| --- | ---: |
| Selected chronological pairs | 1,022,127 |
| Distinct patients | 21,097 |
| Train / validate / test pairs | 985,153 / 10,630 / 26,344 |

Every selected pair has the same patient at both endpoints, two available
frontal images, a nonempty report belonging to each study, and at least one
accepted administration in the exact acquisition interval or an active infusion
overlapping it. All forward combinations are eligible: different admissions,
nonadjacent examinations, and AP-to-PA changes are allowed. There is no minimum
or maximum time gap. The detailed predicates and completed verification are in
the [preprocessing run](../data_preprocessing/runs/medication_filter_20260923/README.md).

Use the split filter, exact patient ID, or pair/study search to find cases. Open
a pair to see both images, enlarge either image, read both complete original
reports, and inspect the medication table. The selected pair and directory
filters are encoded in the URL, so direct links and browser Back restore the
same case. A separate link opens the full patient workspace for broader clinical
context.

Medication records are reread from the original eMAR, eMAR details, and ICU
inputevents through the patient byte-offset index, then checked again with the
selection's administration predicates. The window is the exact inclusive CXR
interval, with no added hours and no admission filter. Recomputed source counts
must match the saved pair before records are returned. Search and source filters
operate on the full interval result; pagination never discards records.

Each record exposes its name, source, recorded status, original start/end times,
timing relative to the first image, amount/unit, rate/unit, route, and dose basis.
Expand it to inspect the original row, eMAR product details, identifiers, and
patient-relative source positions. Missing values remain missing. Counts refer
to source records rather than deduplicated doses; eMAR and ICU records can
describe overlapping treatment. ICU amounts describe complete original
segments and must not be interpreted as amounts delivered only between the
images when a segment crosses a boundary.

On first access, Atlas validates the completed manifest, source fingerprints,
predicate implementation, and pair-shard checksums, then builds a compact Arrow
directory in memory. Loading progress and errors are explicit. The directory
remains until service exit; opened medication records have a separate bounded
memory cache (at most 2 patients / 128 MiB, 300 seconds idle). No browsing index,
raw clinical payload, report, or image cache is written to disk. This page is not
embedded in the older case-level offline HTML exports.

To open another complete selection with the same schema and current predicates:

```bash
# From code/. Use the CXR source/index belonging to the same installation.
.venv/bin/python -m mimic_atlas --port 8767 \
  --medication-cohort data/medworld_0923
```

APIs: `/api/medication-cohort`, `/api/medication-cohort/pairs`,
`/api/medication-cohort/pairs/{id}`, and
`/api/medication-cohort/pairs/{id}/medications`. Images use the existing image API.
The feature follows the project's [shadcn/ui Dashboard reference](https://ui.shadcn.com/examples/dashboard)
with count cards, filters, paired image/report panels, and paginated records.

Validation, from `code/`:

```bash
.venv/bin/python -m pytest mimic_atlas/tests/test_medication_cohort.py -q
.venv/bin/python -m mimic_atlas.browser_check_medications
```

The browser check covers real source/report identity, all 369 records of a
nonadjacent AP-to-PA pair, active infusions at the first image, filtering and
pagination, raw-field expansion, enlarged images, patient navigation, direct
links, and a 390 px viewport. A separate backend check read all 7,693 records of
the most densely recorded pair over 77 pages. Detailed evidence stays in
`runs/medication_render_20260923/` and `runs/medication_browser_20260923/`.

## MIMIC-CXR-VQA browser

Open <http://127.0.0.1:8767/#vqa> or select VQA Questions in the sidebar. The browser reads
`code/data/MIMIC_CXR_VQA/MIMIC-Ext-MIMIC-CXR-VQA/dataset/{train,valid,test}.json`
and retrieves chest X-rays through the existing CXR image API by exact `image_id`.

- Displays question counts, distinct image counts, patient counts, empty-answer rates, and clickable question-type distributions for the three official splits.
- Supports filters for split, verify / choose / query, 7 content types, and empty / nonempty answers, plus search by question, answer, idx, or patient / study / image ID. Each page contains 25 questions.
- Open Chest X-ray and Questions to view the original question, reference answers, an enlargeable image, and all questions for that image across official splits, with separate pagination. Links open the patient's timeline and clinical records. Original English text and answer arrays are preserved; an empty set is displayed as `[]`, distinct from missing data. Region names in questions are not pixel-level bounding-box annotations.
- Full source totals: 377,391 questions, 142,797 distinct chest X-rays, and 55,716 distinct patients. The train / valid / test splits contain 290,031 / 73,567 / 13,793 questions. Train and valid share 698 patients; test has no patients in common with either. The official VQA split and Atlas CXR split are separate fields.

On first access, `ijson` streams the source files while a compact question directory is built in the background and kept only in memory. It does not copy original JSON, clinical records, or images to disk. The directory remains until the service exits and is outside the patient LRU budget; restart the service after source updates. Missing source files cause explicit errors, and partial splits are not published. Images reuse the existing bounded cache and resolution controls. The question directory is not loaded until the VQA page is opened. This page is not included in case-level offline HTML.

APIs: `/api/vqa/summary`, `/api/vqa/questions`, and `/api/vqa/questions/{split}/{position}`.
`position` is zero-based within a split. Responses preserve the source `idx` without assuming that idx values are globally unique across splits.

Validation, from `code/`:

```bash
.venv/bin/python -m pytest mimic_atlas/tests -q
.venv/bin/python -m mimic_atlas.browser_check_vqa
```

Browser checks cover full-cohort counts, patient intersections, empty-answer filtering, pagination, image matching, same-image questions, enlargement, patient navigation, searches with no results, and a 390px layout. Screenshots and results are retained in `runs/vqa_browser_20260918/`. Styling follows the project's [shadcn/ui Dashboard reference](https://ui.shadcn.com/examples/dashboard), using light statistic cards, bar charts, and filterable tables.
