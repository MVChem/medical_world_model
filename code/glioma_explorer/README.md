# Glioma Atlas

A local HTML viewer for UCSF-ALPTDG and MU-Glioma-Post. The source lives in the
project's existing `code/` directory.

- **Full application:** [http://127.0.0.1:8766](http://127.0.0.1:8766). The service runs on the machine that holds the data; forward port 8766 for remote access.
- **Full HTML:** [code/data/glioma_explorer/runs/glioma_atlas.html](../data/glioma_explorer/runs/glioma_atlas.html). The entry file is about 1.17 MB and covers all 501 patients. It loads the adjacent `full_data/` on demand, requires no Python server, and can also run directly in the chat file preview. Keep the entire directory together when downloading it to another machine.
- **Previews:** [UCSF desktop view](../data/glioma_explorer/runs/desktop.png), [MU view](../data/glioma_explorer/runs/full-mu.png), and [mobile layout](../data/glioma_explorer/runs/full-mobile.png).

## Available Data

Both datasets have been downloaded and verified in full. The full HTML export
has read all **7,746 NIfTI files**: 501 patients, 1,192 MRI time points, 4,768
standard sequence volumes, 1,190 longitudinal segmentations, and 1,788 additional
UCSF difference/label files.

| Dataset | Imaging | Tables and Charts |
| --- | --- | --- |
| UCSF-ALPTDG | T1, T1 CE, T2, and FLAIR at both time points for all 298 patients; axial/coronal/sagittal views; synchronized slices; segmentation toggle and opacity | 596 visits; age, diagnosis, sex, grade, and follow-up interval; four-region volumes calculated from original masks; patient search and CSV export |
| MU-Glioma-Post | Four-sequence MRI for 203 patients and 596 time points; 594 segmentations; comparisons using original time-point identifiers, retaining single visits and missing annotations | Clinical-to-imaging identifier matching, follow-up timelines, scanner distributions, segmentation volumes, search, and CSV export |

The full HTML supports every patient, every local time point, all four MRI
sequences, all three anatomical planes, and every slice. Segmentation visibility
and opacity are adjustable. Nonconsecutive MU time points, single visits, and
missing masks are retained. This replaces the earlier HTML that contained only
two cases and 336 sampled slices.

The complete clinical-field view displays the selected patient's original
records. The complete imaging-file view exposes every NIfTI for that patient,
including UCSF T1 CE minus T1, longitudinal differences, and difference
segmentations. The raw-table view provides all **12 worksheets** from the four
original workbooks, retaining header rows, field codes, and every row and column,
with search, pagination, and CSV export.

The HTML entry file does not embed the entire imaging collection. The adjacent
`full_data/` stores full-volume display assets by patient/file, totaling about
**10.95 GB**, and is part of the complete offline HTML deliverable. Opening the
page loads only the few volumes currently needed; the browser caches at most
eight volumes to avoid loading the whole dataset at once. Physical files remain
under `/home/data2/chk/data/glioma_explorer/runs/`, exposed through the shared
symlink entry point `code/data/glioma_explorer/runs/`.

## Start the Application

Run from the repository root. The verified environment uses Python 3.12:

```bash
PYTHONPATH=code /home/data2/chk/workspace/2026/.venv/bin/python \
  -m glioma_explorer.app --port 8766
```

The default bind address is `127.0.0.1`. The application does not upload images
or require a GPU. HTML, CSS, JavaScript, and SVG charts are served locally, with
no Node build step, remote fonts, or chart CDN. Other environments can install
this directory's `requirements.txt`.

The local setup started the user service `glioma-atlas.service`. Check or stop it
with:

```bash
systemctl --user status glioma-atlas.service
systemctl --user stop glioma-atlas.service
```

The default data root is `/home/data2/chk/data`; set `GLIOMA_DATA_ROOT` before
starting the service to change it. **Datasets and generated files are stored only
in the data directory; the source tree references them through symlinks.** The
local layout is:

```text
data/
  UCSF-ALPTDG/
    100001/
      100001_time1_t1ce.nii.gz
      ...
    ...
    UCSF_PostopGlioma_Table S1 R1 V5.0_UNBLINDED_FINAL.xlsx
    extraction_manifest.json
  MU-Glioma-Post/
    PatientID_0003/Timepoint_1/     # Four sequences and tumorMask; original identifiers
    ...
    image_manifest.json           # Per-file SHA-256, size, and NIfTI header information
    matching_audit.json            # Imaging/clinical time matching and missing-mask audit
    MU-Glioma-Post_ClinicalData-July2025.xlsx
    MU-Glioma-Post_Segmentation_Volumes.xlsx
    MR_Scanner_data.xlsx
  glioma_explorer/
    runs/                         # HTML, screenshots, check reports, and design previews
      glioma_atlas.html            # Full entry point
      full_data/                  # Full-volume display cache and all worksheets; loaded on demand
      full_preview_status.json    # Full source-file reading and display-cache generation record
      full_browser_checks.json    # Browser loading checks for every patient
```

Directory symlinks under `code/data/` point to the two datasets and
`glioma_explorer`. The recommended HTML entry point is
`code/data/glioma_explorer/runs/glioma_atlas.html`. The HTML and adjacent
`full_data/` are accessed through the same directory link so that relative asset
paths remain valid. The original `code/glioma_explorer/runs` entry point also
remains available. Dataset files and these machine-local links are not stored
in Git; exports and browser checks still write directly to the data directory.
Create the entry points in a new workspace with:

```bash
mkdir -p code/data
ln -s /home/data2/chk/data/UCSF-ALPTDG code/data/UCSF-ALPTDG
ln -s /home/data2/chk/data/MU-Glioma-Post code/data/MU-Glioma-Post
ln -s /home/data2/chk/data/glioma_explorer code/data/glioma_explorer
```

UCSF reads NIfTI files directly from the extracted directory and does not require
the ZIP at runtime. A bounded in-memory cache serves recently accessed cases;
serial MRI decoding limits peak memory use. On September 18, the outer ZIP was
extracted at the user's request while retaining standard `.nii.gz` images. Each
file was independently reread, checked against the original ZIP size and CRC32,
and assigned a SHA-256 recorded in `extraction_manifest.json`. After confirming
that browser access worked, the original ZIP was deleted. Download provenance
and the original ZIP verification records remain in the data directory. Restart
the service after updating local data to rebuild the table cache. MU maps original
PatientID/Timepoint filenames to the clinical tables; `t1c/t1n/t2f/t2w` are
displayed as T1 CE/T1/FLAIR/T2, respectively. Time-point identifiers are not
renumbered or required to start at T1. When a mask is missing, only MRI is shown
and volumes remain missing.

When preparing UCSF on another machine for the first time, run the following
extraction and verification command. It neither deletes the source ZIP nor
overwrites an existing directory:

```bash
PYTHONPATH=code /home/data2/chk/workspace/2026/.venv/bin/python \
  -m glioma_explorer.extract_ucsf --archive /path/to/UCSF_POSTOP_GLIOMA_DATASET_FINAL_v1.0.zip \
  --destination /home/data2/chk/data/UCSF-ALPTDG
```

## MU Download and Provenance

The official [TCIA MU-Glioma-Post](https://www.cancerimagingarchive.net/collection/mu-glioma-post/)
entry point uses Aspera. Authorization for public access succeeded on this
machine, but the SSH transfer connection failed. On September 18, the public
third-party mirror [sbandred/mu-glioma-post-raw](https://huggingface.co/datasets/sbandred/mu-glioma-post-raw)
was found and used over HTTPS instead.

The download pins revision `f6acd4d7d19d35304dc4317d9af4bd25094eb6b9`:
**2,978 NIfTI files, 11,890,059,719 bytes (11.89 GB / 11.07 GiB)**. All 596 TCIA
time-point directories were checked; filenames and byte sizes matched the
mirror. SHA-256 hashes of the two mirrored spreadsheets also matched the
downloaded official spreadsheets. For every image, the downloader verifies the
mirror's LFS SHA-256, gzip CRC, and NIfTI header. Official per-file hashes were
unavailable, so these checks are not described as a byte-for-byte comparison
against the official imaging copy.

```bash
# Safe to rerun: skip verified files and resume .part files.
# Save NIfTI files directly, without an outer ZIP.
PYTHONPATH=code /home/data2/chk/workspace/2026/.venv/bin/python \
  -m glioma_explorer.download_mu --workers 8

# Status of the recorded background download run.
cat /home/data2/chk/data/.mu_glioma_post_download/status.json
systemctl --user status mu-glioma-post-download.service
```

Download completion is determined by `phase`, `verified_files`, and
`verified_bytes` in `status.json`; directory size does not imply verification
has completed. A successful download produces
`MU-Glioma-Post/image_manifest.json`. The recorded run also used
`mu-glioma-post-finalize.service`: it waits for full verification, checks all
local paths and sizes, runs data tests, rebuilds the HTML, restarts the
application, and runs browser checks. Final results go to
`.mu_glioma_post_download/finalization_status.json`, with logs in the adjacent
`finalize.log`. When rerunning the downloader independently, run
`python -m glioma_explorer.finalize_mu` afterward, or manually restart the
application service to read the complete directory. Files retain their original
`.nii.gz` format, with no outer archive to clean up.

## Visualization Rules

- **Orientation and space:** NiBabel converts images to RAS+, using neurological display convention. The full HTML uses the patient's first local time point as the reference grid; selecting another time point does not redefine it. Axial views label L on the left and R on the right; coronal views label S above and I below; sagittal views label P on the left and A on the right. When grids differ, display data is resampled to the first segmentation's grid. Masks use nearest-neighbor interpolation only. Display scaling accounts for physical voxel spacing.
- **Brightness:** Each standard MRI uses the 1st–99.5th percentiles of positive intensities as its display window. The full HTML stores an 8-bit full-volume cache under this window, retaining every spatial voxel and slice. These are display assets and cannot replace the original NIfTI files for quantitative intensity analysis or training. Labels retain their original discrete integers. Difference images use a symmetric window based on the 99.5th percentile of absolute intensity, preserving the display of both positive and negative changes.
- **Default slice:** Select the slice with the largest combined area of labels 1, 2, and 3 in the first segmentation, excluding the resection cavity. Future masks do not determine the default slice. After switching planes, select that plane's default slice from the source time point again.
- **Volumes:** Count original segmentation values 1 / 2 / 3 / 4, multiply by voxel volume in mm³ determined from the affine matrix, and divide by 1,000 to obtain mL. UCSF labels correspond to NCR, SNFH, ET, and RC; MU's first class is NETC. Read and verify the NIfTI millimeter units. Do not treat area as volume or reuse the clinical tables' potentially ambiguous `WT` total column. A label absent from an available original mask has zero volume.
- **UCSF limitation:** The public images have already been registered to the second scan. This project is a retrospective data browser, not a prospective evaluation protocol or clinical prediction system. Changes in compartment volume do not directly establish clinical progression.
- **MU timing:** Retain the distinct counts of 597 numeric clinical MRI time entries, 654 scanner-table records, and 596 imaging time points in the official summary. Sorting and deduplicating numeric day values produces 395 candidate adjacent pairs: 389 have images at both endpoints, and 387 have masks at both endpoints. Numeric clinical dates match 593 imaging time points; another 3 imaging time points lack dates, and 4 clinical time entries have no corresponding images. Detailed IDs are in the data directory's `matching_audit.json`. Patients without numeric times remain in the patient list.
- **MU volume tables:** Show record counts, medians, and interquartile ranges separately for each compartment table. These tables lack a consistent time-point key; row numbers are not treated as chronological order, and missing compartment values are not filled with zero.
- **Summaries:** Age means age at the first scan for UCSF and age at diagnosis for MU. Missing ages are excluded from histograms. When there are more than six diagnosis categories, records outside the five largest categories are grouped as "Other diagnoses (combined)," preserving the patient total. Missing grades are shown separately as "Not recorded."

## Figma Design Sources

The Figma connector was not enabled during this work. The following public design
previews were inspected, informing the sidebar navigation, light cards, spacing,
soft colors, and separation of light and dark content areas. No private designs
were accessed, and no Figma nodes or components were imported.

1. [Healthcare Dashboard](https://www.figma.com/community/file/1026733583562048041/healthcare-dashboard/): the preview came from [Figma's official dashboard template page](https://www.figma.com/templates/dashboard-designs/). It informed the metric cards and light/dark sections.
2. [Medical Doctor Patient Dashboard](https://www.figma.com/community/file/1433514985260691679/medical-doctor-patient-dashboard-template), by Reza Al Hasan: the author's [original Dribbble presentation](https://dribbble.com/shots/25135884-Doctor-Patient-Medical-Live-Dashboard) was inspected. It informed the sidebar, data tables, and selected states.

Inspected previews are stored in `runs/design_references/`. They are design
references only and were not published as application assets. The page retains
clickable links to the design sources.

## Generate and Check

Generating the full HTML reads every original NIfTI for every patient and
verifies a lossless round trip of the compressed display data. Original inputs
are unchanged. Previously generated volumes can be reused when their sources
have not changed. Some original MRI headers for `100075` and `100079` omit
spatial units. Their shapes and affines were checked against files from the same
case explicitly marked in millimeters, and the display cache records the basis
for those units. Original files are not modified, and missing units are not
unconditionally assumed to mean millimeters.

```bash
# Generate the full HTML and full-volume files loaded on demand.
# Resume by reusing patients that have already been generated.
PYTHONPATH=code /home/data2/chk/workspace/2026/.venv/bin/python \
  -m glioma_explorer.export_full --workers 12

# Check data, orientation, label interpolation, and the actual API.
PYTHONPATH=code /home/data2/chk/workspace/2026/.venv/bin/python \
  -m unittest glioma_explorer.test_data -v

# Check all 501 patients in the full HTML, every plane, raw tables,
# and the chat sandbox preview.
PYTHONPATH=code /home/data2/chk/workspace/2026/.venv/bin/python \
  -m glioma_explorer.check_full

# With the service running, use Playwright and local Chrome to check
# online interactions, mobile layout, and offline HTML.
PYTHONPATH=code /home/data2/chk/workspace/2026/.venv/bin/python \
  -m glioma_explorer.browser_check
```

Browser checks additionally require `playwright`; API tests use `httpx` from
the existing environment. Check reports are saved in
[runs/browser_checks.json](runs/browser_checks.json). Screenshots and HTML also
go to `runs/` and are excluded from Git under the repository's rules.

Data sources: [UCSF](https://imagingdatasets.ucsf.edu/dataset/2) and
[TCIA MU-Glioma-Post](https://www.cancerimagingarchive.net/collection/mu-glioma-post/).
Detailed research findings are in the
[CLARITY dataset research notes](https://github.com/MVChem/medical_world_model/blob/31c98ca14173620a3cec67da2defe559d05974da/research_notes/0916_clarity_datasets_feasibility.md).
