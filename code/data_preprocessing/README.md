# Medical data preprocessing

Independent dataset preparation code. Original images, reports, and clinical
tables stay in their source directories; `code/data` exposes final datasets by
symbolic links. Detailed configurations, source snapshots, commands, and results
belong in dated `runs/` directories. No recomputable image, mask, or feature
caches are written to disk.

The organization follows the existing [reviewed MedWorld data pipeline](../mimic_atlas/data_processing/README.md),
which documents classification, VQA, reviewed CXR/MRI segmentation, and its
previous temporal cohort. The medication cohort below uses patient-level
selection without admission or gap restrictions.

## Data and storage

| Resource | Project entry |
| --- | --- |
| Original chest images and study reports | `code/data/MIMIC_CXR` |
| Original medication tables | `code/data/mimic-iv-3.1` |
| Patient-level CSV positions | `code/mimic_atlas/runs/patient_index_20260918_offsets` |
| Medication-positive pair selections | `code/data/medworld_0923` |
| Existing reviewed classification/VQA/segmentation data | `code/data/medworld_0922` |

The new selection's actual directory is
`/home/data2/chk/data/medical_world_model/medworld_0923`.
It contains `manifest.json`, four split-specific `*.jsonl.gz` files, and
`evidence.jsonl.gz`. Pair rows reference original images, reports, and a patient
evidence catalog. The catalog records source table positions and identifiers;
the original CSV tables remain authoritative for medication names and doses.

## CXR pairs with recorded medication

Select all chronological combinations of the same patient's usable frontal
chest examinations. Each endpoint must have its own available image and a
nonempty report. Retain a pair when at least one actual administration occurs
between the acquisition times, or an active infusion overlaps that interval.

There is no common-admission requirement, minimum/maximum gap, same-view
requirement, or finding-label requirement. Choose PA if available, otherwise AP,
using the largest image within the selected view. Preserve each endpoint's view,
actual time, image path, and report path. The interval itself remains a recorded
quantity for treatment timing and later model conditioning.

Actual-administration evidence comes from eMAR and medication-category ICU
inputevents. Prescriptions alone do not qualify. eMAR product details do not
count as additional administrations, and source records are not automatically
deduplicated across eMAR and ICU. The run README defines accepted states and
boundary handling precisely.

| Entry | Location |
| --- | --- |
| Selection CLI | `filter_medicated_pairs.py` |
| Administration predicates | `medication_events.py` |
| Original-record lookup | Atlas patient byte-offset index |
| Dataset | `code/data/medworld_0923` |
| Complete rules, counts, verification, commands | [2026-09-23 run](runs/medication_filter_20260923/README.md) |
| Proposed treatment encoder | [Design](runs/medication_filter_20260923/treatment_conditioning_design.md) |

## Build and verify

From the repository root, use the existing Python environment. Replace
`YYYYMMDD` and choose new output, project-link, and run paths:

```sh
PYTHONPATH=code code/.venv/bin/python -m data_preprocessing.filter_medicated_pairs \
  --patient-index code/mimic_atlas/runs/patient_index_20260918_offsets \
  --output /home/data2/chk/data/medical_world_model/medworld_rebuild_YYYYMMDD \
  --project-link code/data/medworld_rebuild_YYYYMMDD \
  --run-dir code/data_preprocessing/runs/medication_filter_rebuild_YYYYMMDD \
  --workers 4
```

The command validates source fingerprints and actual image/report availability,
reads each patient's medication tables once, and publishes the project link only
after successful completion. Existing outputs are never overwritten.

```sh
PYTHONPATH=code code/.venv/bin/python -m data_preprocessing.filter_medicated_pairs --help
PYTHONPATH=code code/.venv/bin/python -m pytest -q code/data_preprocessing/tests
```

Rebuilding requires a new output directory; the exact original invocation and
the equivalent command for this package are kept with the completed run. Clinical
records are read by patient offsets, not by repeatedly scanning the full tables.
Only final pair selections and source evidence references are exported. Reports,
pixels, doses, and raw clinical rows are not copied into the selection files.

## Inspect in MIMIC Atlas

Open [Medication pairs](http://127.0.0.1:8767/#medications) in the local Atlas
service. The page reads this selection directly, resolves both original images
and their own reports, and rereads accepted interval medications from the
original eMAR and ICU tables. Browse by patient or split, open a pair, and
expand individual records to inspect the source fields and dose details.
The medication table is paginated without discarding records.

The displayed interval has no padding or admission restriction. Source record
counts are not deduplicated doses, and an infusion's original amount can cover
time outside the image interval. Original timestamps and fields remain visible
for review. Pair links can be bookmarked or shared through the same local
service. See the [Atlas documentation](../mimic_atlas/README.md#medication-pair-review)
for startup, data-path configuration, and validation.

## Train without medication conditions

The current dataset name is `medworld_0923`; it was renamed from
`medworld_medication_20260923` without changing its manifest or compressed shards.
The MedWorld temporal adapter reads the pair image/report references and actual
time intervals directly. Medication records, evidence references, doses, and
clinical tables are not loaded for training. Both endpoints provide their own
image/report; only the source representation and signed interval enter the world
model, while the other endpoint provides the latent target. Temporal examples
do not require finding labels.

Use [`medworld_0923.json`](../medworld/configs/medworld_0923.json). It selects this
temporal cohort together with the reviewed classification/VQA/segmentation data
at `code/data/medworld_0922`. Global patient holdouts apply across all tasks, so
effective training counts may be lower than the raw split counts shown in Atlas.
Medication conditioning remains separate future work; no formal training run
was launched during this data preparation or integration.

The complete adapter check passed on 2026-09-23. After cross-task patient
holdouts, temporal train / validate / test contain 888,117 / 10,611 / 26,344
pairs. All four task batch checks passed with zero clinical/evidence file reads.
See the [integration run](../medworld/runs/temporal_0923_no_medication_20260923/README.md)
for exact counts, commands, and validation results.

## Latest completed selection

The 2026-09-23 run retained **1,022,127 pairs from 21,097 patients** out of
1,167,399 candidates. All selected rows passed verification; 24 sampled pairs
also matched independently re-read original clinical records. All 69 focused
tests passed after moving the code into this package. Detailed split counts,
record-density distributions, and exact commands remain in the run directory.

## Table 1/2 future supervision

[`build_table12.py`](build_table12.py) adds final supervised task labels under
`code/data/medworld_0923/table12_v1` without copying original images or reports.
It creates fixed train/validation/test records for future VQA, progression,
future reports, 30-day mortality, and remaining length of stay. Clinical outcomes
are built from full linked source examinations and do not require subsequent
images or medication records. The manifest records patient holds, original
source hashes, exclusions, and positive/negative/class support.

```sh
PYTHONPATH=code code/.venv/bin/python -m data_preprocessing.build_table12 \
  --config code/medworld/configs/medworld_0923.json \
  --output code/data/medworld_0923/table12_v1
```

Choose a new output directory for any rebuild; completed manifests are immutable.
Rows link back to original CXR observations and Atlas patient views. Explicit
human Chest ImaGenome comparisons remain test-only; source-anchored silver
comparisons on other patients supply progression training/validation. Outcomes
use documented acquisition, admission, discharge, death, and follow-up rules.
See the [fixed evaluation protocol](../medworld/EVALUATION.md) and
[preparation run](runs/table12_20260923/) for full definitions and actual counts.
