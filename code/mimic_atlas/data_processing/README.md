# MedWorld Human-Reviewed Data Preparation

Build classification and temporal manifests from Atlas's full MIMIC-CXR / MIMIC-IV
tables, with human-reviewed CXR heart/lung and brain MRI segmentation. The active
entry point is `code/data/medworld_0922`. It contains final manifests and links to
the original data, without pixel, mask, or feature caches. The preceding MedWorld
implementation was preserved in GitHub commit `b1cad08`.

This document describes the prepared v2 dataset. The newer patient-level
medication selection has separate rules; see [Medication-linked CXR pairs](#medication-linked-cxr-pairs).

## Data and Storage

| Segmentation source | Patients | Annotated images/volumes | Supervised regions |
| --- | ---: | ---: | --- |
| MIMIC human-reviewed heart/lung | 196 | 200 CXR images | Combined lungs, heart |
| UCSF-ALPTDG | 298 | 596 3D volumes | NETC, SNFH, ET, RC |
| MU-Glioma-Post | 203 | 594 3D volumes | NETC, SNFH, ET, RC |
| Montgomery human lung masks | 138 | 138 CXR images | Combined lungs; external test only |

MU has 596 visits. Two lack a standard segmentation mask and remain in the volume
inventory, but are excluded from segmentation supervision. UCSF longitudinal
difference masks are not counted as additional independent segmentation samples.
MRI contributes 1,190 annotated volumes, not 1,190 2D images. All three training
sources use patient-level splits; every MRI time point from the same patient stays
in the same split.

"Human-reviewed" includes model-generated or semiautomatic drafts that experts
checked and corrected. The active v2 dataset does not read CXAS pseudo labels,
old classification selections, old temporal pairs, or old image arrays.
Montgomery reuses only the original human masks and images. The current code
supports v2 only; the old loaders, CXAS exporter, and legacy path remapping have
been removed. Historical experiment results and data remain in their original
directories.

Raw datasets are stored under `/home/data2/chk/data`. The newly added public
annotations reside at `/home/data2/chk/data/heart-lung-segmentations-data/1.0.0`,
exposed through the project symlink `code/data/heart_lung_human`. MRI uses the
existing `UCSF-ALPTDG` and `MU-Glioma-Post` directories under the same central data
root. The prepared dataset is also exposed through `code/data/medworld_0922`,
which links to the central `medical_world_model/medworld_0922` directory.

All 540 publisher-provided SHA256 checksums were verified after downloading the
public heart/lung annotations. Original PNG files are unchanged. Every MRI T1ce
input and target mask has a recorded SHA256 checksum, verified when loaded.
Sources: [heart/lung annotations](https://physionet.org/content/heart-lung-segmentations-data/1.0.0/),
[UCSF paper](https://pubs.rsna.org/doi/pdf/10.1148/ryai.230182), and
[MU paper](https://www.nature.com/articles/s41597-025-06011-7). Each source's
license and access conditions continue to apply.

## Build and Validate

Run from the repository root using the existing Python environment. Dependencies
are listed in `requirements.txt`; loader validation also requires PyTorch.

```bash
export PYTHONPATH=code
PYTHON=/home/data2/chk/workspace/2026/.venv/bin/python

# Existing dataset: archive the old manifests after a successful build, then publish.
$PYTHON -m mimic_atlas.data_processing --replace

# Check the actual MedWorld loader, patient isolation, image/mask decoding,
# and a CPU segmentation-head forward/backward pass.
$PYTHON -m mimic_atlas.data_processing.validate

# Verify the public human heart/lung annotations without downloading them again.
$PYTHON -m mimic_atlas.data_processing.human_cxr --verify-only
```

The builder first writes to a temporary directory. Overwriting is rejected by
default. `--replace` accepts only a data directory containing the
`.medworld-prepared` marker and archives it as
`medworld_0922_previous_<timestamp>`. If the project entry point is a symlink, the
build and archive are created alongside its central target, preserving the
symlink. Publication uses two directory renames with rollback on failure. These
renames are not an atomic exchange, so rebuild outside training data-loading
periods.

For other configurations, choose a new output under the central data directory,
create a project symlink, and update the data entry point in the configuration.
`--pair-mode all` exports every eligible chronological combination. The default,
`adjacent_random`, retains all adjacent pairs and selects up to 8 nonadjacent
pairs per patient. `--max-patients 100` limits only CXR/IV matching; segmentation
and VQA are still processed in full. Logs go to `runs/build_YYYYMMDD`, and
validation results to `runs/validate_YYYYMMDD/validation.json`. Neither is tracked
in Git.

## Matching and Sampling Rules

- Join CXR and IV by exact `subject_id`. By default, each selected image's
  acquisition time must map unambiguously to the same admission interval, with
  inclusive bounds from `min(edregtime, admittime)` to `dischtime`. This is more
  conservative than the earlier "unique shared admission" rule: ambiguity at
  either endpoint also rejects the pair.
- Preserve the full patient timeline. Adjacent pairing does not skip an unusable
  intermediate study. Sort studies by their earliest acquisition time and reject
  tied times or overlapping acquisition windows. Compute the gap from the actual
  selected images' acquisition times.
- Both endpoints must share an AP or PA view; prefer a shared PA view. For each
  view, use Atlas's existing largest-image selection rule, breaking ties by
  DICOM ID.
- The default interval is 1 hour to 365 days, adjustable with `--min-gap-hours`
  and `--max-gap-days`. `adjacent` selects adjacent pairs only; `random` selects
  random nonadjacent pairs only; `adjacent_random` combines them; `all` exports
  every eligible chronological combination.
- Select the random subset from all eligible nonadjacent combinations using a
  hash of the fixed seed and identifiers. Selection is reproducible and has no
  duplicates; memory retains only a bounded set of candidates per patient.
  Sampling does not depend on future label values or report content. Temporal
  supervision requires nonempty reports and label records at both endpoints;
  missing and uncertain label states are preserved.
- `--linkage patient` explicitly allows pairs across admissions or without a
  matched admission for the same IV patient. It is not enabled by default. Match
  status and candidate admission/ICU IDs for every study are recorded in
  `study_links.jsonl`.
- Use the official CXR patient splits. Existing VQA/segmentation holdouts take
  precedence in the order `human_test > test > validate > train`. Conflicting
  classification/temporal rows are removed, not moved into the test set.
  MedWorld then checks global patient isolation across all tasks. Manifest
  statistics describe exported rows; validation statistics describe the final
  usable rows.

## Manifests and MRI Decoding

```text
medworld_0922/
  manifest.json                 # Rules, sources, counts, and file hashes
  classification.jsonl          # 13 classification labels
  segmentation.jsonl            # All reviewed segmentation: CXR image rows + MRI slice rows
  mri_volumes.jsonl              # 1,192 MRI visits; four sequences and missing-mask status
  mri_segmentation.jsonl         # Axial slice references for 1,190 annotated volumes
  study_links.jsonl             # CXR/IV provenance and linkage audit
  temporal/
    observations.jsonl
    train.jsonl
    validate.jsonl
    test.jsonl
  images -> original MIMIC-CXR/files
  iv -> original mimic-iv-3.1
  vqa -> official CXR-VQA/dataset
  segmentation/
    heart_lung_human -> public human heart/lung annotations in the central directory
    montgomery -> original human lung masks and images
  mri/
    ucsf -> central UCSF-ALPTDG
    mu -> central MU-Glioma-Post
```

The current MRI model input is a **T1ce 2D axial slice**. Original T1/T2/FLAIR
file references remain in the volume inventory but are not yet used as a joint
four-sequence input. Images and masks are reoriented to RAS while preserving
physical aspect ratio. Images are placed on a 512 x 512 canvas; masks are resized
to 256 x 256 with nearest-neighbor interpolation. The intensity window uses the
1st and 99.5th percentiles of positive input-image voxels. Slice bounds depend
only on the image, with a check that no annotated foreground falls outside them.
Tumor-free slices within those bounds are retained. The default
`--mri-axial-stride 1` uses every axial plane in that range.

The six output channels are fixed as `lungs, heart, NETC, SNFH, ET, RC`. CXR
supervises only the first two channels, MRI only the last four, and Montgomery
only the combined-lung channel. Unannotated channels and padding are excluded
from the loss. The new configuration samples the three training datasets evenly.
MRI slices are traversed within randomly ordered volume blocks, preventing the
large MRI slice count from overwhelming CXR and avoiding decompression for every
slice. Decoding uses only a bounded in-memory cache.

Classification retains 13 CheXpert labels, excluding `No Finding`, with values
`-2/-1/0/1`. Missing and uncertain labels are masked in the loss. Temporal pairs
use the actual time difference and support reversed examples with negative time
differences. IV linkage provides provenance and is not currently an EHR model
input. Classification and segmentation receive images only; VQA receives an
image and a question. The temporal source-only interface does not read the
target image or report.

## MedWorld Configuration and Evaluation

The prepared-v2 configuration is `code/medworld/configs/medworld_0922.json`, with
`segmentation_channels=6`, `segmentation_sampling=balanced_dataset`, and generic
imaging prompts. A six-channel segmentation head cannot directly resume from
an old three-channel segmentation-head checkpoint.

```bash
$PYTHON -m medworld.run_experiment \
  --config code/medworld/configs/medworld_0922.json \
  --gpus 1,2 --out code/medworld/runs/paired_YYYYMMDD
```

This preparation run built and validated the dataset without starting formal
training. After training, both slots and no-slots arms must be evaluated on
classification, VQA, human CXR heart/lung segmentation, UCSF/MU MRI segmentation,
and Montgomery human lung segmentation. Report mean IoU and Dice separately
for each dataset. For MRI, first aggregate intersection/union counts across
slices within each volume on the 256 x 256 evaluation grid, then average across
volumes. For CXR, average across images. Overall metrics are macro-averaged
across datasets. Native Qwen has no segmentation head, so its segmentation
results are N/A.

## Local Build and Validation Results: 2026-09-23

The full build and actual MedWorld loader checks completed. Global patient
isolation holds. All 200 human CXR images, 1,190 annotated MRI volumes, and 138
Montgomery images were retained; no segmentation rows were removed because of
cross-task conflicts.

| Dataset (counting unit) | Train | Validate | Test | Human test |
| --- | ---: | ---: | ---: | ---: |
| Classification (images) | 143,592 | 1,255 | 2,475 | — |
| VQA (questions) | 280,441 | 72,825 | 13,793 | — |
| Temporal (forward pairs) | 104,967 | 930 | 1,848 | — |
| Human heart/lung (CXR images) | 142 | 29 | 29 | — |
| UCSF (annotated volumes) | 416 | 88 | 92 | — |
| MU (annotated volumes) | 433 | 80 | 81 | — |
| Montgomery (CXR images) | — | — | — | 138 |

MRI references 165,417 2D axial planes; these slices are not counted as independent
annotated volumes. The 107,745 forward pairs consist of 64,800 adjacent pairs and
42,945 random nonadjacent pairs. Patient isolation removed 9,590 VQA training
questions and 742 validation questions; the test set remains at 13,793 questions.

Validation traversed every temporal pair, sampled actual decoding from each
segmentation source and split, and completed a CPU forward/backward pass through
the six-channel segmentation head using mixed CXR/MRI inputs. It did not load
foundation models or start formal training. Full results are in
[validation.json](runs/validate_20260923/validation.json).

Source tests: the full suite passed 128 tests and 35 subtests. Two subsequently
added symlink publication/rollback tests and the updated adapter tests also
passed.

## Medication-linked CXR pairs

[`data_preprocessing`](../../data_preprocessing/README.md) selects same-patient chronological CXR pairs with their
own reports and an observed administration between images. It does not require
a common admission or limit the time gap. Commands, rules, results, and the
proposed treatment encoder are kept in its [2026-09-23 run](../../data_preprocessing/runs/medication_filter_20260923/README.md).
This selection export requires a new event-aware loader before MedWorld training.
