# Linked MIMIC current-state -> future-state example

This directory contains a small, runnable MIMIC-CXR example for the transition
unit proposed in `../../26iclr`, plus a separate linker that attaches
retrospective MIMIC-IV v3.1 context for audit and appendix figures:

```text
(current CXR, current report, current structured state, elapsed time)
    -> (observed future CXR, future report/state for supervision or evaluation)
```

The builder groups studies by patient, uses `StudyDate` + `StudyTime` for
chronology, keeps the full patient timeline, pairs strictly adjacent studies,
then chooses an exact common AP/PA projection on both sides. It preserves the
official patient split and never skips an intervening incompatible study while
calling the result "adjacent." Automatic cohorts default to deterministic,
seed-controlled sampling with no future-label filter. In one-per-patient mode,
the pair is chosen within each patient before patients are ordered, and neither
operation uses future report or label values.

Some images within one study have slightly different acquisition times. The
study timestamp is the earliest image time, the full acquisition span is kept
in the packet, and ambiguous equal-timestamp study pairs are rejected.

## How MIMIC-CXR links to MIMIC-IV

The link is real, but it is not `study_id = hadm_id`:

```text
MIMIC-CXR subject_id + StudyDate/StudyTime
    -> exact MIMIC-IV subject_id
    -> timestamp contained in one ED/admission interval
    -> hadm_id
    -> timestamp contained in an ICU interval (when applicable)
    -> stay_id, care unit, diagnoses, procedures, ICU events
```

`subject_id` is the only direct patient key. `study_id` and `dicom_id` belong
to MIMIC-CXR, while `hadm_id` and `stay_id` belong to MIMIC-IV. Both datasets
use an aligned de-identified timeline for a shared subject, so acquisition-time
containment supplies the episode link. A generic corpus join must retain
unmatched and ambiguous cases rather than guessing; the four curated cases in
this example each have one common admission.

MIMIC-III v1.4 is also present under the local MIMIC directory, but its public
patient identifiers and date shifts do not provide a patient-level crosswalk to
MIMIC-IV/CXR. Do not draw it as a third database in the same-patient timeline.

## Build and view the CXR examples

```bash
cd /home/data2/chk/workspace/2026/08/04/medical_world_model/code/MIMIC_example

/home/data2/chk/workspace/2026/.venv/bin/python build_mimic_transitions.py \
  --mimic-cxr-root /home/data2/chk/data/MIMIC \
  --output-dir example_output \
  --curated-pairs curated_pairs.json \
  --num-examples 4

python -m http.server 8000 --directory example_output
```

Then open <http://localhost:8000>. The default `symlink` asset mode keeps the
large JPG files in their original MIMIC location. Use `--asset-mode copy` only
if a standalone local gallery is needed, or `--asset-mode none` if no gallery
images should be materialized.

Output is built in a validated staging directory and swapped as one bundle.
Reruns replace only directories carrying the builder's marker file; an
unrecognized nonempty output directory is refused rather than overwritten.

The argument can also point at `/home/data1/data/MIMIC`; the builder detects its
`MIMIC_CXR` child automatically. Run `python build_mimic_transitions.py --help`
for view, time-gap, split, and binary-label-flip filters.

## Attach MIMIC-IV context

Run the linker after building `example_output`:

```bash
/home/data2/chk/workspace/2026/.venv/bin/python link_mimic_iv_context.py \
  --mimic-root /home/data2/chk/data/MIMIC \
  --transitions example_output/transitions.jsonl \
  --output-dir linked_output \
  --include-icu-inputs

/home/data2/chk/workspace/2026/.venv/bin/python -m http.server 8000 --directory .
```

Then open <http://localhost:8000/linked_output/>. Omit
`--include-icu-inputs` for a faster admissions/ICU/diagnoses/procedures-only
run; that avoids scanning the much larger `icu/inputevents.csv.gz` table.

The linked outputs are:

- `linked_output/index.html`: current CXR, linked observational timeline, and
  target CXR in three columns.
- `linked_output/linked_transitions.jsonl`: full CXR packet plus the uniquely
  resolved admission/stay and traceable clinical context.
- `linked_output/appendix_cases.csv`: one compact row per case for plotting or
  assembling a paper figure.
- `linked_output/summary.json`: linkage counts, enabled tables, and caveats.

Database provenance is explicit in the linked gallery, JSONL, and CSV:

| Displayed or stored data | Source database | Role here |
|---|---|---|
| Current/follow-up radiographs, `study_id`, `dicom_id`, acquisition time/view, radiology reports, official split, and CheXpert report labels | MIMIC-CXR-JPG v2.0.0 | Current-side fields may be forecast inputs; future-side fields and label deltas are target/evaluation only. |
| `hadm_id`, admission interval, transfers/care unit, `stay_id`, diagnoses/procedures, and interval ICU procedure/input events | MIMIC-IV v3.1 (`hosp` and `icu`) | Retrospective audit and figure context only; excluded from model inputs in this task. |
| Adjacent-pair choice, elapsed interval/horizon bin, label flips, linkage status/event counts, and curation notes | Derived by this project | Not raw fields from either source database. |

`subject_id` and aligned de-identified timestamps form the bridge. `study_id`
and `dicom_id` remain CXR-only identifiers; `hadm_id` and `stay_id` remain
MIMIC-IV-only identifiers. MIMIC-III is not used in these galleries.

For the current ICLR appendix draft, the most compact three-row selection is:

| Role | subject_id | CXR interval | MIMIC-IV episode | Why it is useful |
|---|---:|---:|---|---|
| A | 12137189 | 20.1 h | hadm 28837774 / SICU stay 36697114 | clear edema/opacity change |
| B | 11137177 | 24.0 h | hadm 26133494 / SICU stay 32062426 | stable effusion/atelectasis control |
| C | 16254738 | 50.5 h | hadm 26192634 / TSICU stay 34370033 | marked opacity change with dense interval context |

The fourth case, subject 16562665 in admission 28565902, is a useful
uncertainty example: the follow-up report cannot confidently exclude a small
pneumothorax. It is hospitalized but no longer inside an ICU interval at the
two selected acquisition times, which usefully demonstrates that a valid
admission link does not imply a `stay_id` match.

For a paper figure, label the middle column "linked retrospective context—not
model input". Post-current medications/procedures are observed co-timed events,
not evidence that an intervention caused the radiographic change. ICD
diagnoses are admission-level discharge coding; ICD procedure `chartdate` has
date-only precision. Future reports and CheXpert values remain evaluation-only.

`curated_pairs.json` is a small local demonstration set: two visibly clear
changes, one stable follow-up, and one uncertainty-limited follow-up. The notes
record local visual/report review, not clinical adjudication. Omit
`--curated-pairs` to use deterministic automatic sampling. A deliberately
target-conditioned audit gallery remains available via
`--selection-strategy change_enriched --allow-multiple-per-patient`, but must
not be used as a training cohort.

## Purposefully diverse 10-transition audit bundle

`representative_pairs_10.json` defines a larger local audit bundle without
changing the pair-wise forecasting contract. It contains 10 strictly adjacent
transitions from seven patients:

- the original four clear/stable/uncertain examples;
- one four-time-point AP chain (three connected transitions at 9.2, 22.9, and
  24.1 hours);
- one three-time-point AP chain (a 47.1-day cross-episode gap followed by a
  5.3-day inpatient follow-up); and
- one separate PA example with a 53.0-day gap.

The selected horizon distribution is four `0-24h`, three `24-72h`, one `3-7d`,
and two `>7d` transitions. Nine rows are AP/train and one is PA/test. Multiple
time points are represented as connected adjacent pairs: the target study and
image of one row are the source study and image of the next row. This is a
purposefully diverse, target-conditioned audit set, not a population-level
representative sample or an unbiased training cohort.

Build the local bundle with:

```bash
/home/data2/chk/workspace/2026/.venv/bin/python build_mimic_transitions.py \
  --mimic-cxr-root /home/data2/chk/data/MIMIC \
  --output-dir representative_output_10 \
  --curated-pairs representative_pairs_10.json \
  --num-examples 10 \
  --split all \
  --view frontal \
  --min-gap-hours 1 \
  --max-gap-days 180 \
  --allow-multiple-per-patient

/home/data2/chk/workspace/2026/.venv/bin/python link_mimic_iv_context.py \
  --mimic-root /home/data2/chk/data/MIMIC \
  --transitions representative_output_10/transitions.jsonl \
  --output-dir representative_linked_output_10 \
  --include-icu-inputs
```

The checked-in selection file is small; the two generated directories remain
local and ignored. The current linked bundle resolves eight transitions to one
common admission and intentionally retains two long-gap unmatched transitions
instead of guessing an episode. Serve this directory and open
`/representative_output_10/` or `/representative_linked_output_10/` to inspect
the galleries.

## Build serious training/validation cohorts

A reproducible one-transition-per-patient training pilot is:

```bash
python build_mimic_transitions.py \
  --mimic-cxr-root /home/data1/data/MIMIC/MIMIC_CXR \
  --output-dir /restricted/mimic_vla_jepa/train_seed42 \
  --split train \
  --view frontal \
  --min-label-flips 0 \
  --selection-strategy deterministic_random \
  --seed 42 \
  --num-examples 20000 \
  --asset-mode none \
  --no-gallery
```

The seed is hashed with stable identifiers; Python's process-randomized
`hash()` is not used. Changing future CheXpert values does not change pair or
patient order. Patients with more eligible follow-ups do not get extra chances
to enter a size-limited one-per-patient cohort.

Use `--all-matching` when every eligible adjacent pair is wanted and its count
is not known in advance. It implies multiple pairs per patient:

```bash
python build_mimic_transitions.py \
  --mimic-cxr-root /home/data1/data/MIMIC/MIMIC_CXR \
  --output-dir /restricted/mimic_vla_jepa/validate_all \
  --split validate \
  --view frontal \
  --min-label-flips 0 \
  --selection-strategy deterministic_random \
  --seed 42 \
  --all-matching \
  --asset-mode none \
  --no-gallery
```

Do not use `--min-label-flips > 0` for an unbiased cohort: it makes inclusion
depend on future labels. One-per-patient automatic builds reject that
combination. `summary.json` records the strategy, seed, requested limit, and
whether the policy is target-conditioned.

## Outputs

- `index.html`: side-by-side `current -> observed future` audit gallery; omitted
  with `--no-gallery`.
- `transitions.jsonl`: the clearest nested transition-packet representation.
- `forecast_manifest.jsonl`: compatible with the existing TC-JEPA loader's
  `source_image`, `target_image`, `prompt` contract; future report/state fields
  are explicitly suffixed `for_eval_only`. This convenience file co-locates
  inputs and targets, so any loader must whitelist `source_image` and `prompt`
  rather than serialize the whole row into a model input.
- `forecast_inputs.jsonl`: strict inference-time rows with **no target image,
  exact realized interval, future report, future state, or state delta**.
- `forecast_targets.jsonl`: future-side targets joined to inputs only by
  an opaque `transition_id` digest (the ID does not spell out the future study
  or DICOM identifier).
- `summary.json`: filters, rejection/audit counts, and important limitations.

The training prompt renders the current report and one prespecified coarse
horizon bin: `0-24h`, `24-72h`, `3-7d`, or `>7d`. The exact realized follow-up
interval remains audit/target metadata and is absent from strict inference
inputs. `source_state` is available as a separate structured input for models
that explicitly consume it; the existing image+prompt loader treats it only as
metadata. Flat rows also include root-relative image paths alongside the
immediately runnable absolute paths.

A transition packet has this shape (abridged):

```json
{
  "patient_id": "p...",
  "current_state": {
    "timestamp": "...",
    "image": {"path": "...", "view": "AP"},
    "text": {"findings": "...", "impression": "..."},
    "structured_findings": {"Edema": "absent"}
  },
  "interval": {"elapsed_hours": 24.0, "horizon_bin": "0-24h", "actions": []},
  "future_state": {
    "role": "observed_followup_anchor",
    "timestamp": "...",
    "image": {"path": "...", "view": "AP"},
    "text_for_evaluation_only": {"findings": "...", "impression": "..."},
    "structured_findings_for_evaluation_only": {"Edema": "present"}
  },
  "state_delta_for_evaluation_only": {
    "binary_chexpert_label_flips": [
      {"finding": "Edema", "current": "absent", "future": "present", "change": "changed_to_present"}
    ]
  }
}
```

## Leakage boundary

The forecasting prompt is constructed by the shared builder/training contract
from the current report and coarse horizon bin only. The exact observed
interval, future report, target state, and label delta may be used as metadata
or evaluation targets, but never as prompt input. If you build a
controllable/oracle editing task later, give it a different `task_type`;
putting `state_delta` in a row advertised as open forecasting changes the
scientific task.

## What this first example does not claim

- The observed follow-up is one retrospective outcome, not a deterministic or
  uniquely correct future.
- CheXpert states are weak labels extracted from radiology reports. This demo
  conservatively calls binary flips `changed_to_present` or
  `changed_to_absent`; it does not infer clinical onset, resolution, improved,
  or worsened severity without a text extractor and evidence spans.
- Curated and `change_enriched` galleries intentionally use target-side
  information. They are audit examples, not unbiased training cohorts.
- Exact AP/PA matching does not control posture, inspiration, rotation,
  magnification, or support-device changes. Acquisition details are shown in
  the gallery and should be treated as confounders.
- The time of the next CXR reflects clinical ordering and observation, so a
  variable-gap pair set is not the same as fixed-horizon prognosis. Use explicit
  horizon bins for benchmark experiments.
- `actions` are intentionally empty. Linking medications/procedures from
  MIMIC-IV is a separate extension and would provide observational context, not
  a causal treatment effect.
- The data and generated artifacts remain governed by the MIMIC data-use
  agreement. Do not redistribute the gallery or manifests as unrestricted data.

## Tests

The tests create a tiny synthetic MIMIC-like fixture; they do not read real
MIMIC data:

```bash
PYTHONPATH=.. python -m unittest discover -s tests -v
```

The current implementation favors clarity over corpus-scale efficiency: a full
metadata scan is required even for a few examples, and `--all-matching` still
retains generated rows in memory. `--no-gallery` avoids constructing a huge
HTML document and image-asset tree, but is not a streaming implementation. On
this local installation, a four-example run takes about 40 seconds and peaks
near 590 MiB RAM. Very large exports should move to a streaming or
SQLite/Parquet pipeline if memory becomes limiting.
