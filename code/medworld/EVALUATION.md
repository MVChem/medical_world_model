# Fixed Table 1/2 evaluation protocol

The schema is `medworld-tables-v1`. The current three rows are raw-input task
decoder, Qwen3.5-9B zero-shot, and MedWorld with slots. Training and native
implementations share reference construction and scoring functions. Historical
metric formats, pseudo-mask tests, and results from earlier model architectures
are not converted into this protocol.

## Table 1

| Task | Primary metric | Fixed reference and prediction |
| --- | --- | --- |
| Future VQA | Accuracy | One official categorical answer; normalized fixed vocabulary |
| Progression | Balanced accuracy | Mean of improved/stable/worsened recalls over source-anchored human region/finding test annotations |
| Future report | RadGraph F1 | Official RadGraph-XL partial F1, averaged over full original reports |
| 30-day mortality | AUROC | Continuous risk of death within 30 days after source acquisition |
| Remaining LOS | MAE in days | Nonnegative prediction of discharge time minus source acquisition time |

All tasks use source image/report evidence only. Future VQA, progression, and
report tasks request their actual positive prediction interval. Mortality uses
720 hours; LOS uses a fixed 24-hour latent readout request, which is independent
of observed remaining stay. LOS is not a 24-hour outcome. Questions identify the
requested future answer or region/finding without revealing its direction.

Future VQA selects the closest eligible earlier frontal examination for each
official target image from the full CXR timeline. Future reports select their
closest eligible source among the forward pairs in `medworld_0923`. Neither
selector examines target labels. These are separately declared task cohorts;
future VQA does not require medication-positive pairing. Explicit source/current
image links and existing global patient holds define a derived longitudinal VQA
split. It is not the published current-image VQA test split: annotations from
any original partition may contribute only to their patient's fixed global
partition. Future test patients' annotations are absent from every training
task. Original annotation partition/ID remain recorded.

Explicit source/current image links define progression: binary presence at both endpoints cannot establish
stability. Existing global holdouts reserve all human Chest ImaGenome gold
patients for test, so train/validation use explicitly anchored silver report
comparisons from other patients. Conflicting directions are excluded and
provenance is recorded separately. Silver comparisons are not segmentation masks.

Outcome preparation starts from full linked CXR study metadata, not just the
follow-up or medication-positive pair list. Each source must lie within one
unambiguous admission/linked ED interval. Death time and discharge time are
labels only. Exact recorded inpatient death takes precedence over date-only
`dod`; dates overlapping the start or 30-day boundary are excluded. Negative
mortality labels require known survival beyond the horizon or adequate
ascertainment under MIMIC-IV 3.1's one-year follow-up after last discharge.
Censored and contradictory records are excluded before inference. LOS includes
discharge by death. Repeated examinations share patient-level splits.
See the [official MIMIC-IV documentation](https://physionet.org/content/mimiciv/3.1/).

Reports use the same first 383 UTF-8 source bytes for every method. Greedy future
report output is capped at 2,047 UTF-8 bytes. Training teacher-forces up to the
2,048-position decoder budget, including EOS; full original target text is
retained for scoring. The official scorer source, model, tokenizer, and isolated
worker runtime are pinned in `radgraph_assets/manifest.json`. No GREEN, lexical
report score, Brier, or ECE columns remain in the current table. See the
[official RadGraph implementation](https://github.com/Stanford-AIMI/radgraph).

Native Qwen uses a fixed, verified single-token alphabetic option codebook for
VQA and progression, Yes/No logits for mortality, and a probability expectation
over `[0, .5, 1, 2, 3, 5, 7, 10, 14, 21, 30, 60, 90, 180, 365]` days for LOS.
These choices are fixed before testing; native scores are not fitted calibration.

## Table 2

| Task | Primary metrics | Aggregation |
| --- | --- | --- |
| Classification | Macro AUROC / AP | 13 report-derived labels; known 0/1 references; same eligible findings with both classes |
| VQA | Exact match / micro F1 | Fixed JSON answer sets, lowercase and trim surrounding whitespace; malformed answers remain errors |
| Segmentation | Mean Dice / IoU | Human-reviewed masks; MRI confusion counts summed within volume, then channel/volume averaging |

The same-exam report is withheld for all current tasks, including from the slots
encoder. Current VQA uses a fixed sample of 100 Verify, 100 Choose, and 100 Query
questions with seed 42; all three methods share the exact 300 IDs and references.
This is a stratified test subset, not the full VQA test score.

Internal segmentation datasets are MIMIC human heart/lung, UCSF-ALPTDG, and
MU-Glioma-Post. The main score equally averages their dataset means. External
Montgomery is reported separately. Each trained arm must report both metrics
for every dataset. Native Qwen has no segmentation head and receives N/A.
Sigmoid predictions use threshold `> 0.5`; only annotated pixels/channels count;
empty prediction against empty reference scores one. MRI volumes, not slices,
are evaluation units.

## Completion and traceability

Fractions are stored on their original 0–1 scale and displayed as percentages;
LOS remains days. Every prediction file binds to fixed reference IDs and hashes.
Missing records never reduce the denominator. Invalid categorical predictions
count as errors. Missing or invalid numeric predictions make a task incomplete,
and an incomplete task has no aggregate score. Both outcome classes and all
three progression classes must have support for their primary scores.

The exporter checks completed final weights, matched initialization, all eight
training sample counts, equal updates/world size, actual pretrained asset
hashes, identical data protocols, per-task reference/scorer hashes, and all
applicable metrics. Partial smoke outputs cannot enter the complete tables.
Artifacts remain in the original run: source snapshots, manifest fingerprints,
per-example predictions, per-task protocols, summaries, and the two Markdown
and JSON tables. Additional methods require separately matched runs.
