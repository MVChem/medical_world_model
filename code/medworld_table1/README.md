# MedWorld-JEPA Table 1 small-model pilot

**September 13 expanded forecast ablations:** [protocol and verification](../../research_notes/0913_table1_expanded_overnight.md).
The prepared overnight queue uses 16,000 training pairs, unchanged 230 validation / 297 test pairs,
and matched Qwen0.8B learned-slots, full-token no-slots, and other-patient shuffled-state conditions.
The immutable launch manifest is `runs/overnight_20260913/jobs.json`; final scores are available only
after each job's required artifacts and completion checks pass.

**September 9 linked-data run:** [live Table 1 and training status](runs/linked_20260909_overnight/REPORT.md),
[protocol and completed old-checkpoint diagnostics](../../research_notes/0909_overnight_linked_training.md).
This new run uses original MIMIC-CXR + IV clinical inputs, excludes Qwen-Gate annotations,
and runs on free physical GPUs 0/1/5/6/7, never GPU 4. The sections below describe the September 8 pilot.

This implements the future-report experiment in `../../research_notes/0907_paper_plan.md`,
section 3 (updated September 8). It is an initial experiment, not the complete
large-model Table 1. Manuscript result cells remain untouched.

## Environment and local data

`.venv` links to `/home/data2/chk/workspace/2026/.venv`; `data/MIMIC` links to
`/home/data1/data/MIMIC`. Extra FLA and clinical-metric dependencies are installed
under `vendor/`, keeping the shared environment unchanged. All images, reports,
features, predictions, logs and trained weights stay local. Generated data is
ignored by Git and newly generated run files use restricted permissions.
Clinical scoring runs in a subprocess with `metric_vendor/` (Transformers
4.57.1 / tokenizers 0.22.2 / huggingface-hub 0.36.0), because RadGraph 0.1.18
requires the removed `encode_plus` API. Training keeps Transformers 5.12.1.

Base weights: Qwen/Qwen3.5-0.8B revision
`2fc06364715b967f1860aea9cf38778875588b17`, SHA-256
`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
The weight file links to a verified existing local cache. V-JEPA 2.1 ViT-B uses
the local public distilled checkpoint's EMA encoder, loaded strictly.

## Cohort and evidence

- Official patient-disjoint splits: 12,000 training pairs / 6,853 patients;
  645 validation pairs / 121 patients; 1,000 test pairs / 195 patients.
- Adjacent chronological studies, matched AP/PA view, 6h–30d acquisition gap.
  Deterministic hash selection; no selection on target findings or changes.
- Fixed Findings + Impression scope; no unsectioned-report fallback.
- Four requested horizon bins: 6–24h, 1–3d, 3–7d, 7–30d. Exact realized gap
  is retained for audit but excluded from model inputs.
- **CXR does not supply report availability timestamps.** This is a retrospective
  report-available pilot; a minimum acquisition gap does not establish a strict
  prospective report cutoff. No intermediate MIMIC-IV events enter any model.
- Stage-1 train studies only. Report-derived task answers are withheld from
  image-only Stage-1 and replay inputs. Stage-2 source reports are admissible
  current evidence; future images/reports enter targets and losses only.

## Model and initial implementation choices

Frozen V-JEPA 2.1 ViT-B image branch at 384px -> 24×24 spatial grid -> 8×8 average
pooling -> trainable visual adapter -> Qwen text backbone with 8 learned slots.
All 8 contextual slot outputs are retained. Horizon and task requests follow
state construction. The predictor is a 4-layer, width-512 Transformer with a
horizon embedding and residual output. It receives only `(S_t, horizon_bin)`.

The Qwen state encoder and report decoder have distinct rank-8 LoRA adapters.
Pretrained base weights stay frozen. The decoder receives projected future
state slots plus a fixed report request; the disease head reads predicted slots.
There is no raw-image/report bypass to those future readouts. Text training uses
teacher forcing; greedy evaluation has no true future prefix.

Stage 1 (1h): image-only report generation + masked six-finding BCE. Stage 2
(7h): normalized latent MSE against a **fixed Stage-1 multimodal target encoder**,
joint future report CE, future finding BCE, and image-only current finding replay.
Loss weights are 1, 1, 0.5 and 0.1. Freezing the Stage-1 target avoids a moving
target in this first run. This pilot does not implement segmentation, SR or the
full multi-dataset Stage-1 task mixture; no dense-task results are claimed.

Input/target reports are capped at 256 Qwen tokens for training. Copy Current
uses the same current-report token cap. Scoring compares
the generated report (at most 256 tokens) with the full selected report sections.
Truncation and small data/model budgets limit interpretation of pilot results.

## Runs and comparisons

`bash launch.sh pilot_20260908` starts a persistent tmux coordinator:

- GPU 0: MedWorld-JEPA, cumulative 1h Stage 1 + 7h Stage 2.
- GPU 1: native Qwen3.5-0.8B vision-language direct future report/finding training, 8h.
- GPU 2: starts after Stage 1; freezes that encoder and trains the same LWM,
  decoder and finding head using exactly the primary run's Stage-2 minibatches,
  update count, learning rates and losses. It can use less actual compute time.
- GPU 5: Copy Current report evaluation. Final learned-model evaluations use
  each model's training GPU after training finishes. GPU 4 is unhealthy and excluded.

This compares equal wall-time direct-model training and a matched-update latent
control; it does not establish equal FLOPs across model families. The direct
baseline uses its native 256px vision encoder, while JEPA uses the 384px frozen
encoder. Large/gated/API baselines from the paper plan are not run here.

Checkpoint schedule: `checkpoint_stage1.pt` at the stage boundary,
`checkpoint_4h.pt` at approximately 4 cumulative training hours, and
`checkpoint_final.pt` at 8 hours. The matched checkpoint follows the exact
primary Stage-2 save step. Checkpoints contain trainable/target modules,
optimizer state, RNG, elapsed training clock and input/config fingerprints;
frozen upstream Qwen weights are referenced rather than duplicated.
Resume restores identical next-batch forward losses in the short check; BF16
kernel backward arithmetic is not bitwise reproducible across process restarts
(observed maximum next-update parameter difference below 2e-4).
Data preparation, loading, validation and checkpoint I/O are outside the
training clock, so final wall-clock completion is slightly later than 8h.

SIGTERM/SIGINT save `checkpoint_interrupted.pt` at the next optimizer boundary.
Resume a run with the same command plus `--resume /path/to/checkpoint.pt`;
for a matched run preserve `--follow`. Use the archived source/config if code
has subsequently changed. Logs append on resume; step numbers identify retries.

Monitor `runs/<name>/runner_status.json`, each `status.json` / `console.log`, and
`runs/<name>/table1.md`. The runner generates test reports and Table-1 rows after
each training job finishes. It records failures instead of presenting them as
completed scores. No notification is sent outside this workspace.

## Evaluation

CheXbert uses the frozen public `StanfordAIMI/RRG_scorers/chexbert.pth`, its
original 14-head architecture, and the original four-state label mapping.
The six fixed findings are Atelectasis, Cardiomegaly, Consolidation, Edema,
Pleural Effusion and Pneumothorax. Official CheXpert labels supervise training;
the common CheXbert extractor supplies evaluation labels, not proxy scores.

Blank and uncertain reference states are excluded by a reference-only mask.
Unknown predictions never remove an evaluation field. Disease/event classes
with no reference-positive support are marked unavailable and excluded from a
reference-defined denominator shared across models. Per-class support and
coverage are written to `metrics.json`.

- Future R@1: 1 actual + 31 different-patient follow-ups, same horizon/view,
  coarse current-positive count and reference coverage mask. Fixed hash seed;
  mean categorical mismatch, fractional credit for top-score ties. Report the
  included subset size. A 32-way pool may be infeasible in sparse strata.
- Finding AUPRC: macro non-interpolated AP from the trained disease head's
  continuous scores. Copy Current has no score interface and receives `—`.
- Transition F1: macro per-finding onset/resolution events over all evaluable
  pairs, including false changes on stable cases.
- CheXbert F1: macro positive finding F1, not embedding cosine.
- RadGraph F1: official `radgraph==0.1.18`, `radgraph-xl`, RG_ER/partial,
  report-level mean. Dependency/model errors are recorded and leave this cell
  unavailable; no lexical/regex substitute is presented as RadGraph.

References: [Qwen](https://huggingface.co/Qwen/Qwen3.5-0.8B),
[V-JEPA 2.1](https://github.com/facebookresearch/vjepa2),
[CheXbert](https://github.com/stanfordmlgroup/CheXbert),
[RadGraph](https://github.com/Stanford-AIMI/radgraph).

## Reproduction and checks

```bash
.venv/bin/python prepare.py
CUDA_VISIBLE_DEVICES=3 .venv/bin/python features.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest tests -q
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py --config configs/smoke.json --run runs/smoke --smoke-steps 2
CUDA_VISIBLE_DEVICES=2 .venv/bin/python train.py --config configs/smoke.json --run runs/smoke_matched --follow runs/smoke
bash launch.sh pilot_20260908
```

Preparation refuses to overwrite a completed cohort/cache. Tests cover clinical
metric masks/ties/stable false positives, exact chunked-CE gradients, source-only
prediction invariance to target perturbations, and stopped target gradients.
