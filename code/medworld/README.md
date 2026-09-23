# MedWorld: raw inputs with optional slots

The current Qwen9B experiment covers both paper tables. Use
[`qwen35_9b_tables12_2gpu_24h_0923.json`](configs/qwen35_9b_tables12_2gpu_24h_0923.json).
The comparison includes a raw-input baseline, native Qwen3.5-9B without project
training, and MedWorld with Qwen3.5-9B-derived slots. Other backbones are outside
this experiment. The fixed definitions are in
[`EVALUATION.md`](EVALUATION.md); earlier metric formats are not accepted.

## Architecture and supervision

The baseline sends original images and optional source reports directly to its
task decoder. It constructs no Qwen, V-JEPA, slot encoder, world predictor, EMA
target, or VSSC branch. Its learned image patch projection, report byte embeddings,
Transformer, and task heads are trained from scratch.

MedWorld gives the same raw inputs to the same decoder and supplies eight slots
as additional conditions. Current tasks use source slots. Future tasks use the
world predictor's forecast slots, conditioned on source slots and the requested
time interval. Only this arm uses temporal latent prediction and VSSC losses.
Both trained arms start from identical task/readout weights and see matching
sample counts and optimizer updates. The baseline's architecture is independent
of the size of the Qwen used by the slots arm.

```text
Raw baseline: raw image / optional report ──→ Task Decoder ──→ task head
MedWorld:     raw image / optional report ──→ Task Decoder ──→ task head
              image / report → Qwen + JEPA → slots ───────────↑
              future tasks: slots → World(source, horizon) ──↑
```

The supervised cycle contains eight tasks:

| Table | Tasks | Inputs |
| --- | --- | --- |
| 1 | Future VQA, progression, future report, 30-day mortality, remaining LOS | Source image/report and prescribed request; question where applicable |
| 2 | Current classification, segmentation, VQA | Image; question for VQA; same-exam report withheld |

Each update selects one supervised task. The slots arm additionally receives a
temporal pair batch with weight 1 and an EMA target (momentum 0.99); VSSC has
weight 0.1 on segmentation updates. The baseline's auxiliary weights are zero.
Medications are not inputs to either arm. Future labels, images, reports, death
records, and discharge times are unavailable to prediction functions.

The report/VQA generators are independent byte Transformers; they do not use a
Qwen language decoder. Future source reports are limited to the same 383 UTF-8
bytes for every arm. Future reports have a 2,047-byte output budget plus EOS;
references remain the original full reports. These are byte limits, not word or
Qwen-token limits. Changing them requires a new matched protocol and run.

## Data

Original datasets stay outside Git. `code/data` exposes symbolic links:

- `medworld_0922`: current classification/VQA and human-reviewed segmentation.
- `medworld_0923`: medication-positive chronological CXR pairs for temporal
  representation learning. The model reads image/report references and elapsed
  hours, without medication evidence or dose inputs.
- `medworld_0923/table12_v1`: fixed future-task train/validation/test labels and
  original-asset references, built by
  [`data_preprocessing.build_table12`](../data_preprocessing/build_table12.py).
- `medworld_evaluation/radgraph-xl`: pinned official report evaluator assets.

Patient holdouts apply across every task. Progression uses explicitly
source-anchored silver comparisons for training/validation and human Chest
ImaGenome gold comparisons for test. Outcome cohorts include eligible source
examinations without later chest images and do not require medication records.
Prepared manifests retain support, exclusions, source hashes, and Atlas links.

Segmentation uses only the 200-image MIMIC heart/lung collection, UCSF-ALPTDG
and MU-Glioma-Post MRI, plus external Montgomery lung masks. No CXAS pseudo masks
are used. MRI slices are aggregated by volume for Dice and IoU. The six output
channels are lungs, heart, NETC, SNFH, ET, and RC; unannotated channels and pixels
are masked. Native Qwen segmentation is N/A.

## Train and evaluate

Run from the repository root in the existing environment:

```sh
PYTHONPATH=code HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 code/.venv/bin/python \
  -m medworld.run_experiment \
  --config code/medworld/configs/qwen35_9b_tables12_2gpu_24h_0923.json \
  --gpus 1,2 --out code/medworld/runs/paired_tables12_YYYYMMDD

# Reuse or complete configured tests for a successfully trained arm.
PYTHONPATH=code code/.venv/bin/python -m medworld.evaluate_run \
  --run code/medworld/runs/paired_tables12_YYYYMMDD/slots --gpus 1,2

PYTHONPATH=code CUDA_VISIBLE_DEVICES='' code/.venv/bin/python -m pytest \
  code/medworld/tests code/medworld_zero_shot_eval/tests -q
```

The paired runner freezes source and configuration before any training. For a
24-hour combined training budget, it calibrates 24 updates per arm (three full
task cycles), discards those probe weights, and estimates one common number of
updates rounded to eight. Formal arms restart from the same initialization.
Both trained arms finish before full table evaluation starts. No timed kill
truncates one arm to fit an estimate. Startup, validation, checkpointing, and
throughput variation affect actual duration; full evaluation takes extra time.
`budget_plan.json` records the estimate and `training_budget.json` actual work.
Set `total_hours=0` for an explicit equal `steps` count.

After training, each arm runs all five future tests, current classification/VQA,
internal reviewed segmentation, and external Montgomery. Native Qwen then runs
both its current and future tests. The exporter requires every applicable cell
to have a complete finite result on identical references. It produces
`TABLE1.md`, `TABLE2.md`, `tables.json`, and `COMPARISON.md` in the original run.
No table is declared complete while an applicable task remains untested.

Use `testing.enabled=false` or launcher `--skip-evaluation` only for calibration
and smoke runs. The older three-task `--smoke` shortcut is not the complete
Table 1/2 preflight. Native Qwen does not undergo project training.

## Saved checkpoints

`last.pt`, `best.pt`, and `final.pt` contain trained parameters only: LoRA,
learned adapters/readouts, task decoder and all enabled heads, world predictor,
VSSC reconstruction when enabled, and compact learned EMA encoder state.
Frozen Qwen/V-JEPA/teacher weights are reconstructed from the original local
assets and checked against hashes. No optimizer moments, RNG states, sampler
positions, or training clocks are stored. Exact training resume is unavailable;
launchers reject `--resume` before acquiring GPUs.

Configuration, source/data fingerprints, task initialization hash, completion
state, update count, world size, and per-task sample counts accompany the
weights for evaluation. `best.pt` uses validation losses; final comparisons use
`final.pt` without test-based checkpoint selection. With the current configuration, real GPU save/reload checks measured **443.46 MiB**
per 9B slots checkpoint (including compact EMA) and **30.00 MiB** per raw baseline
checkpoint. Both include all five future heads. Serialization metadata causes
small run-to-run size differences; see the
[preflight inventory](runs/table12_preflight_20260923/trained_preflight_summary.json).

Commands, source snapshots, logs, metrics, and predictions remain in dated
`runs/` folders. All checkpoint files are excluded from Git.

## Runtime

Preserve the installed Torch/CUDA stack. Optional Qwen acceleration requirements
are in `requirements-acceleration.txt`; `require_fast_kernels=true` rejects a
silent slow fallback for slots. The raw baseline needs none of these Qwen
kernels. Official RadGraph uses a separate CPU worker with pinned Transformers 4
dependencies, leaving the Qwen training environment on Transformers 5.
See the [evaluator preflight record](runs/table12_preflight_20260923/RADGRAPH.md).

Image transforms are computed on demand with bounded RAM prefetch and decoded
image caches. No pixel or feature caches are written to disk.
`segmentation_sampling=balanced_dataset` gives equal training exposure to the
three internal datasets, with deterministic MRI volume blocks. Batch sizes are
per GPU; the current config uses four examples per GPU and four accumulation
steps, giving 32 examples per supervised update on two GPUs.
