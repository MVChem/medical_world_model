# MedWorld: raw inputs with optional slot conditions

New runs should use [`configs/medworld_0923.json`](configs/medworld_0923.json).
Temporal representation learning reads the new paired CXR selection at
`code/data/medworld_0923`, prepared by
[`data_preprocessing`](../data_preprocessing/README.md). Each pair supplies its
source image/report, target image/report, and actual time interval. Medication
evidence helped select this cohort and remains available in Atlas for inspection;
it is not loaded or supplied to the model. The world model receives source slots
and the signed time interval, with the other endpoint used only as a latent target.

Classification, VQA, and reviewed segmentation continue to read
`code/data/medworld_0922`, built by
[`mimic_atlas/data_processing`](../mimic_atlas/data_processing/README.md).
Segmentation uses the **manual-only `medworld-prepared-v2` protocol**: 200 annotated
MIMIC chest radiographs, 596 UCSF-ALPTDG MRI volumes, 594 MU-Glioma-Post MRI volumes,
and 138 Montgomery images reserved for the external human lung test. New training
and evaluation use no CXAS pseudo masks.

The supported downstream tasks are **classification, segmentation and VQA**. Diagnosis is the same disease-label recognition problem and is not counted again.

The architecture is `raw_input_v1`. The baseline passes raw inputs directly
to its task decoder. The slots model passes the same inputs plus eight slots as
additional conditions. Both arms share the same task decoder design: learned image
patch projection, UTF-8 byte embeddings for optional input reports, and a two-layer
Transformer. Image processing belongs to this decoder and does not use the slot
branch's Qwen visual features.

```text
baseline: image / report ───────────────→ Task Decoder → classification / segmentation / VQA
slots:    image / report ───────────────→ Task Decoder → classification / segmentation / VQA
               image → Qwen/JEPA → 8 slots ────↑
```

`slot_conditioning=false` constructs no slot encoder, Qwen backbone, JEPA,
world model, VSSC teacher/reconstruction head, or EMA target. Its loss contains
only the supervised downstream task losses. `true` adds the slot branch and its
configured auxiliary losses while retaining the raw inputs at the task decoder.
The raw architecture's VQA head is a small autoregressive byte decoder trained
from scratch in both arms; it does not load a Qwen language decoder.

The task decoder accepts images, reports, or both. Spatial segmentation requires
images. The slots arm also requires images because its state encoder constructs
visual slots; report-only slots inference is rejected. The current training and
evaluation datasets provide images, with questions additionally supplied to VQA.
They do **not** automatically supply the source reports used to derive classification
labels. Report-input experiments require an explicitly defined input and supervision
protocol to avoid exposing targets through the report.

Both arms start from identical downstream weights, recorded by
`task_initialization_sha256`, and use the same task data and optimizer update count.
Only the slots arm receives temporal/EMA and optional VSSC supervision. This
comparison measures the full slot branch and its training objectives together;
separate ablations are needed to distinguish the contribution of each objective.
The baseline's size is independent of the Qwen model used to construct slots:
there is no separate "0.8B baseline" or "9B baseline" in this architecture.
No pixel or feature caches are written to disk.

Previous 0.8B/9B "no-slot-input" results used a different model with Qwen visual
features and auxiliary losses. They do not measure this baseline. Saved results
remain in their run directories; this implementation does not load previous
architectures or data protocols. New formal training results are still required.

The recommended segmentation head predicts six channels: `lungs`, `heart`, `NETC`,
`SNFH`, `ET`, and `RC`. Each image contributes loss and metrics only for its annotated
channels. CXR uses a union lung mask and heart mask; MRI uses four tumor-region
channels; Montgomery uses the union of its two lung masks. VQA generates a JSON
answer array.

## Configuration

Edit the recommended JSON to set model/training parameters and testing. Its manual
segmentation settings include:

```json
{
  "architecture": "raw_input_v1",
  "slot_conditioning": true,
  "segmentation_channels": 6,
  "segmentation_sampling": "balanced_dataset",
  "observation_prompt": "Medical image observation.",
  "total_hours": 3,
  "baselines": {
    "no_slots": true,
    "qwen": true
  },
  "testing": {
    "enabled": true,
    "tasks": ["classification", "segmentation", "vqa"],
    "human_segmentation": true,
    "reuse_completed": true
  }
}
```

- `architecture`: `raw_input_v1` identifies the supported model and checkpoint
  format.
- `task_patch_size` defaults to 16; `vision_pixels` must be divisible by it.
  Raw images are resized with their aspect ratio preserved and centered on a
  black square before learned patch projection.
- In the raw task/VQA decoders, `answer_tokens`, `context_tokens`, and
  `generation_tokens` bound UTF-8 bytes (plus decoder special tokens), rather
  than Qwen tokenizer tokens. Their former numeric values do not imply the same
  text lengths. The slots branch still applies `context_tokens` through its
  Qwen tokenizer when encoding observation prompts/reports.
- The paired runner sets the raw baseline's temporal and VSSC loss weights to
  zero. Its training pipeline does not fetch temporal batches or update EMA.
- `enabled=false`: disables test workers for calibration/smoke runs.
- `tasks`: selects evaluation tasks. Completed new experiments must include classification, segmentation and VQA.
- `human_segmentation`: additionally tests Montgomery when segmentation is selected.
- `reuse_completed`: reuses complete outputs only when the final-checkpoint hash, data fingerprint, split, prediction hash and counts match. Partial/incompatible outputs fail rather than being silently overwritten. `false` requires empty selected task output folders.

The recommended config tests all three tasks and external human segmentation. Edit only the `testing` section in a completed run's `config.json` to choose subsequent tests; its trained model and task parameters are restored from the checkpoint. A completed comparison must retain all required evaluations. CLI `--skip-evaluation` remains an explicit launcher override for preflights. `--smoke` performs bounded checks instead of full testing.

## Run

Use the local Python environment with `PYTHONPATH=code`, from the repository root:

```bash
# Three updates, gradient and checkpoint-reload audit; no full test suite.
python -m medworld.train --config code/medworld/configs/medworld_0923.json \
  --smoke --gpu 1 --out code/medworld/runs/smoke_YYYYMMDD

# One model; automatically runs the tests selected in its JSON.
python -m medworld.launch_distributed --config code/medworld/configs/medworld_0923.json \
  --gpus 1,2 --out code/medworld/runs/slots_YYYYMMDD

# Recommended: train/test slots, optionally train/test no-slots, then test native Qwen.
python -m medworld.run_experiment --config code/medworld/configs/medworld_0923.json \
  --gpus 1,2 --out code/medworld/runs/paired_YYYYMMDD

# Run or reuse the tests currently selected in a completed run's JSON.
python -m medworld.evaluate_run --run code/medworld/runs/slots_YYYYMMDD --gpus 1,2

python -m unittest discover -s code/medworld/tests -v
```

`medworld.infer` accepts `--image`, `--report`, or
both. VQA also requires `--question`; segmentation requires `--image`.
Report-only input is available for the baseline, while slots checkpoints require
an image. For example:

```bash
python -m medworld.infer --checkpoint code/medworld/runs/baseline_YYYYMMDD/final.pt \
  --task classification --report "Supplied observation text." \
  --gpu 1 --out code/medworld/runs/prediction_YYYYMMDD.json
```

Use `run_experiment` for the complete comparison. `baselines.no_slots` and
`baselines.qwen` independently enable the raw-input baseline and native Qwen
(both default to true). Formal training/testing runs slots first, then the enabled
raw-input baseline, then native Qwen matching the configured slot backbone
(0.8B, 4B or 9B). Native Qwen receives no project training and tests only selected
classification/VQA tasks; it cannot perform segmentation. All evaluated arms use
the same source test cohort and VQA selection.

With both trained arms enabled, `total_hours=3` is an approximate **combined training budget**, not 1.5 hours per arm. The runner first measures nine updates per arm, discards those probe weights, and estimates one common update count from their measured speeds. The estimate subtracts calibration time, reserves 10% for overhead, and rounds down to a complete three-task cycle. Both formal arms restart with identical downstream initialization and must finish **exactly that many optimizer updates**, with identical per-rank batches, accumulation and GPU count. A slower model receives more time; clock estimates never truncate either arm. Startup, validation and throughput variation can change the actual duration. Evaluation is additional time. `budget_plan.json` records the estimate; `training_budget.json` records actual updates and training durations.

Set `total_hours=0` to train both arms for the explicit `steps` count without timing calibration. With the no-slots baseline disabled, a timed run uses the whole budget for the slots model. The low-level `launch_distributed` entry still trains one model only and interprets `total_hours` as that model's budget; baseline switches are handled only by `run_experiment`.

Outputs remain under the experiment: `slots/`, optional `baseline/` and `qwen/`, task predictions/metrics and `evaluation_summary.json`, plus the combined `COMPARISON.md`/`comparison.json`. Comparisons reject unequal optimizer steps, mismatched data/initial weights, or different test IDs/references. Turning `testing.enabled` off skips all held-out tests and native Qwen, while still training the enabled trainable arms. Reports explicitly record skipped baselines. Selecting a task subset compares only that subset.

## Saved checkpoints

All new `last.pt`, `best.pt` (distributed validation), and `final.pt` files use
format 5 and contain trained weights only: LoRA, learned adapters and slot
readouts, the world predictor, the task decoder and task heads, and the VSSC
reconstruction module when enabled. The learned EMA target encoder is retained
as copies of its adaptable parameters and small buffers. Frozen Qwen/V-JEPA
weights and the frozen VSSC teacher are reconstructed from the original local
assets rather than copied into each checkpoint.

Each file also retains the classification positive-class weights, configuration,
model metadata, source/data fingerprints, update count, completion flag, GPU
world size, and downstream sample counts for evaluation and matched comparisons.
Optimizer moments, random-generator state, sampling offsets, and training clocks
are excluded. These files support inference and evaluation; exact training resume
is unavailable, and all training launchers reject `--resume` before acquiring GPUs.
Logs, launch commands, source snapshots, and metrics remain in the run directory.

With `configs/medworld_0923.json` (rank 8, FP32 trainable tensors, predictor
512/4, task decoder 256/2, VSSC weight 0.1), estimated sizes per checkpoint are:

| Arm | Online trained weights | Saved weights including EMA |
| --- | ---: | ---: |
| Raw-input baseline | 17.501 MiB | 17.501 MiB |
| Qwen3.5-0.8B + slots | 112.712 MiB | 154.294 MiB |
| Qwen3.5-9B + slots | 251.007 MiB | 430.884 MiB |

These are tensor payload counts from the actual module constructors on the
PyTorch `meta` device, with no GPU allocation or pretrained tensor loading.
Serialization and metadata add a small amount. The 9B estimate changes only the
Qwen backbone configuration; it does not represent a completed 9B training run.
Keeping all three 9B checkpoint files uses about 1.26 GiB in total. LoRA alone is
only 4.594 MiB for 9B but omits the other learned modules needed by this model.
The reproducible script and per-module counts are retained in
[`runs/weights_only_checkpoint_20260923`](runs/weights_only_checkpoint_20260923/README.md).

## Two-GPU throughput configuration

The supplied config uses a 12-hour combined training estimate. Raw-input decoding
changes memory use and throughput in both arms, especially VQA. Recalibrate batch
sizes and the common update budget before a formal run; historical Qwen-feature
throughput measurements do not validate this architecture. Install the optional
packages in `requirements-acceleration.txt` into the existing PyTorch environment
with `pip install --no-deps --no-build-isolation -r code/medworld/requirements-acceleration.txt`.
For the local CUDA 12.0 compiler, building causal-conv1d requires `CC=gcc-12 CXX=g++-12 MAX_JOBS=4`.
Do not replace the installed PyTorch/CUDA stack. The launcher records dependency versions;
`require_fast_kernels=true` prevents silent Qwen fallback in the slots branch;
the raw-input baseline does not require these Qwen kernels.

`batched_vision_attention` batches equal-length images in one SDPA call instead of
calling attention separately for every image; unequal lengths retain the upstream
implementation. It preserves parameter names and image isolation. `prefetch_preprocessing`
moves raw-input and, when used, Qwen/JEPA image transforms into the bounded RAM prefetch queue; it writes
no intermediate image or feature caches. `cpu_threads` controls PyTorch intra-op threads,
while `cpu_cores_per_rank` reserves an affinity slice for image workers independently.
For GPUs 2/3 the configured two 16-core slices remain on their local NUMA node.
Batch sizes are per rank. Compare samples/second and warm GPU telemetry, not memory
occupancy alone. Checkpoints and validation still pause training periodically.

`decoded_image_cache` bounds each of three process-local LRU caches of immutable
decoded source pixels (current-task grayscale canvases, VQA RGB images, temporal
RGB images). It caches no learned features and writes nothing to disk. At 16,384
entries per cache, the two training ranks use at most about 56 GiB for cached pixel
payloads, plus Python overhead; the local machine has 251 GiB RAM. This amortizes
JPEG decoding as training revisits images. Set it to zero on smaller hosts.

`segmentation_sampling="balanced_dataset"` gives equal training exposure to the
available segmentation datasets. Within each MRI dataset, seeded volume blocks
keep slices from one volume together, with shuffled volume order and shuffled
slices inside each volume. This reduces repeated gzip decompression. The stream
is deterministic from the seed and stream offset. MRI decoding uses
a separate bounded RAM cache; no decoded volumes are stored on disk.

## Data and metrics

- Classification: the existing 13 CheXpert-derived labels (12 disease/finding labels plus Support Devices); masked uncertain/missing labels, AUROC/AP. These report-derived labels are not independent clinical diagnostic ground truth.
- Segmentation: human-reviewed MIMIC lung/heart masks and UCSF-ALPTDG/MU-Glioma-Post tumor masks; Montgomery human lung masks are an external test. Report IoU and Dice separately for each dataset.
- VQA: official local MIMIC-CXR-VQA image/question pairs; full test has 13,793 questions. Strict JSON label-set exact match and micro-F1, with Verify/Choose/Query breakdown. Invalid responses count as errors; this local scorer is not claimed to be the official evaluator.

Global patient holdouts cover all tasks and temporal representation learning. Higher-priority held-out membership removes conflicting training/validation rows; it never moves rows into evaluation. The public 110-answer vocabulary is stored in `datasets/vqa_vocabulary.json`, copied from the previously frozen official `ans2idx.json` metadata, not inferred from test answers.

MRI currently supplies **2D axial T1ce slices** after canonical RAS orientation and
image-derived intensity normalization. T1, T2 and FLAIR paths remain in the source
inventory; these three sequences are not inputs to the current training adapter.
The 596 and 594 counts describe annotated volumes, not independent slice
annotations or unique patients. Repeated visits remain in the same patient split.

The slots branch reads local Qwen and V-JEPA weights; the raw-input baseline
needs neither. No external services are required.

## VQA test sampling

`testing.vqa_per_type` selects this many questions from each of Verify, Choose and Query, using `testing.vqa_seed`. Zero retains the complete test set. The provided two-GPU training config now selects 100 per type with seed 42. Training data is unchanged. The native-model baseline uses the same selection helper; both save exact IDs and their hash. Balanced-subset metrics are distinct from full-test metrics.

## Local assets

The current temporal selection is accessed through `code/data/medworld_0923`;
reviewed downstream task manifests remain at `code/data/medworld_0922`. Original
images, manual masks and other dataset assets under `code/data/` use symbolic
links to the central data store. Preparation records their actual source paths
and fingerprints; generated pixel or feature caches are not written.

Local model weights remain at the paths specified in the training JSON.

### Required segmentation evaluation

Every completed new experiment must evaluate classification/VQA and segmentation
for both trained arms. Segmentation `test` reports the held-out MIMIC human CXR,
UCSF-ALPTDG and MU-Glioma-Post datasets separately; `human_test` evaluates the 138
external Montgomery images. CXAS pseudo masks are excluded from both training
and evaluation.

Predictions use sigmoid > 0.5 and targets > 0.5 inside each annotated channel's
valid ROI. MRI evaluation sums intersection, predicted-positive and target-positive
counts across the selected axial slices of each volume on their **256 × 256
evaluation grids**, then computes that volume's IoU and Dice. Scores average
equally over annotated channels and volumes within each dataset; CXR uses each
image as its evaluation unit. Report per-dataset `mean_iou`, `mean_dice`, and
per-channel scores. Summary scores are an equal-weight macro average over the
datasets included in that evaluation split. Two empty masks score 1. Native Qwen
has no segmentation head and is N/A.
