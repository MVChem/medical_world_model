# MedWorld: three image-conditioned tasks

New runs should use [`configs/medworld_0922.json`](configs/medworld_0922.json).
Its expanded CXR/IV cohort is built under `code/data/medworld_0922` by
[`mimic_atlas/data_processing`](../mimic_atlas/data_processing/README.md), with
all eligible adjacent pairs plus seeded nonadjacent combinations and source-image
symlinks. Classification and temporal data are rebuilt from the full source tables;
segmentation uses the **manual-only `medworld-prepared-v2` protocol**: 200 annotated
MIMIC chest radiographs, 596 UCSF-ALPTDG MRI volumes, 594 MU-Glioma-Post MRI volumes,
and 138 Montgomery images reserved for the external human lung test. New training
and evaluation use no CXAS pseudo masks. The historical adapter remains available
only for reproducing frozen old runs.

The supported downstream tasks are **classification, segmentation and VQA**. Diagnosis is the same disease-label recognition problem and is not counted again. Super-resolution, report generation and temporal task evaluation have been removed from active code. Historical source is retained in Git; obsolete local experiment artifacts were removed in the [September 20 cleanup](https://github.com/MVChem/medical_world_model/blob/e17f2b9d442763b1ac411a519671143a73b9264c/code/CLEANUP_20260920.md).

Every task retains the image as its primary input. VQA also receives the question. The same Qwen visual encoder and shared two-layer Transformer task decoder serve all three tasks:

```text
image → Qwen visual tokens ──→ shared Transformer task decoder → classification linear head
                         ↑                                  → spatial segmentation head
                  optional 8 slots                          → Qwen language decoder → VQA answer
```

`slot_conditioning=false` removes the slots from the task decoder's inputs, leaving a complete image-driven model. `true` adds all eight slots as extra tokens. The recommended segmentation head predicts six channels: `lungs`, `heart`, `NETC`, `SNFH`, `ET`, and `RC`. Each image contributes loss and metrics only for its annotated channels. CXR uses a union lung mask and heart mask; MRI uses the four tumor-region channels; Montgomery uses the union of its two lung masks. VQA generates a JSON answer array. Its autoregressive language decoder is additional to the common Transformer trunk; Qwen3.5's language backbone is a hybrid architecture, not a claim of identical pure-Transformer output heads.

Both arms use identical model initialization, task heads, input data, update counts and loss definitions. Temporal latent prediction/EMA remain **representation-learning objectives**, not downstream tasks. Both arms receive the same temporal auxiliary supervision; optional visual consistency is also retained in both arms, so adding the slot condition is the controlled change. No report/target answer enters a current-task image representation. No pixel or feature caches are written.

## Configuration

Edit the recommended JSON to set model/training parameters and testing. Its manual
segmentation settings include:

```json
{
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

- `enabled=false`: disables test workers for calibration/smoke runs.
- `tasks`: selects evaluation tasks. Completed new experiments must include classification, segmentation and VQA.
- `human_segmentation`: additionally tests Montgomery when segmentation is selected.
- `reuse_completed`: reuses complete outputs only when the final-checkpoint hash, data fingerprint, split, prediction hash and counts match. Partial/incompatible outputs fail rather than being silently overwritten. `false` requires empty selected task output folders.

The recommended config tests all three tasks and external human segmentation. Edit only the `testing` section in a completed run's `config.json` to choose subsequent tests; its trained model and task parameters are restored from the checkpoint. A completed comparison must retain all required evaluations. CLI `--skip-evaluation` remains an explicit launcher override for preflights. `--smoke` performs bounded checks instead of full testing.

## Run

Use the local Python environment with `PYTHONPATH=code`, from the repository root:

```bash
# Three updates, gradient and checkpoint-reload audit; no full test suite.
python -m medworld.train --config code/medworld/configs/medworld_0922.json \
  --smoke --gpu 1 --out code/medworld/runs/smoke_YYYYMMDD

# One model; automatically runs the tests selected in its JSON.
python -m medworld.launch_distributed --config code/medworld/configs/medworld_0922.json \
  --gpus 1,2 --out code/medworld/runs/slots_YYYYMMDD

# Recommended: train/test slots, optionally train/test no-slots, then test native Qwen.
python -m medworld.run_experiment --config code/medworld/configs/medworld_0922.json \
  --gpus 1,2 --out code/medworld/runs/paired_YYYYMMDD

# Run or reuse the tests currently selected in a completed run's JSON.
python -m medworld.evaluate_run --run code/medworld/runs/slots_YYYYMMDD --gpus 1,2

python -m unittest discover -s code/medworld/tests -v
```

Use `run_experiment` for the complete comparison. `baselines.no_slots` and `baselines.qwen` independently enable the two baselines (both default to true). Formal training/testing runs slots first, then the enabled no-slots arm, then native Qwen matching the configured pretrained backbone (0.8B, 4B or 9B). Qwen receives no project training and tests only selected classification/VQA tasks; it cannot perform segmentation. All evaluated arms use the same source test cohort and VQA selection. The no-slots arm keeps the shared task decoder and auxiliary objectives, removing the eight slots only from the decoder's inputs.

With both trained arms enabled, `total_hours=3` is an approximate **combined training budget**, not 1.5 hours per arm. The runner first measures nine updates per arm, discards those probe weights, and estimates one common update count from their measured speeds. The estimate subtracts calibration time, reserves 10% for overhead, and rounds down to a complete three-task cycle. Both formal arms restart from the same pretrained initialization and must finish **exactly that many optimizer updates**, with identical per-rank batches, accumulation and GPU count. A slower model receives more time; clock estimates never truncate either arm. Startup, validation and throughput variation can change the actual duration. Evaluation is additional time. `budget_plan.json` records the estimate; `training_budget.json` records actual updates and training durations.

Set `total_hours=0` to train both arms for the explicit `steps` count without timing calibration. With the no-slots baseline disabled, a timed run uses the whole budget for the slots model. The low-level `launch_distributed` entry still trains one model only and interprets `total_hours` as that model's budget; baseline switches are handled only by `run_experiment`.

Outputs remain under the experiment: `slots/`, optional `baseline/` and `qwen/`, task predictions/metrics and `evaluation_summary.json`, plus the combined `COMPARISON.md`/`comparison.json`. Comparisons reject unequal optimizer steps, mismatched data/initial weights, or different test IDs/references. Turning `testing.enabled` off skips all held-out tests and native Qwen, while still training the enabled trainable arms. Reports explicitly record skipped baselines. Selecting a task subset compares only that subset.

## Two-GPU throughput configuration

The supplied config uses a 12-hour combined training estimate. Install the optional
packages in `requirements-acceleration.txt` into the existing PyTorch environment
with `pip install --no-deps --no-build-isolation -r code/medworld/requirements-acceleration.txt`.
For the local CUDA 12.0 compiler, building causal-conv1d requires `CC=gcc-12 CXX=g++-12 MAX_JOBS=4`.
Do not replace the installed PyTorch/CUDA stack. The launcher records dependency versions;
`require_fast_kernels=true` prevents silent Qwen fallback.

`batched_vision_attention` batches equal-length images in one SDPA call instead of
calling attention separately for every image; unequal lengths retain the upstream
implementation. It preserves parameter names and image isolation. `prefetch_preprocessing`
moves the same Qwen/JEPA image transforms into the bounded RAM prefetch queue; it writes
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
is deterministic from seed and offset, including after resume. MRI decoding uses
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

The manual configuration's six-channel segmentation head cannot directly resume
an old three-channel checkpoint. Reproducing historical weights requires their
original configuration and data protocol. The code reads local Qwen and V-JEPA
weights; no external services are required.

## VQA test sampling

`testing.vqa_per_type` selects this many questions from each of Verify, Choose and Query, using `testing.vqa_seed`. Zero retains the complete test set. The provided two-GPU training config now selects 100 per type with seed 42. Training data is unchanged. The native-model baseline uses the same selection helper; both save exact IDs and their hash. Balanced-subset metrics are distinct from full-test metrics.

## Local assets

The prepared manifests are accessed through `code/data/medworld_0922`. Original
images, manual masks and other dataset assets under `code/data/` use symbolic
links to the central data store. Preparation records their actual source paths
and fingerprints; generated pixel or feature caches are not written.

Local model weights remain at the paths specified in the training JSON.
Historical configs are translated by `asset_paths.py`; unchanged historical
manifests keep their provenance keys for reproduction.

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

## Historical configurations and Qwen3.5-9B run

`configs/qwen35_08b_vssc_2gpu.json` and
`configs/qwen35_9b_vssc_2gpu_48h.json` preserve the historical three-channel,
CXAS-based protocol. They are for reproducing old runs and are not the recommended
configuration for new experiments. Historical experiment records retain their
original protocol and results.

The historical 9B configuration used a 48-hour combined training estimate, equal
optimizer updates for slots/no-slots, smaller batches for 24-GiB GPUs, and the
matched 300-question VQA subset. Evaluation time was additional. A new 9B manual
experiment must start from the manual v2 configuration and choose suitable model
and batch settings; it cannot resume the old three-channel head directly.
