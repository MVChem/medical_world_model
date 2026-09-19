# MedWorld

The shared Medical World Encoder produces an 8×1024 state: four fusion slots and four visual slots.
Classification and report generation read the full state; segmentation and super-resolution read the input image and four visual slots.
Stage 1 alternates between the four current-observation tasks. Stage 2 trains temporal prediction with replay of the current tasks.

## Code structure

```text
medworld/
├── datasets/
│   ├── current.py         # Current-task samples, supervision, and collation
│   ├── temporal.py        # Temporal image pairs and source/target inputs
│   ├── pixels.py          # Online image/mask loading and 4× LR generation
│   ├── protocol.py        # Manifests, hashes, and patient holdouts
│   └── unified.py         # Cross-task splits, deterministic sampling, and batches
├── downstream_tasks/
│   ├── classification/    # decoder.py / loss.py / metrics.py
│   ├── report/            # decoder.py (including teacher-forced loss) / metrics.py
│   ├── segmentation/      # decoder.py / loss.py / metrics.py
│   ├── super_resolution/  # decoder.py / loss.py / metrics.py
│   ├── common/            # Shared components and spatial decoder architecture
│   ├── registry.py        # Task identifiers and label definitions
│   └── training.py        # Stage 1 and replay loss dispatch
├── representation/        # Optional visual slot spatial consistency objective
├── evaluation/            # Evaluation, CheXbert, baseline audits, and comparisons
├── third_party/vjepa2/    # Self-contained V-JEPA 2.1 inference subset and licenses
├── encoder.py / model.py  # Encoder and model assembly
├── runtime.py             # Checkpoints and single-GPU training
├── distributed_train.py   # DDP training
├── launch_distributed.py  # GPU locks, source snapshots, and monitoring
└── run_experiment.py      # Training followed by evaluation and comparison
```

Segmentation and super-resolution use the standard SpatialHead architecture with independent parameters.
Optional training-time representation objectives live in [representation/](representation/README.md).
They are disabled by default and do not change task decoders. The directory refactor preserves the eight-slot definition and each task's input selection.

Historical run artifacts were removed at the user's request on 2026-09-19. New runs store artifacts and source snapshots under `runs/`. Checkpoints require their matching architecture; the current standard decoder cannot restore a historical specialized spatial decoder.
`evaluate.py` and `decoders.py` retain compatibility entry points.

## Data and weights

All data loading lives in [datasets/](datasets/README.md). Paths can be overridden in the training JSON configuration.
Defaults point to existing local data and weights. Loading does not duplicate raw data or create disk caches of pixels or features.
Runtime code does not import sibling projects. External reads are limited to data manifests, source images and labels, model weights, and explicitly selected baseline results.

MIMIC-CXR supplies images, reports, and finding labels. Historical temporal pairs are linked to MIMIC-IV admissions.
**This version does not input MIMIC-IV laboratory results, medications, or vital signs; `ehr=false`.** Human segmentation evaluation uses Montgomery.
Segmentation supervision still reads fixed CXAS labels from `seg_probs.npy`; these are target labels, not cached input features.
Super-resolution uses a fixed 128→512 (4×) protocol, with HR images used only as targets.

## Running

Run commands from the repository root. The Python environment requires PyTorch, transformers, PEFT, numpy, Pillow, scikit-learn, scikit-image, timm, and einops.
All model weights are loaded locally.

```bash
PYTHONPATH=code python -m medworld.train --smoke --gpu auto --out code/medworld/runs/smoke_YYYYMMDD
PYTHONPATH=code python -m medworld.evaluation.evaluate --checkpoint code/medworld/runs/RUN/stage1.pt --task current --split test --gpu auto --out code/medworld/runs/RUN/evaluation
PYTHONPATH=code python -m unittest discover -s code/medworld/tests -v
```

The single maintained training configuration is [configs/qwen35_08b_vssc_4gpu.json](configs/qwen35_08b_vssc_4gpu.json).
It enables VSSC and BF16 for an eight-hour run: approximately four hours per stage.
The smoke command above uses the default configuration with the auxiliary objective disabled.

```bash
PYTHONPATH=code python -m medworld.launch_distributed \
  --config code/medworld/configs/qwen35_08b_vssc_4gpu.json \
  --gpus 0,1,2,3 \
  --out code/medworld/runs/qwen35_08b_vssc_4gpu_YYYYMMDD
```

GPU selection is a launcher argument, not a JSON setting. The launcher sets `CUDA_VISIBLE_DEVICES`.
Per-GPU batches are classification 96; report, segmentation, SR, and temporal 64; replay 16.
These are initial batch sizes for the new architecture, not a completed four-GPU capacity measurement.
`total_hours` and `stage1_hours` set wall-clock budgets; otherwise, training uses step budgets.
Per-task batch sizes are configurable, with separate, smaller batch sizes available for Stage 2 replay.
Check capacity with the intended configuration before a long run; measurements from an older architecture do not establish capacity for the current one.

`medworld.run_experiment` runs training followed by automatic evaluation; see `--help` for arguments.
Native-model comparisons require an audit through `medworld.evaluation.baseline_audit`.
Reports are scored through `medworld.evaluation.clinical_report`. Full evaluation takes additional time after training.
The `evaluation/` package summarizes classification AUROC/AP, report CheXbert F1, segmentation Dice, super-resolution PSNR/SSIM, and temporal prediction metrics.
A native-Qwen baseline with a trained segmentation head has not yet been implemented.
VQA, a separate diagnosis task, and grounding are not yet part of this training pipeline.
