# MedWorld

The shared Medical World Encoder produces an 8×1024 state: four fusion slots and four visual slots.
Classification and report generation read the full state; segmentation and super-resolution read the input image and four visual slots.
Training uses one continuous joint objective from the first update. Each optimizer update combines a temporal batch with one current-task batch, cycling through classification, report, segmentation, and SR. The loss is current-task loss + latent loss + report loss + 0.5 × finding loss. VSSC, when enabled, is added to spatial current-task losses.
The online encoder, predictor, and active task heads receive gradients together. The target encoder is copied at model initialization, receives no gradients, and follows EMA (momentum 0.99) after every optimizer update. There is no stage transition, optimizer reset, replay schedule, or temporal warmup.

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
│   └── training.py        # Current-task loss dispatch
├── representation/        # Optional visual slot spatial consistency objective
├── evaluation/            # Evaluation, CheXbert, baseline audits, and comparisons
├── third_party/vjepa2/    # Self-contained V-JEPA 2.1 inference subset and licenses
├── encoder.py / model.py  # Encoder and model assembly
├── batching.py            # Shared deterministic current + temporal batches
├── runtime.py             # Checkpoints and single-GPU joint training
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
PYTHONPATH=code python -m medworld.evaluation.evaluate --checkpoint code/medworld/runs/RUN/final.pt --task current --split test --gpu auto --out code/medworld/runs/RUN/evaluation
PYTHONPATH=code python -m unittest discover -s code/medworld/tests -v
```

The single maintained training configuration is [configs/qwen35_08b_vssc_4gpu.json](configs/qwen35_08b_vssc_4gpu.json).
It enables VSSC and BF16 for an eight-hour run: a single continuous training budget.
The smoke command above uses the default configuration with the auxiliary objective disabled.

```bash
PYTHONPATH=code python -m medworld.launch_distributed \
  --config code/medworld/configs/qwen35_08b_vssc_4gpu.json \
  --gpus 1,2,3,6 \
  --out code/medworld/runs/qwen35_08b_vssc_4gpu_YYYYMMDD
```

GPU selection is a launcher argument, not a JSON setting. The launcher sets `CUDA_VISIBLE_DEVICES`.
Per-GPU batches are 16 for each current task and 32 for temporal prediction, with accumulation=1.
Each update uses both batches; the four-GPU global sizes are 64 current observations and 128 directed temporal pairs.
These conservative initial sizes have not been measured with the joint architecture. Check capacity before a long run.
`total_hours` sets the distributed wall-clock budget; when zero, `steps` sets the optimizer-update budget. `accumulation` applies to both batches, and `task_batch_sizes` sets their per-rank sizes for either trainer.
Checkpoints are `last.pt` (resume), `final.pt` (completed training), and `best.pt` (distributed validation selection).
The automatic pipeline evaluates `final.pt` once for current tasks and temporal prediction. Both use the same trained online encoder; current tasks bypass the predictor.
Checkpoint format 2 requires EMA state from initialization. Historical two-stage checkpoints/configurations are intentionally rejected; their frozen run source remains available for historical replay.

`medworld.run_experiment` runs training followed by automatic evaluation; see `--help` for arguments.
Native-model comparisons require an audit through `medworld.evaluation.baseline_audit`.
Reports are scored through `medworld.evaluation.clinical_report`. Full evaluation takes additional time after training.
The `evaluation/` package summarizes classification AUROC/AP, report CheXbert F1, segmentation Dice, super-resolution PSNR/SSIM, and temporal prediction metrics.
A native-Qwen baseline with a trained segmentation head has not yet been implemented.
VQA, a separate diagnosis task, and grounding are not yet part of this training pipeline.
