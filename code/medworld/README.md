# MedWorld: three image-conditioned tasks

The supported downstream tasks are **classification, segmentation and VQA**. Diagnosis is the same disease-label recognition problem and is not counted again. Super-resolution, report generation and temporal task evaluation have been removed from active code. Historical source is retained in Git; obsolete local experiment artifacts were removed in the [September 20 cleanup](../CLEANUP_20260920.md).

Every task retains the image as its primary input. VQA also receives the question. The same Qwen visual encoder and shared two-layer Transformer task decoder serve all three tasks:

```text
image → Qwen visual tokens ──→ shared Transformer task decoder → classification linear head
                         ↑                                  → spatial segmentation head
                  optional 8 slots                          → Qwen language decoder → VQA answer
```

`slot_conditioning=false` removes the slots from the task decoder's inputs, leaving a complete image-driven model. `true` adds all eight slots as extra tokens. Segmentation reshapes spatial output tokens, upsamples, and predicts three organ masks; the human test scores only the two lungs. VQA generates a JSON answer array. Its autoregressive language decoder is additional to the common Transformer trunk; Qwen3.5's language backbone is a hybrid architecture, not a claim of identical pure-Transformer output heads.

Both arms use identical model initialization, task heads, input data, update counts and loss definitions. Temporal latent prediction/EMA remain **representation-learning objectives**, not downstream tasks. Both arms receive the same temporal auxiliary supervision; optional visual consistency is also retained in both arms, so adding the slot condition is the controlled change. No report/target answer enters a current-task image representation. No pixel or feature caches are written.

## Configuration

The training JSON sets `slot_conditioning`, model/training parameters and testing:

```json
{
  "slot_conditioning": true,
  "testing": {
    "enabled": true,
    "tasks": ["classification", "segmentation", "vqa"],
    "human_segmentation": true,
    "reuse_completed": true
  }
}
```

- `enabled=false`: training only; no test workers start.
- `tasks`: only these tasks are tested; for example `["classification", "vqa"]` skips segmentation.
- `human_segmentation`: additionally tests Montgomery when segmentation is selected.
- `reuse_completed`: reuses complete outputs only when the final-checkpoint hash, data fingerprint, split, prediction hash and counts match. Partial/incompatible outputs fail rather than being silently overwritten. `false` requires empty selected task output folders.

Defaults test all three tasks and human segmentation. Edit only the `testing` section in a completed run's `config.json` to choose subsequent tests; its trained model and task parameters are restored from the checkpoint. CLI `--skip-evaluation` remains an explicit launcher override for preflights. `--smoke` performs bounded checks instead of full testing.

## Run

Use the local Python environment with `PYTHONPATH=code`, from the repository root:

```bash
# Three updates, gradient and checkpoint-reload audit; no full test suite.
python -m medworld.train --smoke --gpu 1 --out code/medworld/runs/smoke_YYYYMMDD

# One model; automatically runs the tests selected in its JSON.
python -m medworld.launch_distributed --config code/medworld/configs/qwen35_08b_vssc_4gpu.json \
  --gpus 1,2,3,6 --out code/medworld/runs/slots_YYYYMMDD

# Main experiment: separately train both arms, run configured tests, compare.
python -m medworld.run_experiment --config code/medworld/configs/qwen35_08b_vssc_4gpu.json \
  --gpus 1,2,3,6 --out code/medworld/runs/paired_YYYYMMDD

# Run or reuse the tests currently selected in a completed run's JSON.
python -m medworld.evaluate_run --run code/medworld/runs/slots_YYYYMMDD --gpus 1,2,3,6

python -m unittest discover -s code/medworld/tests -v
```

Paired experiments require `total_hours=0` and equal `steps`; a fixed wall-clock budget would give different numbers of updates. Outputs include task predictions/metrics under `evaluation/`, `evaluation_summary.json`, and the pair's `COMPARISON.md`/`comparison.json`. Turning testing off also skips automatic comparison. Selecting a subset compares only that subset. Single-run testing does not launch another training run; use `run_experiment` for the paired experiment.

## Data and metrics

- Classification: the existing 13 CheXpert-derived labels (12 disease/finding labels plus Support Devices); masked uncertain/missing labels, AUROC/AP. These report-derived labels are not independent clinical diagnostic ground truth.
- Segmentation: CXAS pseudo three-organ supervision, Dice; Montgomery two-lung human test separately.
- VQA: official local MIMIC-CXR-VQA image/question pairs; full test has 13,793 questions. Strict JSON label-set exact match and micro-F1, with Verify/Choose/Query breakdown. Invalid responses count as errors; this local scorer is not claimed to be the official evaluator.

Global patient holdouts cover all tasks and temporal representation learning. Higher-priority held-out membership removes conflicting training/validation rows; it never moves rows into evaluation. The public 110-answer vocabulary is stored in `datasets/vqa_vocabulary.json`, copied from the previously frozen official `ans2idx.json` metadata, not inferred from test answers.

Checkpoint format 3 reflects the new task interfaces. Older weights require their original frozen source and cannot resume into this architecture. The code still reads local Qwen and V-JEPA weights; no external services are required.

## VQA test sampling

`testing.vqa_per_type` selects this many questions from each of Verify, Choose and Query, using `testing.vqa_seed`. Zero retains the complete test set. The provided four-GPU training config now selects 100 per type with seed 42. Training data is unchanged. The native-model baseline uses the same selection helper; both save exact IDs and their hash. Balanced-subset metrics are distinct from full-test metrics.

## Local assets

Required manifests, labels and the Qwen 0.8B checkpoint are stored under
`code/data/medworld/`, a local symlink to `/home/data2/chk/data/medworld`.
The V-JEPA checkpoint remains under `code/vjepa2/checkpoints/`.
Historical configs are translated by `asset_paths.py`; unchanged manifests keep
their historical provenance keys so relocation does not change data fingerprints.
