# Segmentation Evaluation: With vs Without Slots

- Codex Chat ID: `01a0bdc7-97fd-73f2-895a-3a3a50a7f030`
- ID source: `CODEX_THREAD_ID` in the current conversation environment; identical to `CODEX_SESSION_ID`.
- Archive date: 2026-09-21 (Asia/Shanghai).
- Experiment: `paired_vssc_2gpu_12h_fast_20260920`.
- Original run directory: [paired_vssc_2gpu_12h_fast_20260920](../../code/medworld/runs/paired_vssc_2gpu_12h_fast_20260920/).
- Local evaluation snapshot: [evaluation_results.json](evaluation_results.json), containing full-precision results, configurations, source-file SHA-256 hashes, and the conversation ID.

## Git Commit Traceability

- Repository: [MVChem/medical_world_model](https://github.com/MVChem/medical_world_model).
- Code and evaluation archive commit: [`59879c56b6ad170ceed74dc07bfe96754d3c8b2b`](https://github.com/MVChem/medical_world_model/commit/59879c56b6ad170ceed74dc07bfe96754d3c8b2b).
- Commit subject: `Optimize MedWorld training and archive complete slots evaluations`.
- Branch: `main`.
- Validation at this commit: `PYTHONPATH=code /home/data2/chk/workspace/2026/.venv/bin/python -m pytest code/medworld/tests -q` — 54 tests and 26 subtests passed; 16 dependency deprecation warnings. `git diff --cached --check` passed before commit.

This commit archives the repository after the recorded 0.8B experiment and includes subsequent 9B configuration and native-backbone comparison support. It is not a claim that every file in this commit was used during the original training. The run's frozen training source, supplemental evaluation source, manifests, and checkpoint/data fingerprints remain the authoritative execution record.

This traceability section is saved in a following documentation commit so it can reference the complete code/archive commit ID. Both commits are published together. To inspect the archived implementation, use `git show 59879c56b6ad170ceed74dc07bfe96754d3c8b2b` or browse the commit link above.

## Experiment Setup

The model with 8 slots and the model without slots were trained independently for 1,350 optimizer updates each, using matching training settings and effective batch sizes on GPUs 2 and 3. The 12-hour budget covered both training arms combined. Calibration determined an equal update count with 10% headroom. Actual training took approximately 5 hours 29 minutes and 4 hours 59 minutes, respectively. Both evaluations used `final.pt`; checkpoints were not selected using test results. Native Qwen 0.8B was not trained and served as a classification and VQA reference.

Supplemental segmentation evaluation completed on 2026-09-21 using the saved weights, without retraining. The frozen training source and weights remain unchanged. Supplemental evaluation source and provenance are stored in `segmentation_backfill_20260921/` under the original run directory.

## Main Segmentation Results

Higher is better for all scores. Δ = with slots − without slots, expressed as an absolute score difference.

| Test set | N | Metric | With slots | Without slots | Δ |
|---|---:|---|---:|---:|---:|
| CXAS pseudo-labels (three organs) | 447 | mean_dice | 0.891610 | 0.893171 | -0.001561 |
| CXAS pseudo-labels (three organs) | 447 | mean_iou | 0.811324 | 0.813820 | -0.002495 |
| Human annotations (two lungs) | 138 | mean_dice | 0.728133 | 0.730064 | -0.001932 |
| Human annotations (two lungs) | 138 | mean_iou | 0.574141 | 0.576439 | -0.002298 |

The model without slots scored slightly higher on both Dice and IoU in both segmentation tests. This run showed no segmentation benefit from slots. The differences are small, and statistical significance has not been tested. Human-annotation and pseudo-label results should be interpreted separately. Native Qwen has no segmentation interface, so its segmentation scores are N/A.

### Segmentation Metric Protocol

- `test`: all 447 CXAS pseudo-labeled images, with three-organ targets.
- `human_test`: all 138 images with human lung annotations, with two-lung targets.
- Predictions use `sigmoid(logits) > 0.5`; targets use `target > 0.5`. Metrics are computed only within the valid-region mask.
- IoU = intersection / union; Dice = 2 × intersection / (predicted area + target area). Both numerator and denominator receive `1e-6` smoothing. Two empty masks score 1.
- Organ scores are averaged equally within each image, then image scores are averaged equally. IoU is computed directly, rather than converted from mean Dice.

## Classification and VQA Evaluation

| Task | N | Metric | With slots | Without slots | Native Qwen |
|---|---:|---|---:|---:|---:|
| classification | 353 | macro_auroc | 0.780537 | 0.774760 | 0.683637 |
| classification | 353 | macro_ap | 0.874094 | 0.871673 | 0.824817 |
| vqa | 300 | exact_match | 0.500000 | 0.460000 | 0.256667 |
| vqa | 300 | micro_f1 | 0.521739 | 0.507463 | 0.269481 |

Classification used 353 test samples. VQA used a fixed seed of 42 to sample 100 questions per type, totaling 300 questions; this was a subset of the VQA test set. Compared with the model without slots, the model with slots improved classification macro AUROC by approximately 0.00578 and VQA exact match by 4 percentage points. Statistical significance has not been tested.

### VQA Breakdown by Question Type

| Type | N | Exact match: with slots | Exact match: without slots | Invalid answers: with slots | Invalid answers: without slots |
|---|---:|---:|---:|---:|---:|
| choose | 100 | 0.4600 | 0.4200 | 0 | 3 |
| query | 100 | 0.3500 | 0.2800 | 5 | 5 |
| verify | 100 | 0.6900 | 0.6800 | 0 | 0 |

## Results, Checkpoints, and Logs

- [Full comparison report](../../code/medworld/runs/paired_vssc_2gpu_12h_fast_20260920/COMPARISON.md)
- [Full comparison JSON](../../code/medworld/runs/paired_vssc_2gpu_12h_fast_20260920/comparison.json)
- [Supplemental segmentation logs, source, and status](../../code/medworld/runs/paired_vssc_2gpu_12h_fast_20260920/segmentation_backfill_20260921/)
- [All evaluation results: with slots](../../code/medworld/runs/paired_vssc_2gpu_12h_fast_20260920/slots/evaluation/)
- [All evaluation results: without slots](../../code/medworld/runs/paired_vssc_2gpu_12h_fast_20260920/baseline/evaluation/)
- [Native Qwen evaluation results](../../code/medworld/runs/paired_vssc_2gpu_12h_fast_20260920/qwen/)
- [Per-step training metrics: with slots](../../code/medworld/runs/paired_vssc_2gpu_12h_fast_20260920/slots/metrics.jsonl)
- [Per-step training metrics: without slots](../../code/medworld/runs/paired_vssc_2gpu_12h_fast_20260920/baseline/metrics.jsonl)
- [Training launch log: with slots](../../code/medworld/runs/paired_vssc_2gpu_12h_fast_20260920/slots.log)
- [Training launch log: without slots](../../code/medworld/runs/paired_vssc_2gpu_12h_fast_20260920/baseline.log)
- [Final checkpoint: with slots](../../code/medworld/runs/paired_vssc_2gpu_12h_fast_20260920/slots/final.pt)
- [Final checkpoint: without slots](../../code/medworld/runs/paired_vssc_2gpu_12h_fast_20260920/baseline/final.pt)

The original `evaluation/<task>/summary.json` files contain aggregate metrics and data/checkpoint fingerprints; `<task>.jsonl` files contain per-sample evaluation records. The archived JSON includes per-finding classification results, per-organ segmentation results, and VQA breakdowns by question type. Per-sample records and large checkpoints remain in the original run directory. Deleting that directory would break these links; this archive does not include checkpoint backups.

## Requirements for Future Experiments

Every completed formal experiment must evaluate segmentation on both the pseudo-label and human-annotation test sets and report IoU and Dice for both trained arms. This requirement is recorded in the project `AGENTS.md`, and the default experiment configuration enables both tests. Calibration and short smoke tests may skip full evaluation.
