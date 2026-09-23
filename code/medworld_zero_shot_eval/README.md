# Medical World Model: native-model baselines

Evaluate pretrained Qwen3.5-0.8B, Qwen3.5-4B, Qwen3.5-9B and MedGemma-1.5-4B without project fine-tuning. Each paired MedWorld experiment evaluates native Qwen at the same backbone size as its slots model, alongside the separately trained raw-input baseline.

Inference uses local Hugging Face Transformers checkpoints with PyTorch, BF16 and SDPA. Classification reads next-token logits directly; current VQA uses `model.generate()`. The supported entry points are `evaluate.py`, `future_evaluate.py` and `sweep.py`. Future-report scoring uses the separately pinned official RadGraph CPU worker.

## Tasks and protocol

- Classification: the same 13-label test cohort as MedWorld; normalized Yes/No next-token probabilities; macro and per-label AUROC/AP. Missing and uncertain labels are masked.
- VQA: the same selected test questions, JSON instruction, greedy generation budget and strict label-set scorer as MedWorld; exact match, micro-F1, per-type metrics and invalid-output counts. No constrained decoding or answer-based filtering.
- Segmentation: N/A because native Qwen has no segmentation head. Both trained arms report Dice and IoU on the reviewed MIMIC heart/lung, UCSF-ALPTDG, MU-Glioma-Post and external Montgomery cohorts; MRI slices are aggregated by volume.

All new summaries use the strict `medworld-tables-v1` Table 2 metric protocol. Ordered cohort IDs, complete question/label references and prediction files have separate SHA256 fingerprints. A comparison rejects missing or changed protocol metadata, different references, incomplete predictions and different VQA selections. The supported Table 2 columns are classification macro AUROC/AP, VQA EM/micro-F1, and segmentation Dice/IoU. Historical metric schemas are not accepted.

Images come from MedWorld's on-demand data loader. Qwen uses its configured visual pixel budget; MedGemma uses its native processor. This matches source images but not visual token counts or compute budgets. The checkpoint inventory is defined in `models.py`; data and evaluation settings come from the supplied MedWorld JSON.

## Single-model evaluation

From the repository root:

```bash
PYTHONPATH=code /home/data2/chk/workspace/2026/.venv/bin/python -m medworld_zero_shot_eval.evaluate \
  --config code/medworld/configs/medworld_0923.json \
  --model qwen08b --tasks classification vqa --gpu auto \
  --out code/medworld_zero_shot_eval/runs/qwen08b_native_YYYYMMDD
```

Model IDs: `qwen08b`, `qwen4b`, `qwen9b`, `medgemma4b`. Add `--limit 2` for a partial GPU check; omit it for full evaluation. Existing nonempty output folders are rejected. Outputs include predictions, aggregate metrics, data/config provenance, weight hashes and live status. Runs do not currently resume partial predictions.

`sweep.py` runs a two-example check for each model before its full evaluation, using one explicitly selected idle GPU by default (models run serially). Use `--skip-smoke` after preflights have passed to run only the test sets. It maintains the experiment registry and writes per-model logs and status under the run folder. Use a dated output name. GPU preference is 1,2,3, then 6,7, then 4,5, with 0 last.

The sweep resolves configured asset paths and saves `evaluation_config.json` before launching workers. Use the current configuration schema and a fresh output directory for new evaluations.

## Table 1: native Qwen3.5-9B forecasting

`future_evaluate.py` evaluates all five Table 1 tasks on the same frozen references as the trained raw-input and slots arms. Its inputs are the source image, the source report capped at 383 UTF-8 bytes, the requested horizon, and an optional task question. Future images, reports, outcome labels and medication records never enter inference.

| Task | Metric | Native readout |
| --- | --- | --- |
| Future VQA | Accuracy | Argmax over fixed single-token codes for the 110 answer labels |
| Progression | Three-class balanced accuracy | Argmax over improved, stable and worsened option codes |
| Future report | RadGraph F1 | Greedy full-report generation; official RadGraph-XL partial F1 |
| Mortality within 30 days | AUROC | Next-token Yes/No softmax probability; fixed 720-hour request |
| Remaining hospital stay | MAE in days | Expected value over a fixed day grid; fixed 24-hour request |

The LOS day grid is `[0, 0.5, 1, 2, 3, 5, 7, 10, 14, 21, 30, 60, 90, 180, 365]`. Its readout is fixed before evaluation and is recorded in the summary. The 24-hour request does not reveal the observed discharge interval. Full original future reports are the references; all three arms share the configured report output byte cap. The formal configuration allows 2,047 output bytes.

```bash
PYTHONPATH=code HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/data2/chk/workspace/2026/.venv/bin/python \
  -m medworld_zero_shot_eval.future_evaluate \
  --config code/medworld/configs/qwen35_9b_tables12_2gpu_24h_0923.json \
  --model qwen9b --gpu auto \
  --out code/medworld_zero_shot_eval/runs/qwen9b_future_YYYYMMDD
```

Prepare `table12_v1/manifest.json` and the pinned local RadGraph assets first. Both the Qwen backbone and RadGraph assets load offline. The worker records ordered predictions, reference and model hashes, per-task scorer protocols, input/output budgets, and timing. A `--limit 1` preflight verifies execution but cannot establish progression balanced accuracy or mortality AUROC when reference classes are absent. Such outputs are explicitly partial and cannot populate the final paper tables.

## Validation

```bash
PYTHONPATH=code /home/data2/chk/workspace/2026/.venv/bin/python -m unittest discover -s code/medworld_zero_shot_eval/tests -v
```

## Retired metrics

The earlier sampled Diagnosis, sample-F1 and literature-specific scoring/export utilities have been removed. There is one active VQA scorer shared with MedWorld. Use `evaluate` or `sweep` for current inference.

Previous run directories were deleted at the user's request on 2026-09-20. Aggregate reports remain in `results/mimic_cxr_vqa_pilot_20260916` and `results/mimic_cxr_vqa_literature_20260916`; their original run links no longer resolve. No historical aggregate scores have been changed.

## Balanced VQA test subset

The `medworld_0923.json` configuration selects 100 Verify, 100 Choose and 100 Query test questions (`testing.vqa_per_type=100`, `testing.vqa_seed=42`). Classification uses every retained test image; its count is recorded in each summary and depends on the prepared dataset and shared patient holdouts. Both evaluators share deterministic answer-independent sampling and save question IDs and their hash in `vqa_selection.json`. Set `vqa_per_type=0` for the complete test set. Aggregate VQA scores on the balanced 300-question subset are not full-test scores.
