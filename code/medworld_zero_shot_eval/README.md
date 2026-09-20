# Medical World Model: native-model baselines

Evaluate pretrained Qwen3.5-0.8B, Qwen3.5-4B, Qwen3.5-9B and MedGemma-1.5-4B without project fine-tuning. Qwen3.5-0.8B is the primary same-backbone baseline for MedWorld; the larger models provide additional reference points.

## Tasks and protocol

- Classification: the same 13-label test cohort as MedWorld; normalized Yes/No next-token probabilities; macro and per-label AUROC/AP. Missing and uncertain labels are masked.
- VQA: the same selected test questions, JSON instruction, greedy generation budget and strict label-set scorer as MedWorld; exact match, micro-F1, per-type metrics and invalid-output counts. No constrained decoding or answer-based filtering.
- Segmentation: unavailable through the native models' interfaces.

Images come from MedWorld's on-demand data loader. Qwen uses its configured visual pixel budget; MedGemma uses its native processor. This matches source images but not visual token counts or compute budgets. The checkpoint inventory is defined in `models.py`; data and evaluation settings come from the supplied MedWorld JSON.

## Single-model evaluation

From the repository root:

```bash
PYTHONPATH=code /home/data2/chk/workspace/2026/.venv/bin/python -m medworld_zero_shot_eval.evaluate \
  --config code/medworld/configs/qwen35_08b_vssc_4gpu.json \
  --model qwen08b --tasks classification vqa --gpu auto \
  --out code/medworld_zero_shot_eval/runs/qwen08b_native_YYYYMMDD
```

Model IDs: `qwen08b`, `qwen4b`, `qwen9b`, `medgemma4b`. Add `--limit 2` for a partial GPU check; omit it for full evaluation. Existing nonempty output folders are rejected. Outputs include predictions, aggregate metrics, data/config provenance, weight hashes and live status. Runs do not currently resume partial predictions.

`sweep.py` runs a two-example check for each model before its full evaluation, using one explicitly selected idle GPU by default (models run serially). Use `--skip-smoke` after preflights have passed to run only the test sets. It maintains the experiment registry and writes per-model logs and status under the run folder. Use a dated output name. GPU preference is 1,2,3, then 6,7, then 4,5, with 0 last.

## Validation

```bash
PYTHONPATH=code /home/data2/chk/workspace/2026/.venv/bin/python -m unittest discover -s code/medworld_zero_shot_eval/tests -v
```

## Historical VQA utilities

This project was renamed from `medworld_vqa`; the compatibility symlink was removed during the September 20 cleanup. The older `prepare`, `run`, `export` and `literature_*` utilities implement separate sampled VQA protocols, including vocabulary-constrained JSON and free short answers. They are not the matched evaluator described above.

Previous run directories were deleted at the user's request on 2026-09-20. Aggregate reports remain in `results/mimic_cxr_vqa_pilot_20260916` and `results/mimic_cxr_vqa_literature_20260916`; their original run links no longer resolve. No historical aggregate scores have been changed.

## Balanced VQA test subset

The current MedWorld JSON selects 100 Verify, 100 Choose and 100 Query test questions (`testing.vqa_per_type=100`, `testing.vqa_seed=42`). Classification still uses all 353 test images. Both evaluators share deterministic answer-independent sampling and save question IDs and their hash in `vqa_selection.json`. Set `vqa_per_type=0` for the complete test set. Aggregate VQA scores on the balanced 300-question subset are not full-test scores.
