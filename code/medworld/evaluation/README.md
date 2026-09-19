# Evaluation

- `evaluate.py`: Evaluates a checkpoint on complete task cohorts using metric functions from downstream_tasks.
- `baseline_audit.py`: Checks native Qwen weights, samples, references, and pixels reconstructed from source images.
- `clinical_report.py`: CLI for clinical report scoring; `chexbert.py` and `clinical_metrics.py` provide local scoring implementations.
- `dense_reference.py`: Computes the bicubic super-resolution reference online.
- `compare_run.py`: Aggregates Stage 1/2 results and audited baseline comparisons.

Usage: `python -m medworld.evaluation.MODULE --help`.
Clinical scoring accepts `--config` to override `clinical_weights`; it does not import scoring code from older experiments.
All outputs belong in the selected run directory and are excluded from Git by default.
