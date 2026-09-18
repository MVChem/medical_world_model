# GPU Usage

- Never use physical GPUs `4` or `5`.
- Prefer GPUs `1`, `2`, and `3`, then `6` and `7`; use GPU `0` last. Set `CUDA_VISIBLE_DEVICES` accordingly.

# Version Control

- Keep PDFs local; do not commit them.
- Keep large files out of Git, except figure images and editable sources in `27cvpr/ppt/`. Check staged file sizes before pushing.

# Experiment Registration

- Keep experiment indexes brief and in English. Store detailed configurations, logs, and results in the original run directories; do not create per-experiment folders under `experiments/`.
- Register ongoing experiments in `experiments/registry.json` with an ID, short summary, status, observation timestamp, and links to the run and live status. Use repository-relative paths.
- When an experiment completes, fails, or is interrupted, move its entry to the history in `experiments/README.md`, recording the date, outcome, and run link.
