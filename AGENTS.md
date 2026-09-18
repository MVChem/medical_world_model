# GPU Usage

- Never use physical GPUs `4` or `5`.
- Prefer GPUs `1`, `2`, and `3`, then `6` and `7`; use GPU `0` last. Set `CUDA_VISIBLE_DEVICES` accordingly.

# Version Control

- Keep PDFs local; do not commit them.
- Keep large files out of Git, except figure images and editable sources in `27cvpr/ppt/`. Check staged file sizes before pushing.
