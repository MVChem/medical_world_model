# Code

Project code and third-party repositories. See [THIRD_PARTY.md](THIRD_PARTY.md) for upstream sources and [AGENTS.md](AGENTS.md) for development guidelines.

- Keep data and large files under `/home/data2/chk/data`; access them through symbolic links in `code/data/`.
- Use the shared Python environment through `code/.venv`.
- Prefer modern HTML for data presentation, using React, FastAPI, or similar tools as needed. Take design inspiration from Figma or other websites.
- Keep project-specific setup and usage in each project's README.

## Active projects

`medworld/` is the main implementation; `medworld_zero_shot_eval/` provides native
model comparisons. `mimic_atlas/` and `glioma_explorer/` provide data browsers.
The retained third-party references are listed in [THIRD_PARTY.md](THIRD_PARTY.md).

Obsolete experiment packages were removed on September 20, 2026. See the
[cleanup record](CLEANUP_20260920.md) for scope, historical source recovery, and
local asset migration. Required MedWorld assets live outside Git under
`/home/data2/chk/data/medworld`, linked as `code/data/medworld/`.
