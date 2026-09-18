# Code

Project code and third-party repositories. See [THIRD_PARTY.md](THIRD_PARTY.md) for upstream sources and [AGENTS.md](AGENTS.md) for development guidelines.

- Keep data and large files under `/home/data2/chk/data`; access them through symbolic links in `code/data/`.
- Use the shared Python environment through `code/.venv`.
- Prefer modern HTML for data presentation, using React, FastAPI, or similar tools as needed. Take design inspiration from Figma or other websites.
- Keep project-specific setup and usage in each project's README.
