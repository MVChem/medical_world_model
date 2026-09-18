# CVPR Paper

- `sections/`: manuscript sections and figure environments.
- `tables/`: LaTeX table sources (`.tex` only).
- `imgs/`: reserved for additional image assets; currently empty.
- `ppt/`: editable slides, figure sources, and authoring scripts. The paper uses PDFs from `ppt/ppt/`.

`main.tex` and `supplementary.tex` are the document entry points.
Shared settings are in `paper_config.tex` and `preamble.tex`; references are in `main.bib`.
The existing CVPR 2026 template (`cvpr.sty`, `ieeenat_fullname.bst`) is retained for the CVPR 2027 draft.

Build with `make` (requires LaTeX, BibTeX, latexmk, LibreOffice, and Python with Matplotlib, NumPy, Pillow, and Arial).
PDFs stay local. Missing figure PDFs are rebuilt from the tracked PowerPoint files and appendix script.
Use `make PYTHON=/path/to/python` to select a Python environment.
Run `make clean` to remove compiled PDFs and temporary files.

Editing guidelines: [AGENTS.md](AGENTS.md).
