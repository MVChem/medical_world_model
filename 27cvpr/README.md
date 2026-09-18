# CVPR Paper

- `sections/`: manuscript sections and figure environments.
- `tables/`: LaTeX table sources (`.tex` only).
- `imgs/`: figures included in the paper.
- `ppt/`: editable slides, figure sources, and authoring scripts.

`main.tex` and `supplementary.tex` are the document entry points.
Shared settings are in `paper_config.tex` and `preamble.tex`; references are in `main.bib`.
The existing CVPR 2026 template (`cvpr.sty`, `ieeenat_fullname.bst`) is retained for the CVPR 2027 draft.

Build with `make` (requires LaTeX, BibTeX, and latexmk).
Run `make clean` to remove compiled PDFs and temporary files.
