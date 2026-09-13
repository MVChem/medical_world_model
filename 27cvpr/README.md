# MedWorld-JEPA — CVPR 2027 working draft

This directory contains the current anonymous CVPR draft for **MedWorld-JEPA**,
including the manuscript, supplementary material, and editable figure assets.

The target is **CVPR 2027**. For now, the layout uses the official **CVPR 2026**
author kit, release `CVPR2026-v1(latex)`. `cvpr.sty` and `ieeenat_fullname.bst`
are unmodified upstream files; see [template provenance](TEMPLATE.md).
The year shown in the PDFs is 2026, matching the official template; the project
directory remains `27cvpr`.

Run `make` here, or `make -C 27cvpr` from the project root. Outputs:

- [main.pdf](main.pdf): two-column manuscript and numbered references.
- [supplementary.pdf](supplementary.pdf): evaluation details and the longitudinal
  case figure, with references to the main paper's tables.

Original MIMIC case images, case metadata, report-reference notes, and historical
decks containing those original images stay local under `.gitignore`. Compiling
the manuscript uses the exported figures included in the repository; rebuilding
the case figure requires its local source materials.

The shared title, target year, and placeholder paper ID are in
[paper_config.tex](paper_config.tex). Both entry points use
`\usepackage[review]{cvpr}` for anonymous review with line numbers.
The 2026 main-paper limit is eight pages including figures and tables, plus
references; recheck the 2027 guidelines and author kit when they become available.

After migration on 2026-09-12, the main content ends on page 7; `main.pdf`
has 9 pages including references. The supplement has 4 pages. Both PDFs and
the two-page experiment plan compile without unresolved references or reported
box overflows. Page previews are generated in `preview/` for this migration.

The planned Tables 1 and 2 now appear in the manuscript. Each has **10
method/settings**, with no category-title rows. The same editable table sources
also produce the two-page [table preview](plans/table1_table2_plan.pdf):

- [Table 1 source](tables/table1_future.tex): future prediction.
- [Table 2 source](tables/table2_downstream.tex): six downstream task families.
- [Execution plan and model selection](plans/README.md).

Results retains three subsections: Experimental setup, Future prediction, and
Downstream tasks. `sections/6_results_analysis.tex` includes the brief setup from
`sections/5_experiments.tex` and the two shared table sources.

Table 1 uses the agreed **four dimensions, two metrics each**: future clinical
status (macro AP/AUROC), disease progression (Transition/Direction macro F1),
future report fidelity (RadGraph partial F1/GREEN), and probabilistic reliability
(macro Brier/classwise ECE). Lower Brier/ECE are better. Onset/resolution
breakdowns, CheXbert F1, and retrieval are supplementary. Zero-shot VLM rows are
marked ZS; the same-scale direct baseline and Ours are task-trained. Planned VLM
finding scores come from Yes/No answer likelihoods. Direction labels and new
scorers still require validation and implementation under this fixed metric set.

Table 2 covers classification, standard VQA, current report generation, phrase
grounding, organ segmentation, and spatial 4x super-resolution. It excludes
temporal understanding. Pseudo-label and human-mask Dice occupy separate columns.
The no-slots baseline and Ours share six-task adaptation and decoder budgets.
The image-based evaluation withholds the same-exam report; previous four-task
report-assisted scores must not be inserted into this expanded plan.

Methods now follows Figure 2's multi-scale design: four fusion-layer slots and
four slots from the VLM's own vision encoder, each sampling one early, two
intermediate, and the final layer. Classification/disease recognition read all
eight; segmentation/SR read the four visual slots plus the task image (LR for SR).
Task gradients jointly train slot readouts and decoders. This design requires
new training; existing prototype results do not validate it. Additional readouts
and image-only adaptation remain planned work. The appendix records the input,
supervision, scoring, and cohort conditions needed before filling the tables.

## Compile

```bash
make
make plan
make plan-preview
```

To build only the main paper, run `make paper`. `make supplementary` first
builds the main paper, then imports its labels via `xr-hyper`. On Overleaf,
compile `main.tex` before switching the main document to `supplementary.tex`.
Keep both PDFs together for links from the supplement to the main paper.
`make clean` removes LaTeX outputs; `make` regenerates both documents.

The main paper and supplement use 10-point Times in two columns on US Letter
paper. Figure 1 and result tables span both columns; Figure 2 is a compact
single-column detail of the eight-slot construction. Long equations use aligned
lines. The standalone experiment plan stays single-column at the CVPR text
width and shares the manuscript's table sources.

The architecture figure is `imgs/fig1.pdf`, exported from
`ppt/ppt/fig1_v7.pptx`. The slot figure is `imgs/fig2_v2.pdf`, with its editable
source in `ppt/ppt/fig2_v2.pptx` and build/export instructions in
`ppt/scripts/fig2_v2/README.md`. The supplementary case figure is
`ppt/ppt/appendix_fig_v3.pdf`. Other figure versions and their authoring scripts
are retained as archival assets.

## Result status

Every scheduled result is marked `TBD`; a dash means the specified row does not
evaluate that metric. No preliminary result has been inserted. Finalize cohorts,
checkpoint versions, adaptation budgets, human-mask data, and reference masks
before running the plan and making performance claims.
