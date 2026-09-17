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

Tables 1 and 2 now include the available results checked on **2026-09-15**.
Table 1 has **11 method/settings** and Table 2 has **13**, with no category-title
rows. Both use **Qwen3.5-9B** for the Qwen zero-shot reference. The same editable
table sources also produce the [table preview](plans/table1_table2_plan.pdf):

- [Table 1 source](tables/table1_future.tex): future prediction.
- [Table 2 source](tables/table2_downstream.tex): six downstream task families.
- [Execution plan and model selection](plans/README.md).

Both tables display scores multiplied by 100, with two decimal places; PSNR
retains its dB unit. Brier/ECE remain lower-is-better. The aggregate JSON and
original experiment results retain their original scales and full precision.

Table 1's bottom three rows use **Qwen3.5-9B**: full-token,
slots, and shuffled state. All three have completed 2,400 forecast updates,
297-pair evaluation, and GREEN scoring from a shared new 9B Stage-1 warm start.
CheXagent's forecast, current classification, and report scores are also filled.
See the [September 15 backfill record](../research_notes/0915_table_results_filled.md);
the previous 0.8B results are preserved in the
[upgrade record](../research_notes/0914_qwen9b_forecast_upgrade.md).

Results has five subsections: Experimental setup, Future prediction, Downstream
tasks, Patient trajectory, and Attention visualization.
`sections/6_results_analysis.tex` includes the brief setup from
`sections/5_experiments.tex` and the two shared table sources.

The **2026-09-17 figure selection** replaces the five earlier Results candidates
with two planned visualizations in dedicated subsections:

- [Patient trajectory](figures/candidate_forecasting.tex), spanning both columns:
  observed CXR timeline, predicted reports, and finding probabilities from a
  fixed Day 0 at the actual follow-up horizons.
- [Attention visualization](figures/candidate_slot_attention.tex), in one column:
  the current CXR and four actual visual-slot readout maps in a 2×2 grid.

Attention is now in the main paper; the supplementary Candidate Visual Analysis
section is no longer included. Both figures remain explicitly marked as pending
until test-case predictions and verified attention exports are available. Other
candidate sources and trial exports remain archived. See the
[selected figure plan](../research_notes/0917_trajectory_and_slot_attention_plan.md).

The September 17 build has 10 main-paper pages (body through page 9) and
5 supplementary pages. Both PDFs compile without unresolved references or box
overflows, and `preview/` has been refreshed. The draft still needs shortening
to fit the eight-page body limit noted above.

First empirical attempts are now available in the separate
[four-figure review PDF](figures/generated/candidate_figures_v1.pdf), with
[overview](figures/generated/candidate_figures_v1_overview.png) and
[rebuild scripts, individual exports and scope notes](figures/README.md).
They use available frozen visual controls and baseline classifiers; missing
full-model conditions remain identified. An additional historical 0.8B
[forecasting visualization](figures/generated/forecasting.pdf) is also available.
These review artifacts do not replace the full-method manuscript placeholders.

Table 1 uses the agreed **four dimensions, two metrics each**: future clinical
status (macro AP/AUROC), disease progression (Transition/Direction macro F1),
future report fidelity (RadGraph partial F1/GREEN), and probabilistic reliability
(macro Brier/classwise ECE). Lower Brier/ECE are better. Onset/resolution
breakdowns, CheXbert F1, and retrieval are supplementary. Zero-shot VLM rows are
marked ZS. The three 9B forecast conditions use 16,000 training pairs,
2,400 updates at effective batch size 32 after a shared 9B Stage-1 warm
start. Validation/test contain 230/297
pairs, with 94 test patients. VLM finding scores use Yes/No answer likelihoods.
All reported forecast rows share per-metric reference masks. Direction labels
remain unadjudicated, so that column stays pending. BioViL-T/CheXWorld forecast
adaptations and their GREEN scoring are complete. Their trained modules use the
common Stage-1 warm start; these are not the original papers' native forecasters.
The 9B slots improve Transition F1 and AUROC over full-token, with lower
Brier/ECE but slightly lower AP, RadGraph, and GREEN. Shuffling collapses all
297 reports to one output and yields zero CheXbert F1; its low ECE accompanies
near-chance AUROC and weak report scores.

Table 2 covers classification, standard VQA, current report generation, phrase
grounding, organ segmentation, and spatial 4x super-resolution. It excludes
temporal understanding. Pseudo-label and human-mask Dice occupy separate columns.
The planned no-slots and full Ours rows remain pending under matched six-task
adaptation. Separate Qwen3.5-9B frozen visual-slot, shuffled-slot, and image-only
rows report the completed dense-task controls. Main-table dense results use
4,096/249/447 training/validation/test images, 20 epochs, and the separate
Montgomery 138-image human lung test. DINOv2/CheXWorld classification heads use
13,681/160/353 images; current-report scoring uses 507 test images. Standard VQA
and MS-CXR grounding are pending. Derived QA, anatomical-region localization,
old report-assisted scores, and separate two-epoch joint pilots do not fill
those pending rows or columns.

Methods now follows Figure 2's multi-scale design: four fusion-layer slots and
four slots from the VLM's own vision encoder, each sampling one early, two
intermediate, and the final layer. Classification/disease recognition read all
eight; segmentation/SR read the four visual slots plus the task image (LR for SR).
Task gradients jointly train slot readouts and decoders. This design requires
new training. Table 1's **9B slot adaptation uses final-layer language queries**,
not the multi-depth 4+4 architecture, and its full-token control retains the
same forecast module. Neither it nor Table 2's frozen visual-slot controls
validate the full proposed system. The appendix records the distinct inputs,
supervision, scoring, and cohorts for the partial results and remaining work.

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

The architecture figure is `imgs/fig1.pdf`, an exact copy of the user-selected
`ppt/ppt/fig1_v7.pdf` as of 2026-09-15; `imgs/fig1.png` is rendered from that PDF.
Its editable source is `ppt/ppt/fig1_v7.pptx`. The slot figure is `imgs/fig2_v2.pdf`, with its editable
source in `ppt/ppt/fig2_v2.pptx` and build/export instructions in
`ppt/scripts/fig2_v2/README.md`. The supplementary case figure is
`ppt/ppt/appendix_fig_v3.pdf`. Other figure versions and their authoring scripts
are retained as archival assets.

## Result status

Available metrics have been inserted; `TBD` means pending, while a dash means
the specified row does not evaluate that metric. Scores are point estimates,
not significance claims. Data and epoch/update counts are matched within the
described protocols, but actual GPU hours and pretraining differ: SwinIR's
4,096-image run uses 8.313 GPU hours including evaluation, versus 0.161 for the
image-only SR run. The comparison is not equal-compute.

CheXagent and all three 9B forecast conditions are complete for their available
metrics. MAIRA-2 awaits checkpoint access. Direction annotation, transition-prior
threshold selection/scoring, formal VQA, MS-CXR grounding, the complete
six-task no-slots/Ours experiments, and cross-task/pretraining overlap audits
remain unfinished. The larger 18,708-image dense experiments are recorded in
the experiment reports and are not mixed into this 4,096-image main-table panel.
