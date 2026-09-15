# Candidate visualizations — first empirical versions

The `generated/` directory contains local case-level exports and is excluded
from Git. The links below refer to those local artifacts. The paper compiles
from the separate `candidate_*.tex` placeholders without these exports.

Open [the four-figure review PDF](generated/candidate_figures_v1.pdf) or
[the overview](generated/candidate_figures_v1_overview.png).
The four pages correspond to the downstream candidates currently shown as
Figures 4–7 in `main.pdf`. These are initial attempts made from available
local data and final checkpoints, with their actual scope labeled on each page.
The full-method manuscript placeholders remain separate because several of
their planned conditions have not been trained or evaluated.

| Candidate | First version | What is still missing |
|---|---|---|
| [Clinical evidence](generated/clinical_evidence.pdf) | One fixed test CXR, two official positive findings; real input-space blur occlusion for DINOv2 and CheXWorld final classification heads, with equal-area random controls | Proposed-slot classifier under matched image-only training; lesion boxes; grounding/localization evaluation |
| [State retrieval](generated/state_retrieval.pdf) | Actual Top-3 rankings from cached frozen visual-depth summaries, shared test bank, different patients; full-cohort label/view audit and random-bank expectation | Fusion slots and the complete eight-slot representation |
| [Slot perturbation](generated/slot_perturbation.pdf) | Same trained decoder and direct image, different patient's cached slot condition; segmentation/SR differences and reference errors, including weak-response cases | Local blur followed by slot re-extraction; full jointly trained model |
| [Spatial outputs](generated/spatial_outputs.pdf) | Matched image-only, frozen visual-slot and shuffled-trained decoders; CXAS pseudo reference, Montgomery human reference, 4× SR and common crop | Matched jointly trained model; full six-task checkpoint |

An additional [longitudinal forecasting trial](generated/forecasting.pdf) uses
saved **Qwen3.5-0.8B pilot** forecasts for report-label onset, resolution and
persistence. It is separate from the pending 9B rows and the proposed multi-depth
4+4 model. Follow-up images are observed references. Report-text consistency
screening and exclusions are recorded in its provenance; this is not clinical
adjudication.

The clinical blur controls match area on the input canvas. They do not match
retained image/anatomical area after native preprocessing: CheXWorld uses a
center crop, and some cells include padding. This is an exploratory sensitivity
check. Displayed probabilities are CPU FP32 re-encodings, with the small
differences from saved BF16 scores documented in the provenance.

Each figure has PDF, PNG and SVG exports. Text and drawn marks remain editable
in the PDF/SVG; radiographs and numerical maps are raster layers. Python scripts
are in [scripts](scripts/). The JSON provenance records identify data,
checkpoints, selection rules and limitations. Per-query rankings and spatial
prediction arrays are retained locally alongside the exports. Inputs and model
weights stay in the existing experiment directories.

## Rebuild

From the project root, using the existing environment:

```bash
PY=/home/data2/chk/workspace/2026/.venv/bin/python
$PY 27cvpr/figures/scripts/build_clinical_evidence.py
$PY 27cvpr/figures/scripts/build_state_retrieval.py
$PY 27cvpr/figures/scripts/build_spatial_figures.py
$PY 27cvpr/figures/scripts/build_forecasting.py
$PY 27cvpr/figures/scripts/build_candidate_review.py
```

The clinical attribution script defaults to CPU and performs actual image
encoding after each perturbation; `--render-only` redraws its saved numerical
results. The review script simply combines the four completed downstream PDFs
without rasterizing their vector layers.

The supplementary slot-attention candidate remains pending. The frozen visual
cache uses mean pooling and has no learned slot-to-patch attention. Decoder
attention over four aggregated slots cannot be relabeled as slot-to-image
attention.
