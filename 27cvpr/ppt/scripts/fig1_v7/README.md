# Figure 1: direct image input for spatial tasks

The current editable deck is `../../ppt/fig1_v7.pptx`, with matching PDF, SVG,
and PNG exports. The manuscript includes its PDF as `../../../imgs/fig1.pdf`.

The black route starts at the current input image `X_t` and reaches the
spatial task decoders directly. For super-resolution this is the low-resolution
input. JEPA features `D_t` continue through the visual adapter to the VLM; the
state `S_t` still conditions task decoding and is the world model's input.

The v7 PowerPoint is the source of truth. Its direct image route consists of
three editable vector connectors and displays only one editable `X_t` label,
with `t` formatted as a subscript. The trailing spatial-task and LR-input
explanation is omitted from the figure; the manuscript text and caption
explain the LR/HR roles. Task query `Q` and state `S_t` still enter the task decoder separately.
The latent world model takes only `S_t` and horizon `h` and predicts future
state slots; the image branch does not enter that predictor.

For super-resolution, both the direct image branch and the state encoder use
the low-resolution observation. The high-resolution image is the MSE target,
not an input to either branch. The task decoder may extract its own spatial
features internally; the image branch does not denote JEPA output features.

`update_image_input.py` validates the image-to-decoder connectors and the
single `X_t` label, then refreshes
the PDF, SVG, and PNG from the **existing v7 PPTX**, then copies the PDF into
`imgs/fig1.pdf`. It does not modify the PPTX or medical thumbnails. The earlier
v6-to-v7 migration has already been applied; the refresh script no longer
rebuilds v7 from v6, so subsequent manual PowerPoint edits are preserved.

From `27cvpr`:

```bash
/home/data2/chk/workspace/2026/.venv/bin/python ppt/scripts/fig1_v7/update_image_input.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```
