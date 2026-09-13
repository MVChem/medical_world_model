# fig1_v6

This version is archived. The manuscript now uses [fig1_v7](../fig1_v7/README.md),
which routes the input image directly to spatial task decoders.

This figure is `../../ppt/fig1_v6.pptx`, with matching PDF, SVG, and PNG exports. It preserves the user's layout edits, repairs the arrows, shows task query Q as an editable text prompt, and crops the PowerPoint canvas to 15.5 × 6.167 inches.

Shapes, connectors, labels, equations, charts, and token rows are editable PowerPoint objects. Only five medical thumbnails are raster crops. All five slot/state groups contain eight tokens. Vector exports retain vector content; SVG text is outlined for portable font appearance.

At the time of this version, its PDF was copied to `../../../imgs/fig1.pdf` and
checked on page 4. The refresh commands below reproduce that archived version
and would replace the newer figure.

## Refresh after editing

From the article directory (`27cvpr`):

```bash
/home/data2/chk/workspace/2026/.venv/bin/python ppt/scripts/fig1_v6/fit_arrows.py
cp ppt/ppt/fig1_v6.pdf imgs/fig1.pdf
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

`crop_ppt.py` trims the native slide canvas to its object bounds. `query_prompt.py` applies the Q prompt-box edit. Both operate on the existing deck. `build.py` recreates the original reference layout and replaces manual edits; it is retained as the original construction script.

Dependencies: python-pptx, Pillow, PyMuPDF, lxml, LibreOffice, and installed Comic Sans MS, Times New Roman, and Arial fonts.

## Backups

- `fig1_v6_before_arrow_fix.pptx`: the user's edited deck before arrow repairs.
- `fig1_v6_before_crop.pptx`: repaired deck before canvas cropping.
- `article_fig1_before_replacement.pdf`: the article's previous Figure 1.

The reference image remains at `../../source/fig1_v6.png`.
