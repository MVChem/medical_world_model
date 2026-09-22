# Introduction concept figure

Recreates the supplied reference as a single editable slide, keeping its 2048:762 aspect ratio. Used by `sections/1_introduction.tex`.

From this directory:

```sh
python -m pip install python-pptx Pillow
python build.py
libreoffice --headless --convert-to pdf --outdir ../../ppt ../../ppt/intro_concept.pptx
pdftoppm -scale-to 2048 -png -singlefile ../../ppt/intro_concept.pdf ../../ppt/intro_concept
```

Paths are relative to this script; assets are included. The manuscript build can regenerate the local PDF from the tracked PPTX. PDFs are not committed.

Text, state subscripts, gradients, cloud, nodes, trajectories and arrows are native PowerPoint objects. Icon components have shallow named groups; labels and task arrows remain independently editable. Use the Selection Pane and Edit Points to change components and curves. Arrowheads follow their paths. Only the two radiographs are cropped bitmaps. The full reference is retained for rebuilding but is not placed in the slide. There are no embedded SVGs.

Fonts are approximated using DejaVu Sans and STIX Two Math. Cloud shading, lung heatmap and state gradients are simplified; trajectories are schematic, not recovered data. Moving an icon does not move its separate label or task arrow. The preview is rendered from the final PPTX using LibreOffice.
