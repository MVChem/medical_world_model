# Figure 2: compact single-column slot construction

Redrawn from `../../ppt/fig2_v1.pptx` and its rendered preview for the
manuscript's 3.28125-inch column. The VLM panels are narrower, with four
colored depth samples and three short gray placeholders per source.
Gray blocks are schematic omitted layers, not a layer-count specification.
The two rows of four output slots, feature-source colors, adapter, decoder,
task loss, and joint-optimization annotation retain the original meaning.

The native slide is 236.25 × 98 points. Most labels are 7.2–9.5 points at
the final paper size. Comic Sans MS and Times New Roman match the previous
figure; both fonts must be installed for consistent rendering.

Dependencies: Python 3, `python-pptx`, `PyMuPDF`, and LibreOffice.
Run from any directory:

```bash
python path/to/fig2_v2/build.py --install
```

This creates `../../ppt/fig2_v2.pptx`, its PDF, and its PNG preview, then
copies the vector PDF to `../../../imgs/fig2_v2.pdf` for the paper.
Paths resolve relative to the script. The figure is drawn entirely from
native shapes; no external artwork is needed to rebuild it.

After manual PPT editing, export without rebuilding:

```bash
python path/to/fig2_v2/build.py --render-only --install
```

Compile the manuscript and supplement with `make -C 27cvpr` from the
project root. The paper includes Figure 2 using `figure` and `\columnwidth`.

All labels and the loss subscript are editable text. The source-layer and
slot rows form four small named groups; individual blocks remain editable
inside each group. Labels, module backgrounds, and each arrow are separate
objects, identifiable through the PowerPoint selection pane. Straight arrows
are native connectors; routed arrows and the merge bracket have editable
vertices. Arrowheads move with their lines. Module labels and connections do
not attach automatically to panel backgrounds, so move them together when
repositioning a module. There is no whole-slide group, embedded SVG, or bitmap.

The PNG is rendered from the final saved PPT via LibreOffice PDF export.
The vector PDF retains the compact slide canvas and publication-size text.
The previous `fig2_v1` files remain available as the original wide layout.
