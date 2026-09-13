# fig1_v5

To export the **manually edited PowerPoint**, run `export.py` with the workspace Python. It updates the PDF, SVG, and PNG with a tight crop and a 2-point margin, without modifying the PowerPoint. Do not run `build.py` to export manual edits: it rebuilds the slide from the original reference.

Recreates `../../source/fig1_v5.png`, using the v4 script's native PowerPoint drawing approach and Comic Sans MS font.

```bash
/home/data2/chk/workspace/2026/.venv/bin/python /home/data2/chk/workspace/2026/08/04/medical_world_model/27cvpr/ppt/scripts/fig1_v5/build.py
```

Outputs in `../../ppt/`: `fig1_v5.pptx`, `fig1_v5.svg`, `fig1_v5.pdf`, and `fig1_v5.png`.

The single slide preserves the reference's 1864 × 843 layout, colors, labels, and equations. Boxes, tokens, arrows, chart bars, document icons, and text are editable PowerPoint objects. Only five medical thumbnails are cropped raster images from the supplied reference; their detail is limited by that source.

SVG and PDF retain vector artwork. SVG text is outlined for portable paper submission; PowerPoint text remains editable. The PNG is a 3× PDF preview. No additional canvas margins are added.

Requires `python-pptx`, Pillow, PyMuPDF, LibreOffice, and Comic Sans MS, available in the workspace.
