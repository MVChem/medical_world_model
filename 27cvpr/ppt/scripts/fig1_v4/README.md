# fig1_v4

Rebuild from this directory:

```bash
/home/data2/chk/workspace/2026/.venv/bin/python build.py
```

Outputs: `../../ppt/fig1_v4.pptx`, `.svg`, `.pdf`, and `.png`.

The diagram uses native editable PowerPoint shapes, connectors, and Comic Sans MS text. The VLM transformer blocks are omitted; the task head is a single native trapezoid. The input strip and patient state each contain eight pink slots. Shared image features use local ports, and both future-token columns feed the prediction loss.

Only the two study thumbnails and segmentation thumbnail are raster crops from `../../source/fig1_v4.png`. SVG and PDF preserve the remaining artwork as vectors. SVG text is outlined for portable paper export. The canvas trims the reference's wide top and bottom margins.

Requires `python-pptx`, Pillow, PyMuPDF, LibreOffice, and Comic Sans MS (available in this workspace).
