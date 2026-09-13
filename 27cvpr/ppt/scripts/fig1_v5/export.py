"""Export the manually edited PowerPoint without rebuilding it."""
from pathlib import Path
import subprocess
import tempfile

import fitz
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "ppt/fig1_v5.pptx"
with tempfile.TemporaryDirectory(prefix="fig1_v5_export_") as work:
    work = Path(work)
    subprocess.run([
        "libreoffice", f"-env:UserInstallation={(work / 'profile').as_uri()}",
        "--headless", "--convert-to", "pdf:impress_pdf_Export",
        "--outdir", str(work), str(SOURCE),
    ], check=True, capture_output=True)
    with fitz.open(work / "fig1_v5.pdf") as original:
        assert len(original) == 1, "Expected a single figure slide"
        page = original[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
        im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        diff = ImageChops.difference(im, Image.new("RGB", im.size, "white"))
        mask = diff.convert("L").point(lambda v: 255 if v > 5 else 0)
        bounds = mask.getbbox()
        assert bounds, "Exported slide is blank"
        crop = fitz.Rect([v / 3 for v in bounds])
        crop = (crop + (-2, -2, 2, 2)) & page.rect
        # Reframe vector PDF content, retaining a small 2-point safety margin.
        with fitz.open() as tight:
            target = tight.new_page(width=crop.width, height=crop.height)
            target.show_pdf_page(target.rect, original, 0, clip=crop)
            tight.save(SOURCE.with_suffix(".pdf"), garbage=4, deflate=True)
            SOURCE.with_suffix(".svg").write_text(target.get_svg_image(text_as_path=True))
            target.get_pixmap(matrix=fitz.Matrix(3, 3)).save(SOURCE.with_suffix(".png"))
        print(f"Cropped {page.rect.width:.1f} × {page.rect.height:.1f} pt "
              f"to {crop.width:.1f} × {crop.height:.1f} pt")
print(SOURCE.with_suffix(".pdf"))
