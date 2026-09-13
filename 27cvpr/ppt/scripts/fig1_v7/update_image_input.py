"""Validate the direct-image route and refresh exports of the editable v7 deck.

The v7 PowerPoint is the source of truth. Never rebuild it from v6: doing so
would discard any later manual edits. This script leaves the PPTX untouched.
"""
from pathlib import Path
from zipfile import ZipFile
import shutil
import subprocess
import tempfile

import fitz
from lxml import etree as ET

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "ppt/ppt/fig1_v7.pptx"
NS = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}


def bounds(shape):
    transform = shape.find("p:spPr/a:xfrm", NS)
    offset = transform.find("a:off", NS)
    extent = transform.find("a:ext", NS)
    return tuple(int(value) for value in (
        offset.get("x"), offset.get("y"), extent.get("cx"), extent.get("cy")
    ))


with ZipFile(SOURCE) as archive:
    root = ET.fromstring(archive.read("ppt/slides/slide1.xml"))
shapes = root.find("p:cSld/p:spTree", NS)


def named(name):
    return [shape for shape in shapes
            if shape.find(".//p:cNvPr", NS) is not None
            and shape.find(".//p:cNvPr", NS).get("name") == name]


route = named("Input image to spatial task decoder")
assert len(route) == 3, "Expected the three-segment direct image route"
assert all(shape.tag == f"{{{NS['p']}}}cxnSp" for shape in route), "Image route must remain editable connectors"
assert not named("Current spatial features to spatial task decoder"), "Remove the obsolete JEPA-to-decoder bypass"
ix, iy, iw, _ = bounds(named("Current chest radiograph")[0])
rx, ry, rw, rh = bounds(route[0])
assert rw == 0 and ix <= rx <= ix + iw and abs(ry + rh - iy) < 100, "Direct route must start at the input image"
dx, dy, dw, _ = bounds(named("Task decoder")[0])
rx, ry, rw, rh = bounds(route[-1])
assert rw == 0 and dx <= rx <= dx + dw and abs(ry + rh - dy) < 100, "Direct route must end at the task decoder"
labels = named("Equation: Xt (spatial input)")
assert len(labels) == 1, "Label the image branch once with X_t"
runs = labels[0].findall(".//a:r", NS)
assert "".join(labels[0].xpath(".//a:t/text()", namespaces=NS)) == "Xt", "The image branch label must contain only X_t"
assert any(
    run.findtext("a:t", namespaces=NS) == "t"
    and run.find("a:rPr", NS) is not None
    and int(run.find("a:rPr", NS).get("baseline", "0")) < 0
    for run in runs
), "Render t as a subscript in X_t"
assert not named("Label: direct image input for spatial tasks"), "Remove the trailing spatial-task and LR-input explanation"

with tempfile.TemporaryDirectory(prefix="fig1_v7_export_") as directory:
    work = Path(directory)
    subprocess.run([
        "libreoffice", f"-env:UserInstallation={(work / 'profile').as_uri()}",
        "--headless", "--convert-to", "pdf:impress_pdf_Export",
        "--outdir", str(work), str(SOURCE),
    ], check=True, capture_output=True)
    with fitz.open(work / "fig1_v7.pdf") as source:
        assert len(source) == 1
        page = source[0]
        clip = fitz.Rect()
        for drawing in page.get_drawings():
            if drawing["fill"] == (1, 1, 1) and drawing["rect"].get_area() > .99 * page.rect.get_area():
                continue
            clip |= drawing["rect"]
        clip = (clip + (-2, -2, 2, 2)) & page.rect
        with fitz.open() as exported:
            target = exported.new_page(width=clip.width, height=clip.height)
            target.show_pdf_page(target.rect, source, 0, clip=clip)
            exported.save(SOURCE.with_suffix(".pdf"), garbage=4, deflate=True)
            SOURCE.with_suffix(".svg").write_text(target.get_svg_image(text_as_path=True))
            scale = 4200 / target.rect.width
            target.get_pixmap(matrix=fitz.Matrix(scale, scale)).save(SOURCE.with_suffix(".png"))

shutil.copy2(SOURCE.with_suffix(".pdf"), ROOT / "imgs/fig1.pdf")
print(f"Validated unchanged PowerPoint source: {SOURCE}")
print(ROOT / "imgs/fig1.pdf")
