"""Fit connectors in the manually edited deck, preserving all other slide objects."""
from pathlib import Path
from zipfile import ZipFile
import shutil
import subprocess
import tempfile
from lxml import etree as ET
import fitz

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / 'ppt/fig1_v6.pptx'
BACKUP = HERE / 'fig1_v6_before_arrow_fix.pptx'
NS = {'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
      'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
if not BACKUP.exists():
    shutil.copy2(OUT, BACKUP)
with ZipFile(OUT) as z:
    entries = [(info, z.read(info.filename)) for info in z.infolist()]
xml = dict((info.filename, data) for info, data in entries)['ppt/slides/slide1.xml']
root = ET.fromstring(xml)
shapes = root.find('p:cSld/p:spTree', NS)


def named(name):
    return [s for s in shapes if s.find('.//p:cNvPr', NS) is not None
            and s.find('.//p:cNvPr', NS).get('name') == name]


def bounds(s):
    t = s.find('p:spPr/a:xfrm', NS)
    off, ext = t.find('a:off', NS), t.find('a:ext', NS)
    return tuple(int(v) for v in (off.get('x'), off.get('y'), ext.get('cx'), ext.get('cy')))


def segment(s, start, end):
    t = s.find('p:spPr/a:xfrm', NS)
    x1, y1 = start
    x2, y2 = end
    t.attrib.pop('flipH', None)
    t.attrib.pop('flipV', None)
    if x2 < x1:
        t.set('flipH', '1')
    if y2 < y1:
        t.set('flipV', '1')
    t.find('a:off', NS).attrib.update({'x': str(min(x1, x2)), 'y': str(min(y1, y2))})
    t.find('a:ext', NS).attrib.update({'cx': str(abs(x2-x1)), 'cy': str(abs(y2-y1))})


# The blue route meets the decoder's flat top edge, clear of its rounded corner.
dx, dy, dw, dh = bounds(named('Task decoder')[0])
blue = named('Current spatial features to spatial task decoder')
x, y, _, _ = bounds(blue[2])
port = dx + round(26 * 7620)
segment(blue[2], (x, y), (port, y))
segment(blue[3], (port, y), (port, dy))
# Rejoin the upper output elbow and terminate both output arrows at the panel.
px, _, _, _ = bounds(named('Task loss panel')[0])
top = named('Task decoder to top task outputs')
x, y, w, _ = bounds(top[0])
_, ty, _, _ = bounds(top[2])
segment(top[1], (x+w, y), (x+w, ty))
segment(top[2], (x+w, ty), (px, ty))
lower = named('Task decoder to lower task outputs')[0]
x, y, _, _ = bounds(lower)
segment(lower, (x, y), (px, y))
# Close the one-pixel query-to-decoder gap.
query = named('Task query to task decoder')[0]
x, y, _, _ = bounds(query)
segment(query, (x, y), (x, dy))

with tempfile.TemporaryDirectory(prefix='fig1_v6_fit_') as work:
    work = Path(work)
    updated = work / OUT.name
    with ZipFile(updated, 'w') as z:
        for info, data in entries:
            z.writestr(info, ET.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
                       if info.filename == 'ppt/slides/slide1.xml' else data)
    shutil.copy2(updated, OUT)
    subprocess.run(['libreoffice', f'-env:UserInstallation={(work / "profile").as_uri()}',
                    '--headless', '--convert-to', 'pdf:impress_pdf_Export',
                    '--outdir', str(work), str(OUT)], check=True, capture_output=True)
    with fitz.open(work / 'fig1_v6.pdf') as source:
        page = source[0]
        # Tight vector exports; leave the user's PowerPoint canvas untouched.
        clip = fitz.Rect()
        for path in page.get_drawings():
            if path['fill'] == (1, 1, 1) and path['rect'].get_area() > .99 * page.rect.get_area():
                continue  # Ignore the full-slide white background.
            clip |= path['rect']
        clip = (clip + (-2, -2, 2, 2)) & page.rect
        with fitz.open() as pdf:
            target = pdf.new_page(width=clip.width, height=clip.height)
            target.show_pdf_page(target.rect, source, 0, clip=clip)
            pdf.save(OUT.with_suffix('.pdf'), garbage=4, deflate=True)
            OUT.with_suffix('.svg').write_text(target.get_svg_image(text_as_path=True))
            target.get_pixmap(matrix=fitz.Matrix(4200/target.rect.width, 4200/target.rect.width)).save(OUT.with_suffix('.png'))
print(OUT)
