"""Trim the editable slide canvas to the object bounds with a three-pixel margin."""
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as ET
import shutil

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / 'ppt/fig1_v6.pptx'
BACKUP = HERE / 'fig1_v6_before_crop.pptx'
NS = {'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
      'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
if not BACKUP.exists():
    shutil.copy2(OUT, BACKUP)
with ZipFile(OUT) as z:
    entries = [(info, z.read(info.filename)) for info in z.infolist()]
data = {info.filename: value for info, value in entries}
slide = ET.fromstring(data['ppt/slides/slide1.xml'])
pres = ET.fromstring(data['ppt/presentation.xml'])
transforms = slide.findall('p:cSld/p:spTree/*/p:spPr/a:xfrm', NS)
coords = [(t.find('a:off', NS), t.find('a:ext', NS)) for t in transforms]
size = pres.find('p:sldSz', NS)
margin = 3 * 7620
left = max(0, min(int(o.get('x')) for o, e in coords)-margin)
top = max(0, min(int(o.get('y')) for o, e in coords)-margin)
right = min(int(size.get('cx')), max(int(o.get('x'))+int(e.get('cx')) for o,e in coords)+margin)
bottom = min(int(size.get('cy')), max(int(o.get('y'))+int(e.get('cy')) for o,e in coords)+margin)
for off, ext in coords:
    off.set('x', str(int(off.get('x'))-left))
    off.set('y', str(int(off.get('y'))-top))
size.set('cx', str(right-left))
size.set('cy', str(bottom-top))
for name, root in [('ppt/slides/slide1.xml', slide), ('ppt/presentation.xml', pres)]:
    data[name] = ET.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
with ZipFile(OUT, 'w') as z:
    for info, value in entries:
        z.writestr(info, data[info.filename])
print(f'Canvas: {(right-left)/7620:.1f} × {(bottom-top)/7620:.1f} px; all shapes remain editable.')
