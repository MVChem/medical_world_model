"""Replace the query token with an editable example text prompt."""
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as ET

OUT = Path(__file__).resolve().parents[2] / 'ppt/fig1_v6.pptx'
NS = {'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
      'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
with ZipFile(OUT) as z:
    entries = [(info, z.read(info.filename)) for info in z.infolist()]
xml = next(data for info,data in entries if info.filename == 'ppt/slides/slide1.xml')
root = ET.fromstring(xml)
shapes = root.find('p:cSld/p:spTree', NS)

def named(name):
    return next(s for s in shapes if s.find('.//p:cNvPr', NS) is not None
                and s.find('.//p:cNvPr', NS).get('name') in {name, {'Task query token': 'Task query prompt', 'Equation: q': 'Equation: Q'}.get(name)})

prompt = named('Task query token')
pr = prompt.find('p:spPr', NS)
t = pr.find('a:xfrm', NS)
y = int(t.find('a:off', NS).get('y'))
t.find('a:off', NS).set('x', str(1305*7620))
t.find('a:ext', NS).set('cx', str(170*7620))
t.find('a:ext', NS).set('cy', str(44*7620))
prompt.find('.//p:cNvPr', NS).set('name', 'Task query prompt')
for fill in list(pr):
    if ET.QName(fill).localname in ['gradFill', 'solidFill', 'noFill']:
        pr.remove(fill)
fill = ET.SubElement(pr, '{'+NS['a']+'}solidFill')
ET.SubElement(fill, '{'+NS['a']+'}srgbClr', val='F3FAFE')
pr.insert(list(pr).index(pr.find('a:ln', NS)), fill)
body = prompt.find('p:txBody', NS)
if body is not None:
    prompt.remove(body)
body = ET.fromstring(f'''<p:txBody xmlns:p="{NS['p']}" xmlns:a="{NS['a']}">
<a:bodyPr wrap="none" anchor="ctr" lIns="30480" rIns="30480" tIns="0" bIns="0"/>
<a:lstStyle/><a:p><a:pPr algn="ctr"/><a:r><a:rPr lang="en-US" sz="930"><a:solidFill><a:srgbClr val="171717"/></a:solidFill><a:latin typeface="Arial"/></a:rPr><a:t>Segment the lungs.</a:t></a:r><a:endParaRPr lang="en-US"/></a:p></p:txBody>''')
prompt.append(body)
q = named('Equation: q')
q.find('.//a:t', NS).text = 'Q'
q.find('p:spPr/a:xfrm/a:off', NS).set('x', str(1448*7620))
q.find('.//p:cNvPr', NS).set('name', 'Equation: Q')
# Position the arrow below the prompt, ending on the decoder top edge.
arrow = named('Task query to task decoder').find('p:spPr/a:xfrm', NS)
decoder = named('Task decoder').find('p:spPr/a:xfrm/a:off', NS)
arrow.find('a:off', NS).set('x', str(1390*7620))
arrow.find('a:off', NS).set('y', str(y+44*7620))
arrow.find('a:ext', NS).set('cy', str(int(decoder.get('y'))-y-44*7620))
with ZipFile(OUT, 'w') as z:
    for info, data in entries:
        z.writestr(info, ET.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
                   if info.filename == 'ppt/slides/slide1.xml' else data)
print(OUT)
