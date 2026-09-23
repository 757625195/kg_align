from pathlib import Path
from zipfile import ZipFile
from lxml import etree
import sys

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}


def q(local):
    return f"{{{W}}}{local}"


def val(el, name="val"):
    return None if el is None else el.get(q(name))


path = Path(sys.argv[1])
with ZipFile(path) as z:
    styles = etree.fromstring(z.read("word/styles.xml"))
    doc = etree.fromstring(z.read("word/document.xml"))

print(f"FILE {path}")
print("STYLES")
for st in styles.xpath("./w:style[@w:type='paragraph']", namespaces=NS):
    sid = st.get(q("styleId"))
    name = val(st.find("w:name", NS))
    based = val(st.find("w:basedOn", NS))
    nxt = val(st.find("w:next", NS))
    ppr = st.find("w:pPr", NS)
    rpr = st.find("w:rPr", NS)
    size = val(rpr.find("w:sz", NS)) if rpr is not None else None
    font = None
    if rpr is not None and rpr.find("w:rFonts", NS) is not None:
        font = rpr.find("w:rFonts", NS).get(q("ascii"))
    bold = rpr is not None and rpr.find("w:b", NS) is not None
    italic = rpr is not None and rpr.find("w:i", NS) is not None
    jc = val(ppr.find("w:jc", NS)) if ppr is not None else None
    sp = ppr.find("w:spacing", NS) if ppr is not None else None
    spacing = None if sp is None else {k.split('}')[-1]: v for k, v in sp.attrib.items()}
    ind = ppr.find("w:ind", NS) if ppr is not None else None
    indent = None if ind is None else {k.split('}')[-1]: v for k, v in ind.attrib.items()}
    print(sid, repr(name), "based", based, "next", nxt, "font", font, "halfpt", size,
          "b", bold, "i", italic, "jc", jc, "spacing", spacing, "ind", indent)

print("PARAGRAPHS")
for i, p in enumerate(doc.xpath(".//w:body/w:p", namespaces=NS)):
    text = "".join(p.xpath(".//w:t/text()", namespaces=NS)).strip().replace("\n", " ")
    pstyle = p.find("w:pPr/w:pStyle", NS)
    sid = val(pstyle)
    if text:
        print(i, sid, repr(text[:140]))
