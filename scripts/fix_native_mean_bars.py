from copy import deepcopy
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from docx import Document
from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / (
    "关系感知结构上下文与多尺度语义融合的跨语言知识图谱实体对齐_"
    "Springer投稿删减版_20260826.docx"
)
OUTPUT = ROOT / "outputs" / (
    "关系感知结构上下文与多尺度语义融合的跨语言知识图谱实体对齐_"
    "Springer投稿删减版_均值符号修正_20260826.docx"
)

MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
NS = {"m": MATH_NS}
M_VAL = f"{{{MATH_NS}}}val"


def replace_detached_overbars(root: etree._Element) -> int:
    replaced = 0
    for group in list(root.xpath(".//m:groupChr", namespaces=NS)):
        char = group.find("./m:groupChrPr/m:chr", NS)
        pos = group.find("./m:groupChrPr/m:pos", NS)
        if char is None or char.get(M_VAL) != "¯":
            continue
        if pos is not None and pos.get(M_VAL) != "top":
            continue

        expression = group.find("./m:e", NS)
        if expression is None:
            raise RuntimeError("Mean overbar has no expression")

        bar = etree.Element(f"{{{MATH_NS}}}bar")
        bar_properties = etree.SubElement(bar, f"{{{MATH_NS}}}barPr")
        bar_position = etree.SubElement(bar_properties, f"{{{MATH_NS}}}pos")
        bar_position.set(M_VAL, "top")

        control = group.find("./m:groupChrPr/m:ctrlPr", NS)
        if control is not None:
            bar_properties.append(deepcopy(control))
        bar.append(deepcopy(expression))

        group.getparent().replace(group, bar)
        replaced += 1
    return replaced


def main() -> None:
    with ZipFile(SOURCE) as source_zip:
        document_xml = source_zip.read("word/document.xml")
        root = etree.fromstring(document_xml)
        replaced = replace_detached_overbars(root)
        if replaced != 6:
            raise RuntimeError(f"Expected to replace 6 detached mean bars, replaced {replaced}")

        updated_xml = etree.tostring(
            root,
            xml_declaration=True,
            encoding="UTF-8",
            standalone=True,
        )
        with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED) as output_zip:
            for entry in source_zip.infolist():
                payload = updated_xml if entry.filename == "word/document.xml" else source_zip.read(entry)
                output_zip.writestr(entry, payload)

    with ZipFile(OUTPUT) as result_zip:
        if result_zip.testzip() is not None:
            raise RuntimeError("Generated DOCX archive is corrupt")
        result_root = etree.fromstring(result_zip.read("word/document.xml"))
        detached = result_root.xpath(
            ".//m:groupChr[m:groupChrPr/m:chr[@m:val='¯']]",
            namespaces=NS,
        )
        remaining_tildes = result_root.xpath(
            ".//m:groupChr[m:groupChrPr/m:chr[@m:val='~']]",
            namespaces=NS,
        )
        native_bars = result_root.xpath(".//m:bar", namespaces=NS)
        if detached or len(remaining_tildes) != 2 or len(native_bars) != 13:
            raise RuntimeError(
                "Unexpected equation structure after mean-bar conversion: "
                f"detached={len(detached)}, tildes={len(remaining_tildes)}, "
                f"native_bars={len(native_bars)}"
            )

    Document(OUTPUT)
    print(f"WROTE {OUTPUT}")
    print("converted_mean_bars=6 remaining_tildes=2 native_bars=13")


if __name__ == "__main__":
    main()
