from pathlib import Path

from docx import Document
from docx.oxml.ns import qn


ROOT = Path(__file__).resolve().parents[1]
JOBS = (
    (
        ROOT
        / "outputs"
        / "Relation-Aware Neighbor Context and Semantics-Guided Selection for Knowledge Graph Entity Alignment.docx",
        ROOT
        / "outputs"
        / "Relation-Aware Neighbor Context and Semantics-Guided Selection for Knowledge Graph Entity Alignment_格式统一版.docx",
    ),
    (
        ROOT / "outputs" / "1_springer_proceedings_中文版.docx",
        ROOT / "outputs" / "1_springer_proceedings_中文版_格式统一版.docx",
    ),
)


def clear_direct_heading_format(paragraph) -> None:
    """Make the heading inherit its complete layout from Heading 2."""
    paragraph.style = "Heading 2"
    p_pr = paragraph._p.get_or_add_pPr()
    keep = {qn("w:pStyle"), qn("w:numPr"), qn("w:sectPr")}
    for child in list(p_pr):
        if child.tag not in keep:
            p_pr.remove(child)


def normalize_document(source: Path, output: Path) -> None:
    document = Document(source)
    found = set()
    for paragraph in document.paragraphs:
        text = paragraph.text.replace("\u2060", "").strip()
        for number in ("4.2", "4.3"):
            if text.startswith(number + " ") or text.startswith(number + "\t"):
                clear_direct_heading_format(paragraph)
                found.add(number)
                break

    if found != {"4.2", "4.3"}:
        raise RuntimeError(f"Expected sections 4.2 and 4.3 in {source}; found {sorted(found)}")

    document.save(output)
    print(output)


if __name__ == "__main__":
    for input_path, output_path in JOBS:
        normalize_document(input_path, output_path)
