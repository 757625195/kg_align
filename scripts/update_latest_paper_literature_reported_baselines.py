from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/Users/auroralin/Downloads") / (
    "关系感知结构上下文与多尺度语义融合的跨语言知识图谱实体对齐_"
    "术语与近期文献修订稿_20260827.docx"
)
OUTPUT = ROOT / "outputs" / (
    "关系感知结构上下文与多尺度语义融合的跨语言知识图谱实体对齐_"
    "文献报告值修订版_20260828.docx"
)


def find_paragraph(document: Document, prefix: str):
    matches = [
        paragraph
        for paragraph in document.paragraphs
        if paragraph.text.strip().startswith(prefix)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one paragraph beginning {prefix!r}, found {len(matches)}"
        )
    return matches[0]


def replace_text(paragraph, text: str) -> None:
    for child in list(paragraph._p):
        if child.tag != qn("w:pPr"):
            paragraph._p.remove(child)
    run = paragraph.add_run(text)
    run.font.name = "宋体"
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")


def format_table(document: Document) -> None:
    table = document.tables[2]
    if (len(table.rows), len(table.columns)) != (14, 5):
        raise RuntimeError("Unexpected comparison table shape")

    for row_index, row in enumerate(table.rows):
        row._tr.get_or_add_trPr().append(
            __import__("docx").oxml.OxmlElement("w:cantSplit")
        )
        for column_index, cell in enumerate(row.cells):
            paragraph = cell.paragraphs[0]
            paragraph.alignment = 0 if column_index < 2 else 1
            for run in paragraph.runs:
                run.font.name = "Times New Roman"
                run.font.size = Pt(9)
                run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")
        if row_index == 0:
            row._tr.get_or_add_trPr().append(
                __import__("docx").oxml.OxmlElement("w:tblHeader")
            )


def validate(document: Document) -> None:
    body = "\n".join(paragraph.text for paragraph in document.paragraphs)
    required_once = [
        "5.2 与文献报告结果的定位性比较",
        "表 3 文献报告结果与本文模型的定位性比较",
        "不构成统一协议下的严格排名",
        "DBP15K 的既有方法未统一报告 MRR",
    ]
    for phrase in required_once:
        if body.count(phrase) != 1:
            raise RuntimeError(f"Expected one occurrence of {phrase!r}")

    table = document.tables[2]
    if (len(table.rows), len(table.columns)) != (14, 5):
        raise RuntimeError("Comparison table shape changed")
    table_text = "\n".join(cell.text for row in table.rows for cell in row.cells)
    for value in (
        "0.3083",
        "0.4118",
        "0.6294",
        "0.7075",
        "0.8220",
        "0.2404",
        "0.6599",
        "0.8469",
        "0.7295",
        "0.6324",
    ):
        if value not in table_text:
            raise RuntimeError(f"Missing literature comparison value {value}")


def main() -> None:
    document = Document(SOURCE)

    replace_text(
        find_paragraph(document, "5.2 与经典方法的定位性比较"),
        "5.2 与文献报告结果的定位性比较",
    )
    replace_text(
        find_paragraph(document, "表 3 中 DBP15K"),
        "表 3 采用既有文献公开报告的结果，而非本地重跑结果。DBP15K ZH–EN 中，"
        "MTransE、JAPE、BootEA 和 RDGCN 的数值取自 RDGCN 的文献汇总[5]，"
        "RREA-text 取自 RREA 原论文的文本增强设置[6]；OpenEA EN–FR-V2 的数值取自"
        "OpenEA 官方基准的五折平均[8]。本文模型在 DBP15K 上报告三个随机种子的均值，"
        "在 OpenEA 上报告官方第一折的三个随机种子均值。由于各来源在数据划分、候选实体范围、"
        "名称与属性输入、模型变体和重复运行方式上并不完全一致，表 3 只用于定位性能区间，"
        "不构成统一协议下的严格排名。",
    )
    caption = find_paragraph(document, "表 3 与经典实体对齐方法的定位性比较")
    replace_text(caption, "表 3 文献报告结果与本文模型的定位性比较")
    caption.paragraph_format.keep_with_next = True

    replace_text(
        find_paragraph(document, "在 DBP15K ZH–EN 上，本文模型的 Hits@1"),
        "在上述定位性比较中，本文模型在 DBP15K ZH–EN 上的 Hits@1 为 0.7295，"
        "高于 MTransE、JAPE、BootEA 和 RDGCN 的文献报告值，低于文本增强的 RREA-text；"
        "Hits@10 为 0.8490，略高于 BootEA 和 RDGCN，低于 RREA-text。DBP15K 的既有方法"
        "未统一报告 MRR，因而不比较该指标。OpenEA EN–FR-V2 上，本文模型的 Hits@1、"
        "Hits@10 和 MRR 分别为 0.6324、0.8196 和 0.6978，高于 MTransE、JAPE 和 GCN-Align，"
        "但低于 BootEA、KDCoE 和 RDGCN。该结果说明本文模型达到可用的实体检索水平，"
        "其主要实验证据仍应来自同一协议下的多随机种子结果、组件消融和受控基线，而不是跨文献名次。",
    )

    format_table(document)
    validate(document)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT)

    reopened = Document(OUTPUT)
    validate(reopened)
    print(OUTPUT)


if __name__ == "__main__":
    main()
