from __future__ import annotations

import copy
import csv
import html.entities
import json
import math
import re
import statistics
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Pt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "1.docx"
OUTPUT = ROOT / "outputs" / "1_按修稿优先级修订版_20260831.docx"
MATH_DEPS = ROOT / "tmp" / "docx_math_deps"
DIRECT_RUNS = (
    ROOT
    / "outputs"
    / "methodology_controls_20260829"
    / "summary_existing"
    / "direct_joint_runs.tsv"
)
DIRECT_AGGREGATE = (
    ROOT
    / "outputs"
    / "methodology_controls_20260829"
    / "summary_existing"
    / "direct_joint_aggregate.tsv"
)
SOFTMAX_RUNS = (
    ROOT
    / "outputs"
    / "revision_all_neighbor_softmax_20260831"
    / "all_neighbor_softmax"
    / "summary_direct"
    / "test_runs.tsv"
)
SOFTMAX_AGGREGATE = (
    ROOT
    / "outputs"
    / "revision_all_neighbor_softmax_20260831"
    / "all_neighbor_softmax"
    / "summary_direct"
    / "test_aggregate.tsv"
)
RREA_ROOT = ROOT / "outputs" / "unified_existing_baselines_20260826" / "rrea_basic"

sys.path.insert(0, str(MATH_DEPS))
import mathml2omml  # noqa: E402
from latex2mathml.converter import convert as latex_to_mathml  # noqa: E402


def normalize(text: str) -> str:
    return " ".join(text.replace("\u00a0", " ").split())


def paragraph_text(paragraph) -> str:
    return normalize("".join(paragraph._p.xpath(".//w:t/text()|.//m:t/text()")))


def find_paragraph(document: Document, prefix: str):
    for paragraph in document.paragraphs:
        if paragraph_text(paragraph).startswith(prefix):
            return paragraph
    raise RuntimeError(f"Paragraph not found: {prefix!r}")


def clear_paragraph(paragraph) -> None:
    for child in list(paragraph._p):
        if child.tag != qn("w:pPr"):
            paragraph._p.remove(child)


def set_run_font(run, size: float | None = None, bold: bool | None = None) -> None:
    run.font.name = "Times New Roman"
    fonts = run._element.get_or_add_rPr().get_or_add_rFonts()
    fonts.set(qn("w:ascii"), "Times New Roman")
    fonts.set(qn("w:hAnsi"), "Times New Roman")
    fonts.set(qn("w:eastAsia"), "Times New Roman")
    fonts.set(qn("w:cs"), "Times New Roman")
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold


def latex_to_omml(expression: str):
    mathml = latex_to_mathml(expression.strip(), display="inline")
    omml = mathml2omml.convert(mathml, html.entities.name2codepoint)
    omml = re.sub(
        r"(<m:groupChr><m:groupChrPr>.*?)</m:groupChr>(<m:e>)",
        r"\1</m:groupChrPr>\2",
        omml,
    )
    omml = omml.replace("<m:oMath>", f"<m:oMath {nsdecls('m')}>", 1)
    return parse_xml(omml)


def add_mixed(paragraph, text: str, size: float | None = None) -> None:
    cursor = 0
    for match in re.finditer(r"\$([^$]+)\$", text):
        if match.start() > cursor:
            run = paragraph.add_run(text[cursor : match.start()])
            set_run_font(run, size=size)
        paragraph._p.append(latex_to_omml(match.group(1)))
        cursor = match.end()
    if cursor < len(text):
        run = paragraph.add_run(text[cursor:])
        set_run_font(run, size=size)


def set_mixed(paragraph, text: str) -> None:
    clear_paragraph(paragraph)
    add_mixed(paragraph, text)


def set_plain(paragraph, text: str) -> None:
    clear_paragraph(paragraph)
    run = paragraph.add_run(text)
    set_run_font(run)


def insert_paragraph_before(document: Document, anchor, text: str, style: str = "Normal"):
    paragraph = document.add_paragraph(style=style)
    add_mixed(paragraph, text)
    anchor._p.addprevious(paragraph._p)
    return paragraph


def replace_display_equation(paragraph, expression: str, tag: str) -> None:
    clear_paragraph(paragraph)
    paragraph.style = "Formula"
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph._p.append(latex_to_omml(expression))
    run = paragraph.add_run(f"\u2003\u2003({tag})")
    set_run_font(run)


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def metric_pair(row: dict[str, str]) -> str:
    prefix = "test_" if "test_Hits@1_mean" in row else ""
    return (
        f"{float(row[f'{prefix}Hits@1_mean']):.4f} ± {float(row[f'{prefix}Hits@1_std']):.4f} / "
        f"{float(row[f'{prefix}MRR_mean']):.4f} ± {float(row[f'{prefix}MRR_std']):.4f}"
    )


def sample_mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def load_rrea() -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for dataset in ("dbp15k_zh_en", "openea_en_fr"):
        for seed in (42, 43, 44):
            path = RREA_ROOT / f"{dataset}_seed{seed}.json"
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                grouped.setdefault(dataset, []).append(payload)
    result: dict[str, dict[str, object]] = {}
    for dataset in ("dbp15k_zh_en", "openea_en_fr"):
        payloads = grouped.get(dataset, [])
        if not payloads:
            raise FileNotFoundError(f"No RREA result found for {dataset} in {RREA_ROOT}")
        result[dataset] = {
            "runs": payloads,
            "seeds": [int(payload["seed"]) for payload in payloads],
            **{
                metric: sample_mean_std(
                    [float(payload["test"][metric]) for payload in payloads]
                )
                for metric in ("Hits@1", "Hits@10", "MRR")
            },
        }
    return result


def set_cell(cell, text: str, *, bold: bool = False, size: float = 9.0, align=None) -> None:
    paragraph = cell.paragraphs[0]
    clear_paragraph(paragraph)
    run = paragraph.add_run(text)
    set_run_font(run, size=size, bold=bold)
    if align is not None:
        paragraph.alignment = align
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.line_spacing = 1.0
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_cell_mixed(cell, text: str, *, size: float = 9.0, align=None) -> None:
    paragraph = cell.paragraphs[0]
    clear_paragraph(paragraph)
    add_mixed(paragraph, text, size=size)
    if align is not None:
        paragraph.alignment = align
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.line_spacing = 1.0
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_cell_margins(cell, top=80, start=100, bottom=80, end=100) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for edge, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        element = tc_mar.find(qn(f"w:{edge}"))
        if element is None:
            element = OxmlElement(f"w:{edge}")
            tc_mar.append(element)
        element.set(qn("w:w"), str(value))
        element.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths: list[int]) -> None:
    table.autofit = False
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths)))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), "100")
    tbl_ind.set(qn("w:type"), "dxa")
    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(width))
            tc_w.set(qn("w:type"), "dxa")
            set_cell_margins(cell)
    set_table_pagination(table)


def set_table_pagination(table) -> None:
    for row in table.rows:
        row_pr = row._tr.get_or_add_trPr()
        if row_pr.find(qn("w:cantSplit")) is None:
            row_pr.append(OxmlElement("w:cantSplit"))
    header_pr = table.rows[0]._tr.get_or_add_trPr()
    if header_pr.find(qn("w:tblHeader")) is None:
        repeat = OxmlElement("w:tblHeader")
        repeat.set(qn("w:val"), "true")
        header_pr.append(repeat)


def set_cell_no_wrap(cell) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    if tc_pr.find(qn("w:noWrap")) is None:
        tc_pr.append(OxmlElement("w:noWrap"))


def create_table_before(
    document: Document,
    anchor,
    headers: list[str],
    rows: list[list[str]],
    widths: list[int],
    style,
    font_size: float = 9.0,
):
    table = document.add_table(rows=1, cols=len(headers))
    table.style = style
    for index, value in enumerate(headers):
        set_cell(
            table.cell(0, index),
            value,
            bold=True,
            size=font_size,
            align=WD_ALIGN_PARAGRAPH.CENTER,
        )
    for values in rows:
        row = table.add_row()
        for index, value in enumerate(values):
            alignment = WD_ALIGN_PARAGRAPH.LEFT if index in {0, 1} else WD_ALIGN_PARAGRAPH.CENTER
            set_cell(row.cells[index], value, size=font_size, align=alignment)
    set_table_geometry(table, widths)
    anchor._p.addprevious(table._tbl)
    return table


def replace_table(
    document: Document,
    old_table,
    headers: list[str],
    rows: list[list[str]],
    widths: list[int],
    font_size: float = 9.0,
):
    anchor = document.add_paragraph()
    old_table._tbl.addnext(anchor._p)
    new_table = create_table_before(
        document,
        anchor,
        headers,
        rows,
        widths,
        old_table.style,
        font_size=font_size,
    )
    old_table._tbl.getparent().remove(old_table._tbl)
    anchor._p.getparent().remove(anchor._p)
    return new_table


def fmt_mean_std(value: tuple[float, float]) -> str:
    return f"{value[0]:.4f} ± {value[1]:.4f}"


def fmt_baseline_metric(value: tuple[float, float], runs: int) -> str:
    return fmt_mean_std(value) if runs > 1 else f"{value[0]:.4f}"


def row_index(rows: list[dict[str, str]], variant: str, dataset: str) -> dict[str, str]:
    return next(
        row
        for row in rows
        if row.get("variant") == variant and row.get("dataset") == dataset
    )


def revise_prose(document: Document) -> None:
    set_plain(
        find_paragraph(document, "Neighbor selection must account"),
        "Neighbor-context construction must preserve relation type and identify relevant evidence. Two people may both connect to Shanghai, but a birthplace edge and "
        "a workplace edge describe different facts [5, 7]. Other adjacent entities may add little evidence for the current match. The proposed model therefore preserves "
        "relation types, uses entity semantics to condition structural-neighbor weighting, and combines the resulting context with the semantic representation.",
    )
    set_plain(
        find_paragraph(document, "This study addresses three questions"),
        "This study addresses three questions. Does relation-aware one-hop neighbor context add alignment evidence beyond the semantic representation? When the "
        "neighborhood and trainable fusion capacity are fixed, how does semantic conditioning compare with structural conditioning during neighbor scoring and context "
        "gating? Which structural and semantic components account for the observed joint-model performance?",
    )
    set_plain(
        find_paragraph(document, "Knowledge graph entity alignment identifies"),
        "Knowledge graph entity alignment identifies entities that refer to the same object across independently constructed graphs. "
        "This paper presents a relation-aware neighbor-context model in which entity semantics conditions structural context construction before joint encoding. "
        "The structural encoder preserves relation types during message passing and combines states from different propagation depths. The semantic encoder "
        "extracts token, phrase, and global views from one entity-text sequence. The fusion module scores the complete one-hop outgoing neighborhood, applies "
        "1.5-entmax to suppress weak neighbors, and combines the resulting structural context with entity semantics. Experiments use three random seeds on five "
        "datasets, while controlled component studies use DBP15K ZH–EN and OpenEA EN–FR-15K-V2. The full model reaches Hits@1 values of 0.7285 and 0.6344 on "
        "these two datasets and improves over the semantic branch by 9.09 and 9.66 percentage points. Replacing semantic conditioning with structural conditioning "
        "changes the neighbor-scoring and context-gating path under the same neighborhood and fusion capacity. The results identify relation-aware neighbor context "
        "as the main source of improvement in the two controlled settings.",
    )

    set_plain(
        find_paragraph(document, "MTransE learns mappings"),
        "MTransE learns mappings between language-specific embedding spaces [2], while JAPE adds attribute links to a shared structural space [3]. These methods "
        "connect graphs through explicit geometric mappings and make limited use of relation-specific local neighborhoods. Relation-aware encoders address this gap by "
        "incorporating edge types during neighborhood aggregation.",
    )
    set_plain(
        find_paragraph(document, "RDGCN introduces relation evidence"),
        "RDGCN introduces relation evidence through interaction between an entity graph and a relation dual graph [5]. RREA uses relation reflection to preserve edge-type "
        "differences [6], and RPR-RHGT encodes selected multi-step relation paths with a graph Transformer [10]. The present model uses relation-aware message passing to "
        "produce structural states, then studies how entity semantics conditions the construction of a complete one-hop outgoing-neighbor context. This mechanism differs "
        "from dual-graph interaction, relation reflection, and explicit path encoding.",
    )
    set_plain(
        find_paragraph(document, "Existing methods establish the value"),
        "These studies establish the value of graph structure and text but implement their interaction at different levels. RDGCN and RREA strengthen structural "
        "representations through relation-aware graph encoding [5, 6], whereas MCLEA combines modality-specific representations with contrastive objectives [13]. "
        "The present study isolates a different mechanism. It tests whether one entity representation should condition the construction of another branch's local "
        "neighbor context while the candidate neighborhood and fusion capacity remain fixed.",
    )
    set_plain(
        find_paragraph(document, "The paper presents a relation-aware structural encoder"),
        "The method defines a relation-aware structural branch that preserves edge-type differences and combines propagation depths. This branch supplies self and "
        "neighbor states for the controlled study of structural-context construction.",
    )
    set_plain(
        find_paragraph(document, "The paper introduces semantics-guided structural neighbor selection"),
        "The method introduces semantics-conditioned structural-context construction. The semantic representation controls the neighbor query, the neighbor-compatibility "
        "gate, and the self-versus-neighbor context gate before final joint encoding. A structural-conditioning control changes this common conditioning source while "
        "preserving the neighborhood, parameter count, weighting function, and retrieval representation.",
    )
    set_plain(
        find_paragraph(document, "Experiments with three random seeds on five datasets"),
        "The experiments cover five datasets and three random seeds. Controlled component studies use two representative cross-lingual pairs. A seed-42 RREA-basic "
        "reference run uses the same data splits, candidate sets, validation checkpoint rule, and CSLS selection protocol. Per-seed results and paired differences are reported "
        "to separate average effects from run-to-run variation.",
    )

    set_plain(
        find_paragraph(document, "Text-enhanced entity alignment shows"),
        "Text-enhanced entity alignment shows that semantic evidence can complement graph structure. RREA reports text-enhanced and structure-only settings [6], and "
        "MCLEA aligns modality-specific and joint representations through contrastive objectives [13]. These approaches combine information after or alongside branch "
        "encoding. The present model addresses a narrower mechanism: one semantic state conditions the weighting of structural neighbors and the construction of a "
        "self-versus-neighbor context before the final structure–semantic gate. The structural-conditioning control changes the conditioning source but retains the same "
        "neighbor set and trainable fusion capacity.",
    )

    set_mixed(
        find_paragraph(document, "For entity i, the fusion module receives four inputs"),
        r"For entity $i$, let $s_i=z_i^{\mathrm{sem}}$ be its semantic representation, $t_i=z_i^{\mathrm{str}}$ its self structural representation, and "
        r"$t_{i,j}$ the structural state of outgoing neighbor $j$. The complete model uses conditioning source $u_i=s_i$, whereas the structural-query control "
        r"uses $u_i=t_i$. The same source $u_i$ is used throughout neighbor-context construction. The projections $q_i=W_Qu_i$, $k_{i,j}=W_Kt_{i,j}$, and "
        r"$v_{i,j}=W_Vt_{i,j}$ produce the query, key, and value vectors.",
    )
    set_mixed(
        find_paragraph(document, "Equation (14) measures"),
        "Equation (14) measures compatibility between conditioning source $u_i$ and structural neighbor $j$. A larger dot product gives a higher initial score. The "
        "factor $d^{-1/2}$, with $d=128$, controls the numerical scale of the dot product.",
    )
    replace_display_equation(
        find_paragraph(document, "pij=σ"),
        r"p_{ij}=\sigma\!\left(W_P\phi(t_{i,j},u_i)\right)",
        "16",
    )
    set_mixed(
        find_paragraph(document, "The gate value"),
        "The scalar gate $p_{ij}$ lies between zero and one and measures compatibility between neighbor $j$ and the common conditioning source. The model combines this gate "
        "with the scaled dot-product score before normalization.",
    )
    replace_display_equation(
        find_paragraph(document, "αi=entmax"),
        r"\boldsymbol{\alpha}_i=\operatorname{entmax}_{1.5}\!\left(\frac{\mathbf{a}_i+\log(\mathbf{p}_i+\epsilon)}{T}+\mathbf{m}_i\right)",
        "17",
    )
    set_mixed(
        find_paragraph(document, "In Eq. (17)"),
        r"In Eq. (17), $\mathbf{a}_i$ and $\mathbf{p}_i$ collect the scores and gate values for all padded neighbor positions. Mask bias $m_{ij}$ equals zero for a valid "
        r"neighbor and $-\infty$ for padding, so the mask is applied after temperature scaling. Temperature $T$ controls concentration. The 1.5-entmax transformation "
        r"can assign exact zero weight to a valid low-scoring neighbor [21]. Valid weights sum to one; an entity without valid neighbors receives an all-zero summary.",
    )
    set_mixed(
        find_paragraph(document, "The model then uses the self structural representation"),
        "The model then uses the self structural representation, neighbor summary, and the same conditioning source $u_i$ to compute a context gate for each dimension.",
    )
    replace_display_equation(
        find_paragraph(document, "gic=σ"),
        r"g_i^c=\sigma\!\left(W_C[\phi(t_i,\bar{c}_i);u_i]\right)",
        "18",
    )
    set_plain(
        find_paragraph(document, "The semantic query changes structural-neighbor weights"),
        "In the complete model, semantics conditions the query projection, neighbor-compatibility gate, and self-versus-neighbor context gate before the final joint gate. "
        "The structural-query control replaces this common conditioning source with the self structural state. The final joint gate still combines the resulting structural "
        "context with the original semantic representation.",
    )
    set_mixed(
        find_paragraph(document, "The structural-query control tests the query signal"),
        "The structural-query control retains the complete one-hop outgoing neighborhood, dynamic padding, 1.5-entmax, temperature, trainable layers, final joint gate, "
        "InfoNCE objective, and joint retrieval. It sets $u_i=t_i$ instead of $u_i=s_i$ in Eqs. (14), (16), and (18). The control therefore changes the conditioning "
        "source for neighbor scoring and structural-context gating. The parameter-matched late-fusion control forms a joint representation after separate branch encoding "
        "and omits neighbor context.",
    )

    set_mixed(
        find_paragraph(document, "The model uses AdamW"),
        "The model uses AdamW [24] with β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸, weight decay 1×10⁻⁵, batch size 512, and a cosine learning-rate schedule whose final rate is "
        "20% of the initial value. The initial "
        "learning rate is 5×10⁻⁴ for DBP15K and 3×10⁻⁴ for OpenEA and EventEA. The representation dimension is 128. The structural encoder has three relation-aware "
        "layers. The global view uses a two-layer, four-head Transformer, the phrase view uses convolution widths 3 and 5, and dropout is 0.1. The entmax and InfoNCE "
        "temperatures are 0.25 and 0.07. Training uses seeds 42, 43, and 44 and lasts at most 36 epochs on DBP15K and 50 epochs on OpenEA and EventEA. Validation "
        "MRR is evaluated every five epochs. The checkpoint with the highest raw-cosine validation MRR is retained; early stopping uses patience four and a minimum "
        "improvement of 10⁻³. Nonmatching entities in the same mini-batch are the InfoNCE negatives. Training uses a single stage without external hard-negative mining "
        r"or pseudo-label expansion. After checkpoint selection, validation MRR selects $k_{\mathrm{CSLS}}$ from $\{3,5,7,10,15,20\}$, and the test set is evaluated once. "
        "Entity text concatenates names, adjacent relation names, and attributes; tokens use 300-dimensional GloVe vectors, while uncovered tokens use deterministic "
        "MD5-seeded random vectors. Sequences are truncated or padded to the lengths in Table 1, and masks exclude padding. Relation names are normalized before shared "
        "identifier assignment; DBP15K additionally uses sup_rel_ids. Reverse edges are not added: structural propagation aggregates incoming edges, whereas context "
        "construction reads outgoing neighbors.",
    )


def revise_tables_and_results(document: Document) -> None:
    direct_runs = read_tsv(DIRECT_RUNS)
    aggregate = read_tsv(DIRECT_AGGREGATE)
    softmax_runs = read_tsv(SOFTMAX_RUNS)
    softmax_aggregate = read_tsv(SOFTMAX_AGGREGATE)
    rrea = load_rrea()

    main_rows = {
        dataset: row_index(aggregate, "main", dataset)
        for dataset in ("dbp15k_zh_en", "openea_en_fr")
    }

    sequence_shapes = (
        "38,960 × 32 × 300",
        "39,594 × 32 × 300",
        "39,654 × 32 × 300",
        "30,000 × 32 × 300",
        "58,719 × 32 × 300",
    )
    for row_index_value, shape in enumerate(sequence_shapes, start=1):
        set_cell(
            document.tables[0].cell(row_index_value, 4),
            shape,
            size=8.2,
            align=WD_ALIGN_PARAGRAPH.CENTER,
        )

    set_plain(
        find_paragraph(document, "Table 3 lists results reported"),
        "Table 3 reports a seed-42 RREA-basic reference run under the common evaluation protocol. The baseline uses the official relation-reflection implementation and "
        "its original training objective. The seed-42 run uses the same train/validation/test split, complete target candidate set, validation-based checkpointing, and "
        "CSLS candidate grid as the proposed model. This comparison uses the basic structural variant without text enhancement or iterative semi-supervision.",
    )
    set_plain(
        find_paragraph(document, "Table 3 Contextual comparison"),
        "Table 3 Same-protocol comparison with a reproduced relation-aware baseline",
    )
    baseline_rows: list[list[str]] = []
    for dataset, label in (
        ("dbp15k_zh_en", "DBP15K ZH–EN"),
        ("openea_en_fr", "OpenEA EN–FR-15K-V2"),
    ):
        rrea_runs = len(rrea[dataset]["runs"])
        baseline_rows.append(
            [
                label,
                "RREA-basic (reproduced, seed 42)",
                str(rrea_runs),
                fmt_baseline_metric(rrea[dataset]["Hits@1"], rrea_runs),
                fmt_baseline_metric(rrea[dataset]["Hits@10"], rrea_runs),
                fmt_baseline_metric(rrea[dataset]["MRR"], rrea_runs),
            ]
        )
        main = main_rows[dataset]
        baseline_rows.append(
            [
                label,
                "Proposed model",
                "3",
                f"{float(main['Hits@1_mean']):.4f} ± {float(main['Hits@1_std']):.4f}",
                f"{float(main['Hits@10_mean']):.4f} ± {float(main['Hits@10_std']):.4f}",
                f"{float(main['MRR_mean']):.4f} ± {float(main['MRR_std']):.4f}",
            ]
        )
    replace_table(
        document,
        document.tables[2],
        ["Dataset", "Method", "n", "Hits@1", "Hits@10", "MRR"],
        baseline_rows,
        [1750, 1950, 800, 1500, 1500, 1500],
        font_size=8.7,
    )
    dbp_gap = 100 * (
        float(main_rows["dbp15k_zh_en"]["Hits@1_mean"])
        - rrea["dbp15k_zh_en"]["Hits@1"][0]
    )
    open_gap = 100 * (
        float(main_rows["openea_en_fr"]["Hits@1_mean"]) - rrea["openea_en_fr"]["Hits@1"][0]
    )
    comparison_text = (
        f"The three-seed mean of the proposed model is {abs(dbp_gap):.2f} percentage points {'higher' if dbp_gap >= 0 else 'lower'} than the seed-42 RREA-basic run on DBP15K "
        f"ZH–EN and {abs(open_gap):.2f} points {'higher' if open_gap >= 0 else 'lower'} on OpenEA EN–FR-15K-V2. The reversed ordering shows that relative performance "
        "depends on the dataset. The following controls identify the conditions associated with the proposed model's results."
    )
    insert_paragraph_before(
        document,
        find_paragraph(document, "5.3 Evidence from component controls"),
        comparison_text,
    )

    table4 = document.tables[3]
    existing = [[cell.text.strip() for cell in row.cells] for row in table4.rows]
    softmax_by_dataset = {
        row["dataset"]: row for row in softmax_aggregate
    }
    insert_at = next(i for i, row in enumerate(existing) if row[0] == "Without relation types")
    new_values = [
        "Complete neighborhood + softmax",
        metric_pair(softmax_by_dataset["dbp15k_zh_en"]),
        metric_pair(softmax_by_dataset["openea_en_fr"]),
    ]
    target_row = table4.rows[insert_at]._tr
    template_row = copy.deepcopy(target_row)
    target_row.addprevious(template_row)
    inserted = table4.rows[insert_at]
    for index, value in enumerate(new_values):
        set_cell(
            inserted.cells[index],
            value,
            size=8.7,
            align=WD_ALIGN_PARAGRAPH.LEFT if index == 0 else WD_ALIGN_PARAGRAPH.CENTER,
        )
    set_table_geometry(table4, [3400, 2800, 2800])

    dbp_soft = softmax_by_dataset["dbp15k_zh_en"]
    open_soft = softmax_by_dataset["openea_en_fr"]
    dbp_delta = 100 * (
        float(dbp_soft["test_Hits@1_mean"] if "test_Hits@1_mean" in dbp_soft else dbp_soft["Hits@1_mean"])
        - float(main_rows["dbp15k_zh_en"]["Hits@1_mean"])
    )
    open_delta = 100 * (
        float(open_soft["test_Hits@1_mean"] if "test_Hits@1_mean" in open_soft else open_soft["Hits@1_mean"])
        - float(main_rows["openea_en_fr"]["Hits@1_mean"])
    )
    set_plain(
        find_paragraph(document, "Relation types are the most stable structural factor"),
        f"Relation types and topology initialization produce the largest cross-dataset decreases in Table 4. Removing relation types lowers Hits@1 by 3.62 and 2.78 "
        f"points, while replacing topology initialization with independent learnable entity vectors lowers it by 3.46 and 8.03 points. The complete-neighborhood softmax "
        f"control changes Hits@1 by {dbp_delta:+.2f} and {open_delta:+.2f} points relative to 1.5-entmax while preserving the neighbor set, temperature, query source, "
        "gates, and training protocol. The fixed eight-neighbor softmax row changes both the neighbor budget and normalization and is interpreted only as a combined "
        "control. Phrase-view removal lowers Hits@1 by 0.81 and 3.95 points; the other semantic-view effects are smaller or dataset dependent.",
    )

    # Use traceable per-seed outputs for the structural-conditioning control.
    table6 = document.tables[5]
    structural = row_index(aggregate, "structural_neighbor_query", "dbp15k_zh_en")
    structural_open = row_index(aggregate, "structural_neighbor_query", "openea_en_fr")
    set_cell(table6.cell(2, 0), "Structural-conditioning control", size=8.7)
    set_cell_mixed(
        table6.cell(2, 1),
        "Same neighborhood and gates; $u_i=t_i$ in Eqs. (14), (16), and (18); direct joint retrieval",
        size=8.7,
    )
    set_cell(table6.cell(2, 3), metric_pair(structural), size=8.7)
    set_cell(table6.cell(2, 4), metric_pair(structural_open), size=8.7)
    set_table_geometry(table6, [1500, 2700, 1200, 1800, 1800])

    dbp_conditioning_delta = 100 * (
        float(main_rows["dbp15k_zh_en"]["Hits@1_mean"]) - float(structural["Hits@1_mean"])
    )
    open_conditioning_delta = 100 * (
        float(main_rows["openea_en_fr"]["Hits@1_mean"]) - float(structural_open["Hits@1_mean"])
    )
    set_mixed(
        find_paragraph(document, "Table 6 tests the neighbor-query signal"),
        "Table 6 compares conditioning sources and neighbor access under matched capacity. The complete model and structural-conditioning control share the complete "
        "neighborhood, trainable layers, 1.5-entmax, temperature, objective, and direct joint retrieval. The control replaces $u_i=s_i$ with $u_i=t_i$ in neighbor scoring "
        "and context gating. Parameter-matched late fusion retains the same trainable fusion parameter count but does not read neighbor context.",
    )
    set_mixed(
        find_paragraph(document, "Semantic queries improve Hits@1"),
        f"Semantic conditioning exceeds structural conditioning by {dbp_conditioning_delta:.2f} and {open_conditioning_delta:.2f} Hits@1 points on DBP15K and OpenEA. "
        "Because $u_i$ enters Eqs. (14), (16), and (18), this difference measures the complete conditioning path rather than the dot-product query alone. The full model "
        "exceeds parameter-matched late fusion by 5.28 and 9.22 points. That gap measures the combined effect of neighbor access and the training interaction path.",
    )

    add_per_seed_section(document, direct_runs, softmax_runs)
    revise_scope(document, dbp_conditioning_delta, open_conditioning_delta)
    for table in document.tables:
        set_table_pagination(table)


def run_key(row: dict[str, str]) -> tuple[str, int]:
    return row["dataset"], int(row["seed"])


def add_per_seed_section(
    document: Document,
    direct_runs: list[dict[str, str]],
    softmax_runs: list[dict[str, str]],
) -> None:
    discussion = find_paragraph(document, "6 Discussion")
    insert_paragraph_before(document, discussion, "5.6 Per-seed and paired analysis", "Heading 2")
    insert_paragraph_before(
        document,
        discussion,
        "Appendix A reports the selected CSLS value and test metrics for each seed. Paired differences compare each control with the complete model at the same seed, "
        "so variation caused by changing the random initialization is separated from the component effect. With three pairs, these differences describe the observed "
        "direction and magnitude.",
    )

    main = {
        run_key(row): row for row in direct_runs if row["variant"] == "main"
    }
    controls: dict[str, list[dict[str, str]]] = {
        variant: [row for row in direct_runs if row["variant"] == variant]
        for variant in (
            "no_relation_types",
            "semantic_only",
            "structural_neighbor_query",
            "late_concat_matched",
        )
    }
    controls["all_neighbor_softmax"] = [
        {"variant": "all_neighbor_softmax", **row} for row in softmax_runs
    ]
    labels = {
        "no_relation_types": "Without relation types",
        "semantic_only": "Semantic branch only",
        "structural_neighbor_query": "Structural conditioning",
        "late_concat_matched": "Parameter-matched late fusion",
        "all_neighbor_softmax": "Complete-neighborhood softmax",
    }
    summaries = []
    for variant, rows in controls.items():
        for dataset in ("dbp15k_zh_en", "openea_en_fr"):
            deltas = [
                float(row["Hits@1"]) - float(main[(dataset, int(row["seed"]))]["Hits@1"])
                for row in rows
                if row["dataset"] == dataset
            ]
            mean, std = sample_mean_std(deltas)
            summaries.append((variant, dataset, 100 * mean, 100 * std))
    summary_text = []
    for variant in ("all_neighbor_softmax", "structural_neighbor_query", "late_concat_matched"):
        dbp = next(row for row in summaries if row[0] == variant and row[1] == "dbp15k_zh_en")
        opn = next(row for row in summaries if row[0] == variant and row[1] == "openea_en_fr")
        summary_text.append(
            f"{labels[variant]}: DBP15K {dbp[2]:+.2f} ± {dbp[3]:.2f} points and OpenEA {opn[2]:+.2f} ± {opn[3]:.2f} points"
        )
    insert_paragraph_before(
        document,
        discussion,
        "The paired Hits@1 differences (control minus complete model; mean ± sample standard deviation) are "
        + "; ".join(summary_text)
        + ". The appendix shows whether each average difference is repeated across all three seeds.",
    )

    references = find_paragraph(document, "References")
    insert_paragraph_before(document, references, "Appendix A Per-seed results", "Heading 1")
    insert_paragraph_before(
        document,
        references,
        "Table A1 Full-model results for each random seed",
        "Caption",
    )
    dataset_labels = {
        "dbp15k_zh_en": "DBP15K ZH–EN",
        "dbp15k_ja_en": "DBP15K JA–EN",
        "dbp15k_fr_en": "DBP15K FR–EN",
        "openea_en_fr": "OpenEA EN–FR-V2",
        "eventea_en_en": "EventEA EN–EN",
    }
    dataset_order = {
        "dbp15k_zh_en": 0,
        "dbp15k_ja_en": 1,
        "dbp15k_fr_en": 2,
        "openea_en_fr": 3,
        "eventea_en_en": 4,
    }
    main_rows = []
    for row in sorted(
        [row for row in direct_runs if row["variant"] == "main"],
        key=lambda item: (dataset_order[item["dataset"]], int(item["seed"])),
    ):
        main_rows.append(
            [
                dataset_labels[row["dataset"]],
                row["seed"],
                row["csls_k"],
                f"{float(row['Hits@1']):.4f}",
                f"{float(row['Hits@10']):.4f}",
                f"{float(row['MRR']):.4f}",
            ]
        )
    table_a1 = create_table_before(
        document,
        references,
        ["Dataset", "Seed", "k", "Hits@1", "Hits@10", "MRR"],
        main_rows,
        [1800, 1050, 950, 1750, 1750, 1700],
        document.tables[0].style,
        font_size=8.5,
    )
    for row in table_a1.rows:
        set_cell_no_wrap(row.cells[1])
        set_cell_no_wrap(row.cells[2])

    insert_paragraph_before(
        document,
        references,
        "Table A2 Per-seed results and paired differences for key controls",
        "Caption",
    )
    control_rows = []
    for variant in (
        "all_neighbor_softmax",
        "no_relation_types",
        "semantic_only",
        "structural_neighbor_query",
        "late_concat_matched",
    ):
        for row in sorted(controls[variant], key=lambda item: (item["dataset"], int(item["seed"]))):
            key = run_key(row)
            reference = main[key]
            delta_h1 = 100 * (float(row["Hits@1"]) - float(reference["Hits@1"]))
            delta_mrr = 100 * (float(row["MRR"]) - float(reference["MRR"]))
            control_rows.append(
                [
                    dataset_labels[row["dataset"]],
                    labels[variant],
                    row["seed"],
                    row["csls_k"],
                    f"{float(row['Hits@1']):.4f}/{float(row['MRR']):.4f}",
                    f"{delta_h1:+.2f}/{delta_mrr:+.2f}",
                ]
            )
    table_a2 = create_table_before(
        document,
        references,
        ["Dataset", "Control", "Seed", "k", "Hits@1/MRR", "Paired ΔH@1/ΔMRR (points)"],
        control_rows,
        [1500, 2050, 950, 900, 1600, 2000],
        document.tables[0].style,
        font_size=8.0,
    )
    for row in table_a2.rows:
        set_cell_no_wrap(row.cells[2])
        set_cell_no_wrap(row.cells[3])


def revise_scope(document: Document, dbp_delta: float, open_delta: float) -> None:
    set_plain(
        find_paragraph(document, "The results are stable across the five datasets"),
        "The complete configuration produces the highest Hits@1 on DBP15K FR–EN and the lowest on OpenEA EN–FR-15K-V2. Sample standard deviations remain below "
        "0.006 on all five datasets. Table 2 therefore establishes the reproducibility of the complete configuration across the tested graph settings. Sections 5.3–5.5 "
        "analyze components on DBP15K ZH–EN and OpenEA EN–FR-15K-V2.",
    )
    set_plain(
        find_paragraph(document, "5.5 Controlled test of neighbor-query signals"),
        "5.5 Controlled test of context-conditioning signals",
    )
    set_plain(
        find_paragraph(document, "Table 6 Neighbor access, query signal"),
        "Table 6 Neighbor access, conditioning source, and capacity controls (Hits@1/MRR, mean ± sample standard deviation, n = 3)",
    )
    set_plain(
        find_paragraph(document, "The component controls identify three main sources"),
        "The controlled experiments support two main conclusions on DBP15K ZH–EN and OpenEA EN–FR-15K-V2. Relation types and topology initialization contribute "
        "consistent structural evidence. The complete model also exceeds the semantic branch, which shows that trainable structural context contributes beyond the "
        "text representation. Replacing semantic conditioning with structural conditioning changes the full neighbor-scoring and context-gating path and produces a "
        "smaller difference. The comparison therefore measures this complete conditioning path.",
    )
    set_plain(
        find_paragraph(document, "The main experiments cover four cross-lingual graph pairs"),
        "The main experiments cover four cross-lingual graph pairs and one same-language EventEA EN–EN pair. Component controls and paired analyses cover DBP15K "
        "ZH–EN and OpenEA EN–FR-15K-V2. EventEA demonstrates the performance of the complete configuration in an event-centric setting. Component-level conclusions "
        "apply to the two controlled datasets. The fixed eight-neighbor softmax control changes two factors, whereas the new "
        "complete-neighborhood softmax control isolates the normalization function at the tested temperature. Parameter-matched late fusion still changes both neighbor "
        "access and the training interaction path.",
    )
    set_plain(
        find_paragraph(document, "This paper examines the role"),
        f"This paper examines relation-aware structural context in a joint entity-alignment representation. On DBP15K ZH–EN and OpenEA EN–FR-15K-V2, relation "
        f"types and topology initialization provide consistent structural gains, and the complete model exceeds both the semantic branch and parameter-free mean fusion. "
        f"Replacing semantic conditioning with structural conditioning lowers Hits@1 by {dbp_delta:.2f} and {open_delta:.2f} points, but this control changes the common "
        "conditioning source in neighbor scoring and context gating. The complete-neighborhood softmax control separates the "
        "normalization function from neighbor truncation, while the late-fusion comparison continues to measure the joint effect of neighbor access and the training "
        "interaction path. Results on five datasets show that the complete configuration can be trained in cross-lingual and same-language event-centric settings. "
        "The component conclusions apply to the two datasets used for controlled experiments.",
    )


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    document = Document(SOURCE)
    if len(document.tables) != 6:
        raise RuntimeError(f"Expected 6 source tables, found {len(document.tables)}")
    revise_prose(document)
    revise_tables_and_results(document)
    document.core_properties.title = (
        "Relation-Aware Neighbor Context and Semantics-Guided Selection for Knowledge Graph Entity Alignment"
    )
    document.core_properties.subject = "Revision-priority manuscript"
    document.core_properties.keywords = (
        "knowledge graph entity alignment; relation-aware graph neural networks; "
        "structural-semantic fusion; neighbor context"
    )
    document.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
