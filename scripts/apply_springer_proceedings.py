from copy import deepcopy
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


SRC = Path("outputs/1.docx")
OUT = Path("outputs/1_springer_proceedings.docx")


def set_run_font(run, size=None, bold=None, italic=None):
    run.font.name = "Times New Roman"
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Times New Roman")
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def clear_direct_paragraph_format(p):
    pPr = p._p.get_or_add_pPr()
    keep = {qn("w:pStyle"), qn("w:numPr"), qn("w:sectPr")}
    for child in list(pPr):
        if child.tag not in keep:
            pPr.remove(child)


def style_font(style, size, bold=False, italic=False):
    style.font.name = "Times New Roman"
    style._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Times New Roman")
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.italic = italic


def ensure_style(doc, name, base="Normal"):
    try:
        st = doc.styles[name]
    except KeyError:
        st = doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    if base:
        st.base_style = doc.styles[base]
    return st


def configure_styles(doc):
    normal = doc.styles["Normal"]
    style_font(normal, 10)
    pf = normal.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.first_line_indent = Cm(0.4)
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing_rule = WD_LINE_SPACING.AT_LEAST
    pf.line_spacing = Pt(12)

    title = ensure_style(doc, "Springer Paper Title")
    style_font(title, 14, bold=True)
    pf = title.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.first_line_indent = Cm(0)
    pf.space_before = Pt(0)
    pf.space_after = Pt(24)
    pf.line_spacing_rule = WD_LINE_SPACING.AT_LEAST
    pf.line_spacing = Pt(18)

    author = ensure_style(doc, "Springer Author")
    style_font(author, 10)
    pf = author.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.first_line_indent = Cm(0)
    pf.space_after = Pt(10)
    pf.line_spacing_rule = WD_LINE_SPACING.AT_LEAST
    pf.line_spacing = Pt(11)

    abstract = ensure_style(doc, "Springer Abstract")
    style_font(abstract, 9)
    pf = abstract.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.left_indent = Cm(1)
    pf.right_indent = Cm(1)
    pf.first_line_indent = Cm(0)
    pf.space_before = Pt(30)
    pf.space_after = Pt(18)
    pf.line_spacing_rule = WD_LINE_SPACING.AT_LEAST
    pf.line_spacing = Pt(11)

    keywords = ensure_style(doc, "Springer Keywords", "Springer Abstract")
    pf = keywords.paragraph_format
    pf.space_before = Pt(11)
    pf.space_after = Pt(0)

    h1 = doc.styles["Heading 1"]
    style_font(h1, 12, bold=True)
    pf = h1.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.first_line_indent = Cm(-1)
    pf.left_indent = Cm(1)
    pf.tab_stops.clear_all()
    pf.tab_stops.add_tab_stop(Cm(1), WD_TAB_ALIGNMENT.LEFT)
    pf.space_before = Pt(18)
    pf.space_after = Pt(12)
    pf.line_spacing_rule = WD_LINE_SPACING.AT_LEAST
    pf.line_spacing = Pt(15)
    pf.keep_with_next = True

    h2 = doc.styles["Heading 2"]
    style_font(h2, 10, bold=True)
    pf = h2.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.first_line_indent = Cm(-1)
    pf.left_indent = Cm(1)
    pf.tab_stops.clear_all()
    pf.tab_stops.add_tab_stop(Cm(1), WD_TAB_ALIGNMENT.LEFT)
    pf.space_before = Pt(18)
    pf.space_after = Pt(8)
    pf.line_spacing_rule = WD_LINE_SPACING.AT_LEAST
    pf.line_spacing = Pt(12)
    pf.keep_with_next = True

    h3 = doc.styles["Heading 3"]
    style_font(h3, 10)
    pf = h3.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.first_line_indent = Cm(0)
    pf.left_indent = Cm(0)
    pf.space_before = Pt(18)
    pf.space_after = Pt(0)
    pf.line_spacing_rule = WD_LINE_SPACING.AT_LEAST
    pf.line_spacing = Pt(12)

    cap = doc.styles["Caption"]
    style_font(cap, 9)
    pf = cap.paragraph_format
    pf.first_line_indent = Cm(0)
    pf.line_spacing_rule = WD_LINE_SPACING.AT_LEAST
    pf.line_spacing = Pt(11)
    pf.keep_with_next = True

    ref = ensure_style(doc, "Springer Reference")
    style_font(ref, 9)
    pf = ref.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.left_indent = Cm(0.75)
    pf.first_line_indent = Cm(-0.75)
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing_rule = WD_LINE_SPACING.AT_LEAST
    pf.line_spacing = Pt(11)


def replace_text_preserving_first_run(p, text, bold_label=None):
    # Clear all paragraph content, including hidden/stray OMML objects, while
    # preserving paragraph properties.
    for child in list(p._p):
        if child.tag != qn("w:pPr"):
            p._p.remove(child)
    if bold_label and text.startswith(bold_label):
        r = p.add_run(bold_label)
        set_run_font(r, bold=True)
        r = p.add_run(text[len(bold_label):])
        set_run_font(r)
    else:
        r = p.add_run(text)
        set_run_font(r)


def merge_run_in_heading_preserving_body(p, nxt, heading):
    """Merge a Heading 3 paragraph with its body without flattening OMML."""
    for child in list(p._p):
        if child.tag != qn("w:pPr"):
            p._p.remove(child)
    run = p.add_run(heading + " ")
    set_run_font(run, bold=True)
    for child in list(nxt._p):
        if child.tag != qn("w:pPr"):
            p._p.append(deepcopy(child))


def set_math_run_font(math_run, size_half_points=15):
    """Apply table typography to native Word-math runs."""
    r_pr = math_run.find(qn("w:rPr"))
    if r_pr is None:
        r_pr = OxmlElement("w:rPr")
        math_run.insert(0, r_pr)
    fonts = r_pr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        r_pr.append(fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia"):
        fonts.set(qn(attr), "Cambria Math")
    for tag in ("w:sz", "w:szCs"):
        node = r_pr.find(qn(tag))
        if node is None:
            node = OxmlElement(tag)
            r_pr.append(node)
        node.set(qn("w:val"), str(size_half_points))


def add_page_field(p):
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "2"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    r = p.add_run()
    r._r.extend([begin, instr, separate, text, end])
    set_run_font(r, 9)


def clear_part_paragraphs(part):
    for p in part.paragraphs:
        for child in list(p._p):
            if child.tag != qn("w:pPr"):
                p._p.remove(child)
    return part.paragraphs[0]


def ensure_child(parent, tag):
    child = parent.find(qn(tag))
    if child is None:
        child = OxmlElement(tag)
        parent.append(child)
    return child


def fit_table_to_springer_text_width(table, target_width=6889):
    """Scale an existing table into the template's 12.2 cm text frame."""
    tbl = table._tbl
    old_grid = [int(col.get(qn("w:w"))) for col in tbl.tblGrid.gridCol_lst]
    total = sum(old_grid)
    new_grid = [max(1, round(w * target_width / total)) for w in old_grid]
    new_grid[-1] += target_width - sum(new_grid)

    tbl_pr = tbl.tblPr
    tbl_w = ensure_child(tbl_pr, "w:tblW")
    tbl_w.set(qn("w:type"), "dxa")
    tbl_w.set(qn("w:w"), str(target_width))
    jc = ensure_child(tbl_pr, "w:jc")
    jc.set(qn("w:val"), "center")
    layout = ensure_child(tbl_pr, "w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is not None:
        tbl_pr.remove(tbl_ind)
    cell_mar = ensure_child(tbl_pr, "w:tblCellMar")
    for side in ("left", "right"):
        el = ensure_child(cell_mar, f"w:{side}")
        el.set(qn("w:w"), "70")
        el.set(qn("w:type"), "dxa")

    for col, width in zip(tbl.tblGrid.gridCol_lst, new_grid):
        col.set(qn("w:w"), str(width))

    factor = target_width / total
    for tr in tbl.tr_lst:
        cells = tr.tc_lst
        scaled = []
        for tc in cells:
            tc_w = tc.tcPr.tcW
            old = int(tc_w.get(qn("w:w"))) if tc_w is not None else round(total / len(cells))
            scaled.append(max(1, round(old * factor)))
        if scaled:
            scaled[-1] += target_width - sum(scaled)
        for tc, width in zip(cells, scaled):
            tc_w = ensure_child(tc.get_or_add_tcPr(), "w:tcW")
            tc_w.set(qn("w:type"), "dxa")
            tc_w.set(qn("w:w"), str(width))

    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER


doc = Document(SRC)
configure_styles(doc)

section = doc.sections[0]
section.page_width = Cm(21.0)
section.page_height = Cm(29.7)
section.top_margin = Cm(5.2)
section.bottom_margin = Cm(5.2)
section.left_margin = Cm(4.4)
section.right_margin = Cm(4.4)
section.header_distance = Cm(4.2)
section.footer_distance = Cm(4.1)
section.different_first_page_header_footer = True
settings = doc.settings._element
if settings.find(qn("w:evenAndOddHeaders")) is None:
    settings.insert(0, OxmlElement("w:evenAndOddHeaders"))

paras = doc.paragraphs
paras[0].style = "Springer Paper Title"
paras[1].style = "Springer Author"
replace_text_preserving_first_run(paras[0], paras[0].text)
replace_text_preserving_first_run(paras[1], paras[1].text)

# Merge the standalone Abstract heading with its following paragraph.
if paras[2].text.strip().lower() == "abstract":
    abstract_text = paras[3].text.strip()
    replace_text_preserving_first_run(paras[2], "Abstract. " + abstract_text, "Abstract.")
    paras[2].style = "Springer Abstract"
    paras[3]._element.getparent().remove(paras[3]._element)

# Refresh paragraph list after deletion.
paras = doc.paragraphs
for p in paras:
    txt = p.text.strip()
    if txt.startswith("Keywords:"):
        p.style = "Springer Keywords"
        replace_text_preserving_first_run(p, txt, "Keywords:")

# Format headings and turn third-level headings into bold run-ins.
paras = doc.paragraphs
i = 0
while i < len(paras):
    p = paras[i]
    style_name = p.style.name
    txt = p.text.strip()
    if style_name == "Heading 1":
        m = re.match(r"^(\d+)\s+(.+)$", txt)
        if m:
            replace_text_preserving_first_run(p, f"{m.group(1)}\t{m.group(2)}")
    elif style_name == "Heading 2":
        m = re.match(r"^(\d+\.\d+)\s+(.+)$", txt)
        if m:
            replace_text_preserving_first_run(p, f"{m.group(1)}\t{m.group(2)}")
    elif style_name == "Heading 3" and i + 1 < len(paras):
        heading = txt.rstrip(".") + "."
        nxt = paras[i + 1]
        if nxt.style.name == "Normal":
            merge_run_in_heading_preserving_body(p, nxt, heading)
            p.style = "Heading 3"
            nxt._element.getparent().remove(nxt._element)
            paras = doc.paragraphs
            i += 1
            continue
    i += 1

# Paragraph roles, captions and references.
paras = doc.paragraphs
previous_role = None
for p in paras:
    txt = p.text.strip()
    if not txt:
        continue
    if p.style.name.lower() == "caption" and (txt.startswith("Fig. ") or txt.startswith("Table ")):
        p.style = "Caption"
        p.paragraph_format.alignment = (WD_ALIGN_PARAGRAPH.JUSTIFY if len(txt) > 85 else WD_ALIGN_PARAGRAPH.CENTER)
        p.paragraph_format.space_before = Pt(6 if txt.startswith("Fig.") else 12)
        p.paragraph_format.space_after = Pt(12 if txt.startswith("Fig.") else 6)
        # Bold the label through the first numeric token.
        m = re.match(r"^((?:Fig\.|Table)\s+\d+)\s*", txt)
        if m:
            label = m.group(1)
            # The source already has the caption label in a dedicated bold
            # run. Preserve the remaining runs and any inline OMML objects.
            first_run = next((r for r in p.runs if r.text.strip()), None)
            if first_run is not None and first_run.text.strip().startswith(label):
                first_run.bold = True
        previous_role = "caption"
    elif re.match(r"^\[\d+\]", txt):
        p.style = "Springer Reference"
        previous_role = "reference"
    elif p.style.name == "Normal":
        clear_direct_paragraph_format(p)
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if previous_role in {"heading", "caption", "equation", "table"}:
            p.paragraph_format.first_line_indent = Cm(0)
        previous_role = "normal"
    elif p.style.name.startswith("Heading"):
        previous_role = "heading"
    else:
        previous_role = p.style.name.lower()
    for r in p.runs:
        if r.font.name not in ("Cambria Math", "Courier New"):
            r._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), "Times New Roman")
            r._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), "Times New Roman")
            r._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Times New Roman")

# Table text and geometry: the source tables were 16 cm wide, while the
# Springer proceedings frame is only 12.2 cm wide.
for table_idx, table in enumerate(doc.tables, start=1):
    fit_table_to_springer_text_width(table)
    if table_idx == 1:
        # The native equations in the sequence-shape column cannot wrap.
        # Reserve enough width for the complete N x L x 300 expressions.
        custom = [1100, 780, 800, 1450, 1950, 1009]
        tbl = table._tbl
        for col, width in zip(tbl.tblGrid.gridCol_lst, custom):
            col.set(qn("w:w"), str(width))
        for tr in tbl.tr_lst:
            for tc, width in zip(tr.tc_lst, custom):
                tc_w = ensure_child(tc.get_or_add_tcPr(), "w:tcW")
                tc_w.set(qn("w:type"), "dxa")
                tc_w.set(qn("w:w"), str(width))
    elif table_idx == 6:
        # Give the compact parameter-count header enough room to avoid an
        # isolated final character while retaining the two metric columns.
        custom = [1450, 1800, 1150, 1245, 1244]
        tbl = table._tbl
        for col, width in zip(tbl.tblGrid.gridCol_lst, custom):
            col.set(qn("w:w"), str(width))
        for tr in tbl.tr_lst:
            for tc, width in zip(tr.tc_lst, custom):
                tc_w = ensure_child(tc.get_or_add_tcPr(), "w:tcW")
                tc_w.set(qn("w:type"), "dxa")
                tc_w.set(qn("w:w"), str(width))
    for row_idx, row in enumerate(table.rows):
        for cell in row.cells:
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in cell.paragraphs:
                p.paragraph_format.first_line_indent = Cm(0)
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.AT_LEAST
                p.paragraph_format.line_spacing = Pt(8)
                for r in p.runs:
                    set_run_font(r, size=7.5, bold=True if row_idx == 0 else None)
                for math_run in p._p.xpath(".//m:r"):
                    set_math_run_font(math_run, size_half_points=12)

# Keep figures inside the same text frame and preserve their aspect ratio.
for shape in doc.inline_shapes:
    max_width = Cm(12.2)
    if shape.width > max_width:
        ratio = max_width / shape.width
        shape.width = max_width
        shape.height = int(shape.height * ratio)

# Keep the image with its caption, but do not force the following subsection
# to remain on the same page as the figure caption.
for idx, p in enumerate(doc.paragraphs[:-1]):
    if p._p.xpath(".//w:drawing|.//w:pict"):
        p.paragraph_format.keep_with_next = True
        nxt = doc.paragraphs[idx + 1]
        if nxt.style.name == "Caption" and nxt.text.strip().startswith("Fig."):
            nxt.paragraph_format.keep_with_next = False

# LibreOffice duplicates the embedded right-side number of the long entmax
# equation.  Keep the equation as native OMML, but place its number in a
# separate right-aligned run so Word and LibreOffice produce the same result.
for p in doc.paragraphs:
    math_text = "".join(node.text or "" for node in p._p.xpath(".//m:t"))
    if "entmax1.5" in math_text:
        math_paragraphs = p._p.xpath("./m:oMathPara")
        if not math_paragraphs:
            continue
        math_paragraph = math_paragraphs[0]
        equation = math_paragraph.find(qn("m:oMath"))
        if equation is None:
            continue
        number_runs = [
            run
            for run in equation.iter(qn("m:r"))
            if "".join(node.text or "" for node in run.iter(qn("m:t"))) == "(17)"
        ]
        for run in number_runs:
            run.getparent().remove(run)

        math_paragraph.remove(equation)
        p._p.remove(math_paragraph)
        p_pr = p._p.get_or_add_pPr()
        tabs = p_pr.find(qn("w:tabs"))
        if tabs is None:
            tabs = OxmlElement("w:tabs")
            p_pr.append(tabs)
        for child in list(tabs):
            tabs.remove(child)
        for alignment, position in (("center", "3545"), ("right", "7089")):
            tab = OxmlElement("w:tab")
            tab.set(qn("w:val"), alignment)
            tab.set(qn("w:pos"), position)
            tabs.append(tab)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        lead = OxmlElement("w:r")
        lead.append(OxmlElement("w:tab"))
        p._p.append(lead)
        p._p.append(equation)
        trail = OxmlElement("w:r")
        trail.append(OxmlElement("w:tab"))
        text = OxmlElement("w:t")
        text.text = "(17)"
        trail.append(text)
        p._p.append(trail)

# A paragraph immediately following a table is not indented in the template.
body = doc._element.body
children = list(body)
for idx, child in enumerate(children[:-1]):
    if child.tag == qn("w:tbl"):
        for nxt in children[idx + 1:]:
            if nxt.tag == qn("w:p"):
                pPr = nxt.get_or_add_pPr()
                ind = pPr.find(qn("w:ind"))
                if ind is None:
                    ind = OxmlElement("w:ind")
                    pPr.append(ind)
                ind.set(qn("w:firstLine"), "0")
                break

# Springer running heads: blank first page, even pages show page + author, odd pages title + page.
first = clear_part_paragraphs(section.first_page_header)
first.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
clear_part_paragraphs(section.first_page_footer)

even = clear_part_paragraphs(section.even_page_header)
even.paragraph_format.tab_stops.add_tab_stop(Cm(7.0), WD_TAB_ALIGNMENT.LEFT)
add_page_field(even)
r = even.add_run("\tX. Lin")
set_run_font(r, 9)

odd = clear_part_paragraphs(section.header)
odd.paragraph_format.tab_stops.add_tab_stop(Cm(12.2), WD_TAB_ALIGNMENT.RIGHT)
r = odd.add_run("Relation-Aware Neighbor Context for Entity Alignment")
set_run_font(r, 9)
r = odd.add_run("\t")
set_run_font(r, 9)
add_page_field(odd)
clear_part_paragraphs(section.footer)
clear_part_paragraphs(section.even_page_footer)

# Ask Word to refresh PAGE fields on opening.
update = settings.find(qn("w:updateFields"))
if update is None:
    update = OxmlElement("w:updateFields")
    settings.append(update)
update.set(qn("w:val"), "true")

doc.save(OUT)
print(OUT.resolve())
