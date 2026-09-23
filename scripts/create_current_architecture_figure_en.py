from pathlib import Path
import re
from types import SimpleNamespace

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "paper_assets" / "figure_1_current_protocol_springer_en.png"

W, H = 2400, 1500
ARIAL = "/System/Library/Fonts/Supplemental/Arial.ttf"
ARIAL_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
TIMES_ITALIC = "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf"


def font(path: str, size: int):
    return ImageFont.truetype(path, size)


def formula_to_text(expression: str) -> str:
    replacements = {
        r"\mathcal{G}": "G", r"\mathcal{L}": "L", r"\mathbf{h}": "h",
        r"\mathbf{m}": "m", r"\mathbf{W}": "W", r"\mathbf{e}": "e",
        r"\mathbf{f}": "f", r"\mathbf{z}": "z", r"\mathbf{c}": "c",
        r"\mathbf{v}": "v", r"\mathbf{a}": "a", r"\boldsymbol{\alpha}": "α",
        r"\operatorname{entmax}": "entmax", r"\operatorname{Norm}": "Norm",
        r"\operatorname{Gate}": "Gate", r"\lambda": "λ", r"\alpha": "α",
        r"\epsilon": "ε", r"\ell": "ℓ", r"\to": "→", r"\ldots": "…",
        r"\in": "∈", r"\sum": "Σ", r"\langle": "⟨", r"\rangle": "⟩",
        r"\sqrt": "√", r"\bar": "", r"\mathrm": "", r"\left": "",
        r"\right": "", r"\frac": "", r"\{": "{", r"\}": "}",
    }
    text = expression
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"\^\{([^{}]+)\}", r"^(\1)", text)
    text = re.sub(r"_\{([^{}]+)\}", r"_(\1)", text)
    text = text.replace("{", "").replace("}", "").replace("\\", "")
    return text


def text_bbox(draw, text, used_font, spacing=6):
    return draw.multiline_textbbox((0, 0), text, font=used_font, spacing=spacing)


def fit_font(draw, text, used_font, max_width, max_height, spacing=6, min_size=14):
    candidate = used_font
    while candidate.size > min_size:
        box = text_bbox(draw, text, candidate, spacing)
        if box[2] - box[0] <= max_width and box[3] - box[1] <= max_height:
            return candidate
        candidate = candidate.font_variant(size=candidate.size - 1)
    return candidate


def normalized_font(used_font):
    if isinstance(used_font, int):
        return font(TIMES_ITALIC, max(17, round(used_font * 2.15)))
    return used_font


def centered_text(draw, box_xy, text, used_font, fill="#20262d", spacing=6):
    x1, y1, x2, y2 = box_xy
    if isinstance(used_font, int):
        text = formula_to_text(text)
    selected = normalized_font(used_font)
    selected = fit_font(draw, text, selected, x2 - x1, y2 - y1, spacing)
    box = text_bbox(draw, text, selected, spacing)
    width, height = box[2] - box[0], box[3] - box[1]
    draw.multiline_text(((x1 + x2 - width) / 2, (y1 + y2 - height) / 2 - box[1]),
                        text, font=selected, fill=fill, spacing=spacing, align="center")


def diagram_box(draw, xy, *, fill, outline, title, lines, radius=10,
                title_height=56, width=3, title_font=None):
    x1, y1, x2, y2 = xy
    title_font = title_font or journal.F_TITLE
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)
    if title and title_height:
        draw.line((x1, y1 + title_height, x2, y1 + title_height), fill=outline, width=2)
        centered_text(draw, (x1 + 8, y1 + 3, x2 - 8, y1 + title_height - 2), title, title_font)
    if not lines:
        return
    processed = []
    heights = []
    for text, used_font, color in lines:
        display = formula_to_text(text) if isinstance(used_font, int) else text
        selected = normalized_font(used_font)
        selected = fit_font(draw, display, selected, x2 - x1 - 24, 1000, 5)
        box = text_bbox(draw, display, selected, 5)
        processed.append((display, selected, color))
        heights.append(box[3] - box[1])
    total = sum(heights) + 8 * max(0, len(lines) - 1)
    cursor = y1 + title_height + max(8, (y2 - y1 - title_height - total) / 2)
    for (text, selected, color), height in zip(processed, heights):
        centered_text(draw, (x1 + 12, int(cursor), x2 - 12, int(cursor + height + 4)),
                      text, selected, color, 5)
        cursor += height + 8


def plain_box(draw, xy, text, *, fill, outline, used_font=None, radius=8, width=2, color="#20262d"):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)
    x1, y1, x2, y2 = xy
    centered_text(draw, (x1 + 10, y1 + 8, x2 - 10, y2 - 8), text,
                  used_font or journal.F_BODY, color, 5)


def arrow(draw, points, *, color="#52616d", width=4):
    draw.line(points, fill=color, width=width, joint="curve")
    (x0, y0), (x1, y1) = points[-2], points[-1]
    if abs(x1 - x0) >= abs(y1 - y0):
        head = [(x1, y1), (x1 - 17 if x1 >= x0 else x1 + 17, y1 - 10),
                (x1 - 17 if x1 >= x0 else x1 + 17, y1 + 10)]
    else:
        head = [(x1, y1), (x1 - 10, y1 - 17 if y1 >= y0 else y1 + 17),
                (x1 + 10, y1 - 17 if y1 >= y0 else y1 + 17)]
    draw.polygon(head, fill=color)


journal = SimpleNamespace(
    W=W, H=H, INK="#20262d", PANEL="#8a949e", STRUCT_LINE="#356f8d",
    SEM_LINE="#397d64", FUSION_FILL="#fbf5e9", FUSION_LINE="#a77526",
    EVAL_LINE="#66727d", OUTPUT_FILL="#eef2f7", OUTPUT_LINE="#4c6e91",
    F_PANEL=font(ARIAL_BOLD, 39), F_TITLE=font(ARIAL, 29), F_BODY=font(ARIAL, 27),
    F_SMALL=font(ARIAL, 24), F_TINY=font(ARIAL, 21), F_TAG=font(ARIAL_BOLD, 23),
    F_MATH=font(TIMES_ITALIC, 27), F_MATH_SMALL=font(TIMES_ITALIC, 22),
    box=diagram_box, plain_box=plain_box, arrow=arrow,
)


def create_figure(path: Path) -> None:
    image = Image.new("RGB", (journal.W, journal.H), "white")
    draw = ImageDraw.Draw(image)

    journal.box(draw, (35, 35, 2365, 1085), fill="white", outline=journal.PANEL,
                title="", lines=[], radius=12, title_height=0)
    draw.text((65, 57), "(a)", font=journal.F_PANEL, fill=journal.INK)
    draw.text((130, 58), "Relation-Aware Structural Context and Multi-Scale Semantic Fusion",
              font=journal.F_PANEL, fill=journal.INK)

    journal.box(
        draw, (65, 140, 300, 970), fill="#f7f8fa", outline=journal.EVAL_LINE,
        title="Paired graph input",
        lines=[
            ("Left knowledge graph", journal.F_TINY, journal.INK),
            ("G_L", journal.F_MATH, journal.OUTPUT_LINE),
            ("Right knowledge graph", journal.F_TINY, journal.INK),
            ("G_R", journal.F_MATH, journal.OUTPUT_LINE),
            ("Relation triples", journal.F_TINY, journal.INK),
            ("(h, r, t)", journal.F_MATH, journal.INK),
            ("Names · relation terms\nattribute names · values", journal.F_TINY, journal.INK),
        ], title_height=62,
    )

    journal.box(draw, (345, 140, 1300, 500), fill="#f8fbfd", outline=journal.STRUCT_LINE,
                title="Structural branch: topology initialization and relation-aware propagation",
                lines=[], title_height=62)
    journal.box(
        draw, (375, 235, 630, 450), fill="white", outline="#84aabd",
        title="Topology initialization",
        lines=[
            ("Eight local statistics", journal.F_TINY, journal.INK),
            ("Degree · relation diversity\nneighbor degree · direction balance", journal.F_TINY, journal.INK),
            ("h_i^(0) = W_top f_i + 0.1 e_i", journal.F_MATH_SMALL, journal.STRUCT_LINE),
        ], title_height=48, title_font=journal.F_TAG,
    )
    journal.box(
        draw, (665, 220, 995, 465), fill="white", outline="#84aabd",
        title="Three relation-aware layers",
        lines=[
            ("m_(j->i)^(l) = W_s^(l) h_j^(l) + W_r^(l) e_r", journal.F_MATH_SMALL, journal.STRUCT_LINE),
            ("Incoming-edge aggregation · shared projections\nself/neighbor gate", journal.F_TINY, journal.INK),
        ], title_height=48, title_font=journal.F_TAG,
    )
    journal.box(
        draw, (1030, 235, 1270, 450), fill="white", outline="#84aabd",
        title="Node-wise layer selection",
        lines=[
            ("h_i^(0), ..., h_i^(L)", journal.F_MATH_SMALL, journal.STRUCT_LINE),
            ("Entity-specific fusion\nof propagation depths", journal.F_TINY, journal.INK),
            ("z_i^str", journal.F_MATH, journal.STRUCT_LINE),
        ], title_height=48, title_font=journal.F_TAG,
    )
    journal.arrow(draw, [(300, 320), (345, 320)], color=journal.STRUCT_LINE)
    journal.arrow(draw, [(630, 342), (665, 342)], color=journal.STRUCT_LINE)
    journal.arrow(draw, [(995, 342), (1030, 342)], color=journal.STRUCT_LINE)

    journal.box(draw, (345, 540, 1300, 970), fill="#f8fcfa", outline=journal.SEM_LINE,
                title="Semantic branch: multi-scale encoding of a shared input sequence",
                lines=[], title_height=62)
    journal.plain_box(draw, (375, 635, 605, 745), "300-dimensional token sequence\nprojection + positional encoding",
                      fill="white", outline="#85ae9d", used_font=journal.F_TINY)
    journal.plain_box(draw, (650, 620, 820, 755), "token view\nmean + attention",
                      fill="white", outline="#85ae9d", used_font=journal.F_TINY)
    journal.plain_box(draw, (845, 620, 1015, 755), "phrase view\nConv3/5",
                      fill="white", outline="#85ae9d", used_font=journal.F_TINY)
    journal.plain_box(draw, (1040, 620, 1245, 755), "global view\ntwo-layer Transformer",
                      fill="white", outline="#85ae9d", used_font=journal.F_TINY)
    journal.box(
        draw, (650, 810, 1245, 925), fill="white", outline="#85ae9d",
        title="View gate and semantic representation",
        lines=[
            ("Adaptive view weights · MLP · L2 normalization", journal.F_TINY, journal.INK),
            ("z_i^sem", journal.F_MATH, journal.SEM_LINE),
        ], title_height=42, title_font=journal.F_TAG,
    )
    journal.arrow(draw, [(300, 700), (345, 700)], color=journal.SEM_LINE)
    journal.arrow(draw, [(605, 690), (650, 690)], color=journal.SEM_LINE)
    journal.arrow(draw, [(735, 755), (735, 810)], color=journal.SEM_LINE)
    journal.arrow(draw, [(930, 755), (930, 810)], color=journal.SEM_LINE)
    journal.arrow(draw, [(1140, 755), (1140, 810)], color=journal.SEM_LINE)

    journal.box(draw, (1340, 140, 2035, 970), fill=journal.FUSION_FILL,
                outline=journal.FUSION_LINE, title="One-hop outgoing context and structure-semantic fusion",
                lines=[], title_height=62)
    journal.box(
        draw, (1375, 230, 2000, 405), fill="white", outline="#c9a361",
        title="1  Variable-size neighborhood batching",
        lines=[
            ("Read every one-hop outgoing neighbor along original edges", journal.F_TINY, journal.INK),
            ("Dynamic padding to the batch maximum · no new parameters", journal.F_TINY, journal.INK),
        ], title_height=46, title_font=journal.F_TAG,
    )
    journal.box(
        draw, (1375, 445, 2000, 665), fill="white", outline="#c9a361",
        title="2  Sparse neighbor weighting",
        lines=[
            ("a_ij = <W_Q z_i^sem, W_K z_j^str>/(0.25 sqrt(d)) + log(p_ij + eps)", journal.F_MATH_SMALL, journal.FUSION_LINE),
            ("alpha_i = entmax_1.5(a_i)", journal.F_MATH, journal.FUSION_LINE),
            ("Low-relevance neighbors may receive exactly zero weight", journal.F_TINY, journal.INK),
        ], title_height=46, title_font=journal.F_TAG,
    )
    journal.box(
        draw, (1375, 705, 2000, 910), fill="white", outline="#c9a361",
        title="3  Structural context and joint gate",
        lines=[
            ("cbar_i = sum_j alpha_ij v_j", journal.F_MATH, journal.FUSION_LINE),
            ("Gate between self-structure and neighbor evidence", journal.F_TINY, journal.INK),
            ("z_i^joint = Norm(Gate(c_i^str, z_i^sem))", journal.F_MATH_SMALL, journal.FUSION_LINE),
        ], title_height=46, title_font=journal.F_TAG,
    )
    journal.arrow(draw, [(1300, 342), (1340, 342)], color=journal.STRUCT_LINE)
    journal.arrow(draw, [(1300, 870), (1320, 870), (1320, 540), (1375, 540)], color=journal.SEM_LINE)
    journal.arrow(draw, [(1687, 405), (1687, 445)], color=journal.FUSION_LINE)
    journal.arrow(draw, [(1687, 665), (1687, 705)], color=journal.FUSION_LINE)

    journal.box(
        draw, (2075, 300, 2330, 790), fill=journal.OUTPUT_FILL, outline=journal.OUTPUT_LINE,
        title="Shared encoder outputs",
        lines=[
            ("z_i^str", journal.F_MATH, journal.STRUCT_LINE),
            ("z_i^sem", journal.F_MATH, journal.SEM_LINE),
            ("c_i^str", journal.F_MATH, journal.FUSION_LINE),
            ("z_i^joint", journal.F_MATH, journal.OUTPUT_LINE),
            ("Parameters shared\nacross both graphs", journal.F_TINY, journal.INK),
        ],
    )
    journal.arrow(draw, [(2000, 810), (2050, 810), (2050, 545), (2075, 545)], color=journal.OUTPUT_LINE)

    journal.box(draw, (35, 1115, 2365, 1465), fill="white", outline=journal.PANEL,
                title="", lines=[], radius=12, title_height=0)
    draw.text((65, 1135), "(b)", font=journal.F_PANEL, fill=journal.INK)
    draw.text((130, 1136), "Training Objective and Validation-Guided Retrieval",
              font=journal.F_PANEL, fill=journal.INK)
    journal.box(
        draw, (70, 1210, 965, 1430), fill="#f7f8fa", outline=journal.EVAL_LINE,
        title="Joint and structural supervision on seed alignments",
        lines=[
            ("L = L_joint + lambda_str(p) L_str", journal.F_MATH, journal.EVAL_LINE),
            ("lambda_str(p) = 0.1(1 - p)", journal.F_MATH, journal.EVAL_LINE),
            ("Both terms use bidirectional InfoNCE; structural weight decays to zero", journal.F_TINY, journal.INK),
        ], title_height=48, title_font=journal.F_TAG,
    )
    journal.box(
        draw, (1010, 1210, 2295, 1430), fill="#fbfcfd", outline=journal.OUTPUT_LINE,
        title="Direct joint retrieval and validation selection",
        lines=[
            ("z_eval = z_joint", journal.F_MATH, journal.OUTPUT_LINE),
            ("k_CSLS in {3, 5, 7, 10, 15, 20}", journal.F_MATH, journal.OUTPUT_LINE),
            ("Validation MRR selects k_CSLS per run → all-candidate CSLS → one test report", journal.F_TINY, journal.INK),
        ], title_height=48, title_font=journal.F_TAG,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True, dpi=(300, 300))


if __name__ == "__main__":
    create_figure(OUTPUT)
    print(OUTPUT)
