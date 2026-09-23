from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "paper_assets" / "figure_1_springer_submission_zh_600dpi.png"

WIDTH = 3810
HEIGHT = 2700
HEITI_LIGHT = "/System/Library/Fonts/STHeiti Light.ttc"
HEITI_MEDIUM = "/System/Library/Fonts/STHeiti Medium.ttc"
TIMES_ITALIC = "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf"

INK = "#20262D"
LINE = "#687681"
STRUCT = "#2F6F8F"
STRUCT_FILL = "#F3F8FB"
SEM = "#397D64"
SEM_FILL = "#F3F9F6"
FUSION = "#9A6A1C"
FUSION_FILL = "#FCF8EF"
OUTPUT_LINE = "#506C91"
OUTPUT_FILL = "#F1F4F8"


def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


F_PANEL = load_font(HEITI_MEDIUM, 76)
F_TITLE = load_font(HEITI_MEDIUM, 64)
F_BODY = load_font(HEITI_LIGHT, 64)
F_MATH = load_font(TIMES_ITALIC, 66)
F_MATH_SCRIPT = load_font(TIMES_ITALIC, 46)


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> float:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def wrap_zh_line(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    if not text:
        return [""]
    lines: list[str] = []
    current = ""
    for char in text:
        candidate = current + char
        if current and text_width(draw, candidate, font) > max_width:
            lines.append(current.rstrip())
            current = char.lstrip()
        else:
            current = candidate
    if current:
        lines.append(current.rstrip())
    return lines


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> str:
    lines: list[str] = []
    for source_line in text.split("\n"):
        lines.extend(wrap_zh_line(draw, source_line, font, max_width))
    return "\n".join(lines)


def fit_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    used_font: ImageFont.FreeTypeFont,
    max_width: int,
    max_height: int,
    spacing: int,
    min_size: int = 50,
) -> tuple[ImageFont.FreeTypeFont, str]:
    """Fit wrapped text inside a reserved region without crossing its boundary."""
    candidate = used_font
    while candidate.size >= min_size:
        wrapped = wrap_text(draw, text, candidate, max_width)
        box = draw.multiline_textbbox(
            (0, 0), wrapped, font=candidate, spacing=spacing, align="center"
        )
        if box[2] - box[0] <= max_width and box[3] - box[1] <= max_height:
            return candidate, wrapped
        candidate = candidate.font_variant(size=candidate.size - 1)
    raise ValueError(f"Text does not fit its reserved figure region: {text!r}")


def centered(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: str = INK,
    spacing: int = 18,
) -> None:
    x1, y1, x2, y2 = xy
    font, wrapped = fit_wrapped_text(
        draw, text, font, x2 - x1, y2 - y1, spacing
    )
    box = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=spacing, align="center")
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.multiline_text(
        ((x1 + x2 - width) / 2, (y1 + y2 - height) / 2 - box[1]),
        wrapped,
        font=font,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def math_spans(text: str) -> list[tuple[str, str]]:
    spans: list[tuple[str, str]] = []
    base = ""
    index = 0
    while index < len(text):
        char = text[index]
        if char not in "_^":
            base += char
            index += 1
            continue
        if base:
            spans.append((base, "base"))
            base = ""
        mode = "sub" if char == "_" else "sup"
        index += 1
        if index >= len(text):
            break
        if text[index] in "({":
            opener = text[index]
            closer = ")" if opener == "(" else "}"
            end = text.find(closer, index + 1)
            if end == -1:
                end = index
            token = text[index + 1 : end]
            index = end + 1
        else:
            start = index
            while index < len(text) and (text[index].isalnum() or text[index] == ","):
                index += 1
            token = text[start:index]
        spans.append((token, mode))
    if base:
        spans.append((base, "base"))
    return spans


def centered_math(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    text: str,
    fill: str,
) -> None:
    x1, y1, x2, y2 = xy
    lines = text.split("\n")
    line_height = 92
    total_height = line_height * len(lines)
    top = (y1 + y2 - total_height) / 2
    for line_number, line in enumerate(lines):
        spans = math_spans(line)
        widths = [
            text_width(draw, value, F_MATH if mode == "base" else F_MATH_SCRIPT)
            for value, mode in spans
        ]
        cursor = (x1 + x2 - sum(widths)) / 2
        baseline_y = top + line_number * line_height + 16
        for (value, mode), width in zip(spans, widths):
            font = F_MATH if mode == "base" else F_MATH_SCRIPT
            y = baseline_y
            if mode == "sub":
                y += 34
            elif mode == "sup":
                y -= 18
            draw.text((cursor, y), value, font=font, fill=fill)
            cursor += width


def box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    body: str,
    *,
    fill: str,
    outline: str,
    math: str | None = None,
    title_height: int = 120,
    math_height: int = 150,
) -> None:
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=14, fill=fill, outline=outline, width=5)
    draw.line((x1, y1 + title_height, x2, y1 + title_height), fill=outline, width=4)
    centered(draw, (x1 + 24, y1 + 8, x2 - 24, y1 + title_height - 8), title, F_TITLE)
    body_top = y1 + title_height + 18
    if math:
        centered(draw, (x1 + 26, body_top, x2 - 26, y2 - math_height - 12), body, F_BODY)
        centered_math(draw, (x1 + 26, y2 - math_height, x2 - 26, y2 - 18), math, outline)
    else:
        centered(draw, (x1 + 26, body_top, x2 - 26, y2 - 18), body, F_BODY)


def arrow(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], color: str = LINE) -> None:
    draw.line(points, fill=color, width=9, joint="curve")
    (x0, y0), (x1, y1) = points[-2], points[-1]
    if abs(x1 - x0) >= abs(y1 - y0):
        direction = 1 if x1 >= x0 else -1
        head = [(x1, y1), (x1 - 30 * direction, y1 - 18), (x1 - 30 * direction, y1 + 18)]
    else:
        direction = 1 if y1 >= y0 else -1
        head = [(x1, y1), (x1 - 18, y1 - 30 * direction), (x1 + 18, y1 - 30 * direction)]
    draw.polygon(head, fill=color)


def create_figure(path: Path = OUTPUT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle((55, 45, 3755, 1970), radius=18, fill="white", outline=LINE, width=5)
    centered(draw, (95, 70, 245, 175), "(a)", F_PANEL)

    box(
        draw,
        (120, 240, 590, 1740),
        "输入",
        "两个有向知识图谱\n\n关系三元组 (h, r, t)\n\n名称、关系词与属性\n\n种子实体对",
        fill="#F6F7F8",
        outline=LINE,
    )
    box(
        draw,
        (690, 240, 1810, 900),
        "关系感知结构分支",
        "拓扑统计与共享投影\n关系感知传入消息\n节点级层选择",
        fill=STRUCT_FILL,
        outline=STRUCT,
        math="h_i^(0), ..., h_i^(L)  →  z_i^str",
    )
    box(
        draw,
        (690, 1020, 1810, 1740),
        "多尺度语义分支",
        "同一 token 序列\ntoken、phrase 与 global 视图\n实体级视图门控",
        fill=SEM_FILL,
        outline=SEM,
        math="z_i^tok, z_i^phr, z_i^glo  →  z_i^sem",
    )
    box(
        draw,
        (1930, 240, 3120, 1740),
        "语义引导的结构上下文",
        "完整一跳出邻域\n语义查询与有效掩码\n1.5-entmax 稀疏权重\n自身/邻居上下文门控\n最终结构—语义门控",
        fill=FUSION_FILL,
        outline=FUSION,
        math="α_i = entmax_{1.5}(a_i/T)  →  c_i^str\n→  z_i^joint",
        math_height=190,
    )
    box(
        draw,
        (3240, 520, 3685, 1480),
        "输出",
        "结构表示\n\n语义表示\n\n邻居上下文",
        fill=OUTPUT_FILL,
        outline=OUTPUT_LINE,
        math="z_i^joint",
    )

    arrow(draw, [(590, 600), (690, 600)], STRUCT)
    arrow(draw, [(590, 1380), (690, 1380)], SEM)
    arrow(draw, [(1810, 570), (1870, 570), (1870, 670), (1930, 670)], STRUCT)
    arrow(draw, [(1810, 1380), (1870, 1380), (1870, 1250), (1930, 1250)], SEM)
    arrow(draw, [(3120, 990), (3240, 990)], OUTPUT_LINE)

    draw.rounded_rectangle((55, 2040, 3755, 2650), radius=18, fill="white", outline=LINE, width=5)
    centered(draw, (95, 2070, 245, 2175), "(b)", F_PANEL)

    box(
        draw,
        (250, 2120, 1210, 2595),
        "训练",
        "双向 InfoNCE\n早期结构损失",
        fill="#F6F7F8",
        outline=LINE,
        math="L = L_joint + λ(t)L_str",
        title_height=105,
        math_height=135,
    )
    box(
        draw,
        (1425, 2120, 2385, 2595),
        "主检索",
        "训练与检索使用\n同一联合表示",
        fill=OUTPUT_FILL,
        outline=OUTPUT_LINE,
        math="cos(z_L^joint, z_R^joint)",
        title_height=105,
        math_height=135,
    )
    box(
        draw,
        (2600, 2120, 3560, 2595),
        "验证控制",
        "验证集仅选择\nCSLS 邻域大小",
        fill="#F6F7F8",
        outline=LINE,
        math="k_CSLS in {3, 5, 10, 15, 20}",
        title_height=105,
        math_height=135,
    )
    arrow(draw, [(1210, 2360), (1425, 2360)], LINE)
    arrow(draw, [(2385, 2360), (2600, 2360)], LINE)

    image.save(path, dpi=(600, 600), optimize=True)
    return path


if __name__ == "__main__":
    print(create_figure())
