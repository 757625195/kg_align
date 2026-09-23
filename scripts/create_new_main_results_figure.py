from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MODULE = ROOT / "tmp" / "update_paper_figure1_journal.py"
OUTPUT = ROOT / "outputs" / "paper_assets" / "figure_2_new_main_five_dataset_results.png"

spec = spec_from_file_location("journal_figure", SOURCE_MODULE)
journal = module_from_spec(spec)
spec.loader.exec_module(journal)

DATA = [
    ("DBP15K\nZH–EN", (0.7198, 0.8470, 0.7661), (0.0007, 0.0007, 0.0008)),
    ("DBP15K\nJA–EN", (0.7844, 0.8864, 0.8213), (0.0062, 0.0037, 0.0052)),
    ("DBP15K\nFR–EN", (0.9183, 0.9655, 0.9358), (0.0004, 0.0011, 0.0002)),
    ("OpenEA\nEN–FR", (0.6324, 0.8196, 0.6978), (0.0065, 0.0054, 0.0043)),
    ("EventEA\nEN–EN", (0.7552, 0.9106, 0.8156), (0.0015, 0.0015, 0.0012)),
]

METRICS = [
    ("Hits@1", "#2f6f8f"),
    ("Hits@10", "#4f8a70"),
    ("MRR", "#b07a2a"),
]


def centered_text(draw, xy, text, font, fill):
    x, y = xy
    box = draw.multiline_textbbox((0, 0), text, font=font, spacing=4, align="center")
    draw.multiline_text(
        (x - (box[2] - box[0]) / 2, y),
        text,
        font=font,
        fill=fill,
        spacing=4,
        align="center",
    )


def create_figure(path: Path) -> None:
    width, height = 2400, 1080
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    ink = "#25313a"
    grid = "#d8dee3"
    axis = "#697780"
    title_font = journal.font(journal.HEADING_FONT, 34)
    axis_font = journal.font(journal.BODY_FONT, 24)
    label_font = journal.font(journal.BODY_FONT, 25)
    value_font = journal.font(journal.HEADING_FONT, 19)
    legend_font = journal.font(journal.BODY_FONT, 24)

    draw.text((110, 55), "五个数据集上的主模型测试结果", font=title_font, fill=ink)
    draw.text(
        (110, 108),
        "三随机种子均值；误差线表示样本标准差",
        font=axis_font,
        fill=axis,
    )

    left, right, top, bottom = 170, 2290, 205, 870
    y_min, y_max = 0.55, 1.00
    plot_height = bottom - top

    def y_pos(value):
        return bottom - (value - y_min) / (y_max - y_min) * plot_height

    for tick in (0.6, 0.7, 0.8, 0.9, 1.0):
        y = y_pos(tick)
        draw.line((left, y, right, y), fill=grid, width=2)
        text = f"{tick:.1f}"
        box = draw.textbbox((0, 0), text, font=axis_font)
        draw.text((left - 28 - (box[2] - box[0]), y - 15), text, font=axis_font, fill=axis)

    draw.line((left, top, left, bottom), fill=axis, width=3)
    draw.line((left, bottom, right, bottom), fill=axis, width=3)

    group_width = (right - left) / len(DATA)
    bar_width = 78
    bar_gap = 18
    cluster_width = len(METRICS) * bar_width + (len(METRICS) - 1) * bar_gap

    for group_index, (dataset, values, stds) in enumerate(DATA):
        center = left + group_width * (group_index + 0.5)
        start = center - cluster_width / 2
        for metric_index, ((metric, color), value, std) in enumerate(zip(METRICS, values, stds)):
            x1 = start + metric_index * (bar_width + bar_gap)
            x2 = x1 + bar_width
            y = y_pos(value)
            draw.rounded_rectangle((x1, y, x2, bottom), radius=5, fill=color)

            error_top = y_pos(min(y_max, value + std))
            error_bottom = y_pos(max(y_min, value - std))
            error_x = (x1 + x2) / 2
            draw.line((error_x, error_top, error_x, error_bottom), fill=ink, width=3)
            draw.line((error_x - 12, error_top, error_x + 12, error_top), fill=ink, width=3)
            draw.line((error_x - 12, error_bottom, error_x + 12, error_bottom), fill=ink, width=3)

            value_text = f"{value:.3f}"
            box = draw.textbbox((0, 0), value_text, font=value_font)
            draw.text(
                (error_x - (box[2] - box[0]) / 2, error_top - 30),
                value_text,
                font=value_font,
                fill=ink,
            )

        centered_text(draw, (center, bottom + 30), dataset, label_font, ink)

    legend_y = 1000
    legend_total = 630
    legend_x = (width - legend_total) / 2
    for metric, color in METRICS:
        draw.rounded_rectangle((legend_x, legend_y, legend_x + 46, legend_y + 25), radius=3, fill=color)
        draw.text((legend_x + 62, legend_y - 6), metric, font=legend_font, fill=ink)
        legend_x += 210

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True, dpi=(300, 300))


if __name__ == "__main__":
    create_figure(OUTPUT)
    print(OUTPUT)
