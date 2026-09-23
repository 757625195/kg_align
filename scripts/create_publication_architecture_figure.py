from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "outputs" / "paper_assets"
PNG_OUTPUT = ASSET_DIR / "figure_1_publication_architecture_padded_600dpi.png"
PDF_OUTPUT = ASSET_DIR / "figure_1_publication_architecture_padded_vector.pdf"
SVG_OUTPUT = ASSET_DIR / "figure_1_publication_architecture_padded_vector.svg"

ARIAL = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
ARIAL_BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
if ARIAL.exists():
    font_manager.fontManager.addfont(ARIAL)
if ARIAL_BOLD.exists():
    font_manager.fontManager.addfont(ARIAL_BOLD)

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 6.8,
        "mathtext.fontset": "stixsans",
        "axes.linewidth": 0.6,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

INK = "#24313A"
MUTED = "#56656F"
LINE = "#8D9AA3"
LIGHT_LINE = "#C9D1D6"
PANEL = "#F8FAFB"
STRUCT = "#2F708E"
STRUCT_FILL = "#EAF4F8"
STRUCT_LIGHT = "#F5FAFC"
SEM = "#3D7F67"
SEM_FILL = "#ECF6F1"
SEM_LIGHT = "#F6FBF8"
FUSION = "#9B681C"
FUSION_FILL = "#FFF6E5"
FUSION_LIGHT = "#FFFBF2"
OUTPUT = "#526F99"
OUTPUT_FILL = "#EEF2F8"
ZERO = "#C7CED3"


def rounded_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    face: str,
    edge: str,
    lw: float = 0.9,
    radius: float = 0.7,
    zorder: int = 1,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.12,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def label(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 6.2,
    color: str = INK,
    weight: str = "normal",
    ha: str = "center",
    va: str = "center",
    style: str = "normal",
    zorder: int = 5,
):
    return ax.text(
        x,
        y,
        text,
        fontsize=size,
        color=color,
        fontweight=weight,
        fontstyle=style,
        ha=ha,
        va=va,
        linespacing=1.12,
        zorder=zorder,
    )


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = LINE,
    lw: float = 1.0,
    connectionstyle: str = "arc3",
    zorder: int = 4,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=7.5,
        linewidth=lw,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def routed_arrow(ax, points: list[tuple[float, float]], *, color: str, lw: float = 1.0):
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=lw,
            solid_capstyle="round",
            zorder=3,
        )
    arrow(ax, points[-2], points[-1], color=color, lw=lw)


def section_group(ax, x, y, w, h, title, *, edge, face):
    rounded_box(ax, x, y, w, h, face=face, edge=edge, lw=1.1, radius=0.65)
    ax.plot([x, x + w], [y + h - 3.3, y + h - 3.3], color=edge, linewidth=0.8, zorder=2)
    label(ax, x + 1.2, y + h - 1.65, title, size=8.0, color=edge, weight="bold", ha="left")


def small_module(ax, x, y, w, h, title, body, *, edge, face, title_size=6.6, body_size=5.9):
    rounded_box(ax, x, y, w, h, face=face, edge=edge, lw=0.75, radius=0.45)
    label(ax, x + w / 2, y + h - 1.85, title, size=title_size, color=edge, weight="bold")
    label(ax, x + w / 2, y + h * 0.39, body, size=body_size, color=INK)


def draw_graph(ax, cx, cy, scale, *, color):
    nodes = [(-1.45, 0.1), (0.0, 1.15), (1.45, 0.2), (0.35, -1.15)]
    edges = [(0, 1), (1, 2), (0, 3), (3, 2)]
    for source, target in edges:
        x0, y0 = nodes[source]
        x1, y1 = nodes[target]
        arrow(
            ax,
            (cx + x0 * scale, cy + y0 * scale),
            (cx + x1 * scale, cy + y1 * scale),
            color=color,
            lw=0.65,
        )
    for index, (dx, dy) in enumerate(nodes):
        circle = Circle(
            (cx + dx * scale, cy + dy * scale),
            radius=0.42 * scale,
            edgecolor=color,
            facecolor="white" if index else color,
            linewidth=0.75,
            zorder=5,
        )
        ax.add_patch(circle)


def draw_token_strip(ax, x, y, widths, labels, colors):
    cursor = x
    for width, text, color in zip(widths, labels, colors):
        rounded_box(ax, cursor, y, width, 2.8, face="white", edge=color, lw=0.6, radius=0.38)
        label(ax, cursor + width / 2, y + 1.4, text, size=5.0, color=color, weight="bold")
        cursor += width + 0.35


def draw_layer_stack(ax, x, y):
    colors = ["#D7EAF2", "#C8E1EC", "#B6D7E5", "#A6CEDF"]
    for index, color in enumerate(colors):
        yy = y + index * 1.8
        rounded_box(ax, x + index * 0.22, yy, 5.4, 1.35, face=color, edge=STRUCT, lw=0.55, radius=0.25)
        label(ax, x + 2.7 + index * 0.22, yy + 0.68, rf"$h_i^{{({index})}}$", size=6.0, color=STRUCT)


def draw_neighbor_set(ax, x, y):
    center = (x + 5.0, y + 5.9)
    outer = [
        (x + 2.0, y + 8.2),
        (x + 7.7, y + 8.4),
        (x + 8.4, y + 5.4),
        (x + 5.5, y + 2.7),
        (x + 1.6, y + 4.1),
    ]
    weights = [1.0, 0.70, 0.30, 0.88, 0.20]
    for index, ((nx, ny), weight) in enumerate(zip(outer, weights), start=1):
        arrow(ax, center, (nx, ny), color=STRUCT, lw=0.65)
        node = Circle(
            (nx, ny),
            0.55,
            facecolor=STRUCT_FILL if weight > 0.25 else "white",
            edgecolor=STRUCT if weight > 0.25 else ZERO,
            linewidth=0.8,
            zorder=5,
        )
        ax.add_patch(node)
        label(ax, nx, ny, rf"$j_{index}$", size=4.8, color=STRUCT if weight > 0.25 else ZERO)
    node = Circle(center, 0.68, facecolor=STRUCT, edgecolor=STRUCT, linewidth=0.8, zorder=6)
    ax.add_patch(node)
    label(ax, center[0], center[1], "$i$", size=5.8, color="white", weight="bold", zorder=7)


def draw_sparse_weights(ax, x, y):
    values = [0.46, 0.31, 0.0, 0.23, 0.0]
    for index, value in enumerate(values):
        yy = y + (4 - index) * 1.50
        ax.add_patch(Rectangle((x, yy), 4.4, 0.62, facecolor="#EEF1F3", edgecolor="none", zorder=2))
        if value > 0:
            ax.add_patch(
                Rectangle((x, yy), 4.4 * value / 0.5, 0.62, facecolor=FUSION, edgecolor="none", zorder=3)
            )
        else:
            label(ax, x + 0.42, yy + 0.31, "0", size=5.2, color=ZERO, weight="bold")
        label(ax, x - 0.35, yy + 0.31, rf"$j_{index + 1}$", size=4.8, color=MUTED, ha="right")


def create_figure() -> tuple[Path, Path, Path]:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.35, 4.5), dpi=600)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70.87)
    ax.set_aspect("equal")
    ax.axis("off")

    # Panel (a): encoding and semantics-guided structural context.
    rounded_box(ax, 0.8, 15.3, 98.4, 54.5, face="white", edge=LIGHT_LINE, lw=0.8, radius=0.9)
    label(ax, 2.2, 67.9, "(a)", size=9.1, weight="bold", ha="left")
    label(ax, 7.0, 67.9, "Encoding and neighbor-context construction", size=8.2, weight="bold", ha="left")

    # Compact color legend.
    legend = [(STRUCT, "Structure"), (SEM, "Semantics"), (FUSION, "Interaction")]
    cursor = 74.1
    for color, text in legend:
        ax.add_patch(Circle((cursor, 67.9), 0.48, facecolor=color, edgecolor=color, zorder=5))
        label(ax, cursor + 0.9, 67.9, text, size=5.4, color=MUTED, ha="left")
        cursor += 8.2

    # Inputs.
    section_group(ax, 2.4, 20.0, 14.4, 44.7, "Inputs", edge=LINE, face=PANEL)
    small_module(ax, 3.5, 44.2, 12.2, 16.1, "Directed KGs", "", edge=STRUCT, face=STRUCT_LIGHT)
    draw_graph(ax, 7.0, 51.8, 1.18, color=STRUCT)
    draw_graph(ax, 12.2, 51.8, 1.18, color=STRUCT)
    label(
        ax,
        9.6,
        47.0,
        "$G_L$ and $G_R$",
        size=6.0,
        color=MUTED,
    )
    label(ax, 9.6, 45.4, r"directed triples $(h,r,t)$", size=5.6, color=STRUCT)

    small_module(ax, 3.5, 21.3, 12.2, 19.9, "Entity text", "", edge=SEM, face=SEM_LIGHT)
    draw_token_strip(
        ax,
        4.25,
        33.4,
        [2.8, 3.2, 3.2],
        ["name", "rel.", "attr."],
        [SEM, "#5A8D78", "#739B86"],
    )
    label(ax, 9.6, 27.5, "one shared\ntoken sequence", size=5.5, color=MUTED)

    # Structural branch.
    section_group(ax, 19.0, 44.2, 36.2, 20.5, "Relation-aware structural encoder", edge=STRUCT, face=STRUCT_FILL)
    small_module(
        ax,
        20.2,
        47.0,
        9.4,
        12.8,
        "Topology init.",
        "8 features\nshared\nprojection\n+ entity residual",
        edge=STRUCT,
        face="white",
        title_size=6.1,
        body_size=5.2,
    )
    label(ax, 24.9, 48.3, r"$\tilde{f}_i\rightarrow h_i^{(0)}$", size=5.8, color=STRUCT)

    small_module(ax, 30.8, 47.0, 10.8, 12.8, "Relation msg.", "", edge=STRUCT, face="white", title_size=5.8)
    label(ax, 36.2, 55.8, "shared relation map", size=5.0, color=MUTED)
    draw_layer_stack(ax, 33.0, 48.0)
    arrow(ax, (29.6, 53.4), (30.8, 53.4), color=STRUCT, lw=1.0)

    small_module(ax, 42.8, 47.0, 11.2, 12.8, "Layer selection", "entity-specific\ndepth weights", edge=STRUCT, face="white", title_size=6.0, body_size=5.7)
    label(ax, 48.4, 49.3, r"$\sum_{p=0}^{L} a_i^{(p)}u_i^{(p)}$", size=5.4, color=STRUCT)
    arrow(ax, (41.6, 53.4), (42.8, 53.4), color=STRUCT, lw=1.0)

    # Semantic branch.
    section_group(ax, 19.0, 20.0, 36.2, 20.9, "Multi-scale semantic encoder", edge=SEM, face=SEM_FILL)
    small_module(
        ax,
        20.2,
        23.0,
        7.2,
        12.8,
        "Input map",
        "300→128\nLayerNorm\n+ position",
        edge=SEM,
        face="white",
        title_size=5.7,
        body_size=5.5,
    )
    arrow(ax, (27.4, 29.4), (28.0, 29.4), color=SEM, lw=1.0)

    small_module(ax, 28.0, 23.0, 6.2, 12.8, "Token", "mean +\nattention\npooling", edge=SEM, face="white", body_size=5.6)
    small_module(ax, 34.8, 23.0, 6.2, 12.8, "Phrase", "Conv 3/5\n+ pooling", edge=SEM, face="white", body_size=5.6)
    small_module(ax, 41.6, 23.0, 7.0, 12.8, "Global", "2-layer\nTransformer\n+ pooling", edge=SEM, face="white", body_size=5.4)
    small_module(
        ax,
        49.2,
        23.0,
        4.8,
        12.8,
        "Gate",
        "view\nweights",
        edge=SEM,
        face="white",
        title_size=5.8,
        body_size=5.8,
    )
    arrow(ax, (34.2, 29.4), (34.8, 29.4), color=SEM, lw=0.75)
    arrow(ax, (41.0, 29.4), (41.6, 29.4), color=SEM, lw=0.75)
    arrow(ax, (48.6, 29.4), (49.2, 29.4), color=SEM, lw=0.75)

    # Semantics-guided context and final joint representation.
    section_group(ax, 58.0, 20.0, 39.7, 44.7, "Semantics-guided structural context", edge=FUSION, face=FUSION_FILL)

    small_module(ax, 59.3, 47.0, 10.3, 12.7, "1-hop neighbors", "", edge=STRUCT, face="white", title_size=5.6)
    draw_neighbor_set(ax, 59.5, 48.1)
    label(ax, 64.5, 48.2, "all outgoing\nneighbors", size=5.3, color=MUTED)

    small_module(ax, 70.5, 47.0, 9.1, 12.7, "Scoring", "semantic\nquery\n+ pair gate", edge=FUSION, face="white", title_size=5.9, body_size=5.6)
    label(ax, 75.05, 48.7, r"$a_{ij}=q_i^\top k_j/\sqrt{d}$", size=5.6, color=FUSION)

    small_module(ax, 80.5, 47.0, 7.5, 12.7, "1.5-entmax", "", edge=FUSION, face="white", title_size=5.3)
    draw_sparse_weights(ax, 81.7, 48.2)
    label(ax, 84.25, 56.2, r"$T=0.25$", size=5.1, color=FUSION)

    small_module(ax, 88.9, 47.0, 7.6, 12.7, "Summary", "", edge=FUSION, face="white", title_size=6.0, body_size=5.4)
    label(ax, 92.7, 52.4, "$\\bar{c}_i=$\n$\\sum_j\\alpha_{ij}v_j$", size=6.0, color=FUSION)

    arrow(ax, (69.6, 53.4), (70.5, 53.4), color=STRUCT, lw=1.0)
    arrow(ax, (79.6, 53.4), (80.5, 53.4), color=FUSION, lw=1.0)
    arrow(ax, (88.0, 53.4), (88.9, 53.4), color=FUSION, lw=1.0)

    small_module(
        ax,
        71.0,
        38.5,
        25.5,
        6.0,
        "Structural context gate",
        r"self state $t_i$  +  summary $\bar{c}_i$  $\rightarrow$  $c_i^{str}$",
        edge=STRUCT,
        face="white",
        title_size=6.4,
        body_size=5.7,
    )
    arrow(ax, (92.7, 47.0), (92.7, 44.5), color=FUSION, lw=1.0)

    small_module(
        ax,
        71.0,
        26.1,
        25.5,
        8.9,
        "Joint gate + fixed residual",
        "",
        edge=OUTPUT,
        face=OUTPUT_FILL,
        title_size=6.5,
        body_size=5.7,
    )
    label(ax, 78.0, 28.75, r"$s_i+c_i^{str}$", size=6.0, color=INK)
    arrow(ax, (81.5, 28.75), (85.0, 28.75), color=OUTPUT, lw=0.9)
    rounded_box(ax, 85.0, 27.3, 9.7, 2.9, face="white", edge=OUTPUT, lw=0.85, radius=0.65)
    label(ax, 89.85, 28.75, r"$z_i^{joint}$", size=6.8, color=OUTPUT, weight="bold")
    arrow(ax, (83.7, 38.5), (83.7, 35.0), color=OUTPUT, lw=1.0)

    # Clean branch hand-offs. Secondary dependencies are written inside the
    # destination modules so connector paths remain unambiguous.
    arrow(ax, (54.0, 53.4), (59.3, 53.4), color=STRUCT, lw=1.05)
    rounded_box(ax, 55.1, 53.9, 3.0, 1.9, face="white", edge=STRUCT, lw=0.55, radius=0.35, zorder=5)
    label(ax, 56.6, 54.85, r"$t_i$", size=5.8, color=STRUCT, weight="bold", zorder=6)
    arrow(ax, (54.0, 29.4), (71.0, 29.4), color=SEM, lw=1.05)
    rounded_box(ax, 60.8, 29.9, 3.0, 1.9, face="white", edge=SEM, lw=0.55, radius=0.35, zorder=5)
    label(ax, 62.3, 30.85, r"$s_i$", size=5.8, color=SEM, weight="bold", zorder=6)

    # Input routes.
    arrow(ax, (15.7, 53.5), (20.2, 53.5), color=STRUCT, lw=1.0)
    arrow(ax, (15.7, 29.4), (20.2, 29.4), color=SEM, lw=1.0)

    # Panel (b): training and retrieval protocol.
    rounded_box(ax, 0.8, 1.0, 98.4, 12.2, face="white", edge=LIGHT_LINE, lw=0.8, radius=0.9)
    label(ax, 2.2, 11.4, "(b)", size=9.1, weight="bold", ha="left")
    label(ax, 7.0, 11.4, "Training and retrieval protocol", size=8.0, weight="bold", ha="left")

    small_module(ax, 18.0, 3.0, 12.0, 7.1, "Seed pairs", r"mini-batch $(L,R)$", edge=OUTPUT, face=OUTPUT_FILL, body_size=5.7)
    small_module(
        ax,
        34.0,
        3.0,
        25.0,
        7.1,
        "Bidirectional InfoNCE",
        "$\\mathcal{L}=\\mathcal{L}_{joint}+\\lambda_{str}(p)\\mathcal{L}_{str}$\n$\\lambda_{str}(p):\\ 0.1\\rightarrow 0$",
        edge=FUSION,
        face=FUSION_LIGHT,
        body_size=5.7,
    )
    small_module(
        ax,
        63.0,
        3.0,
        18.0,
        7.1,
        "Direct joint retrieval",
        "same $z^{joint}$\ncosine + CSLS",
        edge=OUTPUT,
        face=OUTPUT_FILL,
        body_size=5.7,
    )
    small_module(
        ax,
        85.0,
        3.0,
        12.0,
        7.1,
        "Validation",
        r"select $k_{CSLS}$ only",
        edge=LINE,
        face=PANEL,
        body_size=5.6,
    )
    arrow(ax, (30.0, 6.55), (34.0, 6.55), color=LINE, lw=1.0)
    arrow(ax, (59.0, 6.55), (63.0, 6.55), color=LINE, lw=1.0)
    arrow(ax, (81.0, 6.55), (85.0, 6.55), color=LINE, lw=1.0)

    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995)
    fig.savefig(PNG_OUTPUT, dpi=600, facecolor="white", bbox_inches=None, pad_inches=0)
    fig.savefig(PDF_OUTPUT, facecolor="white", bbox_inches=None, pad_inches=0)
    fig.savefig(SVG_OUTPUT, facecolor="white", bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return PNG_OUTPUT, PDF_OUTPUT, SVG_OUTPUT


if __name__ == "__main__":
    for output in create_figure():
        print(output)
