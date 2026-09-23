from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MODULE = ROOT / "tmp" / "update_paper_figure1_journal.py"
OUTPUT = ROOT / "outputs" / "paper_assets" / "figure_1_new_main_validated_protocol.png"

spec = spec_from_file_location("journal_figure", SOURCE_MODULE)
journal = module_from_spec(spec)
spec.loader.exec_module(journal)


def create_figure(path: Path) -> None:
    image = Image.new("RGB", (journal.W, journal.H), "white")
    draw = ImageDraw.Draw(image)

    journal.box(
        draw,
        (35, 35, 2365, 1085),
        fill="white",
        outline=journal.PANEL,
        title="",
        lines=[],
        radius=12,
        title_height=0,
    )
    draw.text((65, 57), "(a)", font=journal.F_PANEL, fill=journal.INK)
    draw.text((130, 58), "关系感知结构上下文与多尺度语义联合编码", font=journal.F_PANEL, fill=journal.INK)

    journal.box(
        draw,
        (65, 140, 300, 970),
        fill="#f7f8fa",
        outline=journal.EVAL_LINE,
        title="双图输入",
        lines=[
            ("左知识图谱", journal.F_SMALL, journal.INK),
            (r"\mathcal{G}_L", 14, journal.OUTPUT_LINE),
            ("右知识图谱", journal.F_SMALL, journal.INK),
            (r"\mathcal{G}_R", 14, journal.OUTPUT_LINE),
            ("关系三元组", journal.F_SMALL, journal.INK),
            (r"(h,r,t)", 12, journal.INK),
            ("名称 · 关系词\n属性名称 · 属性值", journal.F_SMALL, journal.INK),
        ],
        title_height=62,
    )

    journal.box(
        draw,
        (345, 140, 1300, 500),
        fill="#f8fbfd",
        outline=journal.STRUCT_LINE,
        title="结构分支：拓扑初始化与关系感知传播",
        lines=[],
        title_height=62,
    )
    journal.box(
        draw,
        (375, 235, 630, 450),
        fill="white",
        outline="#84aabd",
        title="拓扑初始化",
        lines=[
            ("8 维局部统计", journal.F_TINY, journal.INK),
            ("度 · 关系多样性\n邻居度 · 方向平衡", journal.F_TINY, journal.INK),
            (r"\mathbf{h}_i^{(0)}=\mathbf{W}_{top}\mathbf{f}_i+0.1\mathbf{e}_i", 8, journal.STRUCT_LINE),
        ],
        title_height=48,
        title_font=journal.F_TAG,
    )
    journal.box(
        draw,
        (665, 220, 995, 465),
        fill="white",
        outline="#84aabd",
        title="三层关系感知消息传播",
        lines=[
            (r"\mathbf{m}_{j\to i}^{(\ell)}=\mathbf{W}_s^{(\ell)}\mathbf{h}_j^{(\ell)}+\mathbf{W}_r^{(\ell)}\mathbf{e}_r", 8, journal.STRUCT_LINE),
            ("聚合传入边 · 共享投影 · 自身/邻居门控", journal.F_TINY, journal.INK),
        ],
        title_height=48,
        title_font=journal.F_TAG,
    )
    journal.box(
        draw,
        (1030, 235, 1270, 450),
        fill="white",
        outline="#84aabd",
        title="节点级层选择",
        lines=[
            (r"\mathbf{h}_i^{(0)},\ldots,\mathbf{h}_i^{(L)}", 9, journal.STRUCT_LINE),
            ("按实体融合\n不同传播深度", journal.F_TINY, journal.INK),
            (r"\mathbf{z}_i^{str}", 13, journal.STRUCT_LINE),
        ],
        title_height=48,
        title_font=journal.F_TAG,
    )
    journal.arrow(draw, [(300, 320), (345, 320)], color=journal.STRUCT_LINE)
    journal.arrow(draw, [(630, 342), (665, 342)], color=journal.STRUCT_LINE)
    journal.arrow(draw, [(995, 342), (1030, 342)], color=journal.STRUCT_LINE)

    journal.box(
        draw,
        (345, 540, 1300, 970),
        fill="#f8fcfa",
        outline=journal.SEM_LINE,
        title="语义分支：共享输入上的多尺度编码",
        lines=[],
        title_height=62,
    )
    journal.plain_box(
        draw,
        (375, 635, 605, 745),
        "300 维 token 序列\n投影 + 位置编码",
        fill="white",
        outline="#85ae9d",
        used_font=journal.F_TINY,
    )
    journal.plain_box(draw, (650, 620, 820, 755), "token 视图\n均值 + 注意力", fill="white", outline="#85ae9d", used_font=journal.F_TINY)
    journal.plain_box(draw, (845, 620, 1015, 755), "phrase 视图\nConv3/5", fill="white", outline="#85ae9d", used_font=journal.F_TINY)
    journal.plain_box(draw, (1040, 620, 1245, 755), "global 视图\n2 层 Transformer", fill="white", outline="#85ae9d", used_font=journal.F_TINY)
    journal.box(
        draw,
        (650, 810, 1245, 925),
        fill="white",
        outline="#85ae9d",
        title="视图门控与语义表示",
        lines=[
            ("三个视图自适应加权 · MLP · 二范数归一化", journal.F_TINY, journal.INK),
            (r"\mathbf{z}_i^{sem}", 13, journal.SEM_LINE),
        ],
        title_height=42,
        title_font=journal.F_TAG,
    )
    journal.arrow(draw, [(300, 700), (345, 700)], color=journal.SEM_LINE)
    journal.arrow(draw, [(605, 690), (650, 690)], color=journal.SEM_LINE)
    journal.arrow(draw, [(735, 755), (735, 810)], color=journal.SEM_LINE)
    journal.arrow(draw, [(930, 755), (930, 810)], color=journal.SEM_LINE)
    journal.arrow(draw, [(1140, 755), (1140, 810)], color=journal.SEM_LINE)

    journal.box(
        draw,
        (1340, 140, 2035, 970),
        fill=journal.FUSION_FILL,
        outline=journal.FUSION_LINE,
        title="一跳出邻居上下文与结构—语义融合",
        lines=[],
        title_height=62,
    )
    journal.box(
        draw,
        (1375, 230, 2000, 405),
        fill="white",
        outline="#c9a361",
        title="① 变长邻域整理",
        lines=[
            ("沿原始边方向读取全部一跳出邻居", journal.F_SMALL, journal.INK),
            ("按批次最大邻居数动态补齐 · 无新增参数", journal.F_TINY, journal.INK),
        ],
        title_height=46,
        title_font=journal.F_TAG,
    )
    journal.box(
        draw,
        (1375, 445, 2000, 665),
        fill="white",
        outline="#c9a361",
        title="② 稀疏邻居权重",
        lines=[
            (r"a_{ij}=\frac{\langle \mathbf{W}_Q\mathbf{z}_i^{sem},\mathbf{W}_K\mathbf{z}_j^{str}\rangle}{0.25\sqrt{d}}+\log(p_{ij}+\epsilon)", 7, journal.FUSION_LINE),
            (r"\boldsymbol{\alpha}_i=\operatorname{entmax}_{1.5}(\mathbf{a}_i)", 11, journal.FUSION_LINE),
            ("低相关邻居可获得精确零权重", journal.F_TINY, journal.INK),
        ],
        title_height=46,
        title_font=journal.F_TAG,
    )
    journal.box(
        draw,
        (1375, 705, 2000, 910),
        fill="white",
        outline="#c9a361",
        title="③ 结构上下文与联合门控",
        lines=[
            (r"\bar{\mathbf{c}}_i=\sum_j\alpha_{ij}\mathbf{v}_j", 11, journal.FUSION_LINE),
            ("自身结构/邻域证据门控", journal.F_TINY, journal.INK),
            (r"\mathbf{z}_i^{joint}=\operatorname{Norm}(\operatorname{Gate}(\mathbf{c}_i^{str},\mathbf{z}_i^{sem}))", 8, journal.FUSION_LINE),
        ],
        title_height=46,
        title_font=journal.F_TAG,
    )
    journal.arrow(draw, [(1300, 342), (1340, 342)], color=journal.STRUCT_LINE)
    journal.arrow(draw, [(1300, 870), (1320, 870), (1320, 540), (1375, 540)], color=journal.SEM_LINE)
    journal.arrow(draw, [(1687, 405), (1687, 445)], color=journal.FUSION_LINE)
    journal.arrow(draw, [(1687, 665), (1687, 705)], color=journal.FUSION_LINE)

    journal.box(
        draw,
        (2075, 300, 2330, 790),
        fill=journal.OUTPUT_FILL,
        outline=journal.OUTPUT_LINE,
        title="共享编码输出",
        lines=[
            (r"\mathbf{z}_i^{str}", 14, journal.STRUCT_LINE),
            (r"\mathbf{z}_i^{sem}", 14, journal.SEM_LINE),
            (r"\mathbf{c}_i^{str}", 14, journal.FUSION_LINE),
            (r"\mathbf{z}_i^{joint}", 14, journal.OUTPUT_LINE),
            ("左右图共享\n模型参数", journal.F_SMALL, journal.INK),
        ],
    )
    journal.arrow(draw, [(2000, 810), (2050, 810), (2050, 545), (2075, 545)], color=journal.OUTPUT_LINE)

    journal.box(
        draw,
        (35, 1115, 2365, 1465),
        fill="white",
        outline=journal.PANEL,
        title="",
        lines=[],
        radius=12,
        title_height=0,
    )
    draw.text((65, 1135), "(b)", font=journal.F_PANEL, fill=journal.INK)
    draw.text((130, 1136), "训练目标与验证驱动的检索协议", font=journal.F_PANEL, fill=journal.INK)

    journal.box(
        draw,
        (70, 1210, 965, 1430),
        fill="#f7f8fa",
        outline=journal.EVAL_LINE,
        title="种子对齐上的联合与结构监督",
        lines=[
            (r"\mathcal{L}=\mathcal{L}_{joint}+\lambda_{str}(p)\mathcal{L}_{str}", 12, journal.EVAL_LINE),
            (r"\lambda_{str}(p)=0.1(1-p)", 12, journal.EVAL_LINE),
            ("两项均为双向 InfoNCE；结构权重随训练进度衰减至 0", journal.F_TINY, journal.INK),
        ],
        title_height=48,
        title_font=journal.F_TAG,
    )
    journal.box(
        draw,
        (1010, 1210, 2295, 1430),
        fill="#fbfcfd",
        outline=journal.OUTPUT_LINE,
        title="直接联合检索与验证集选择",
        lines=[
            (r"\mathbf{z}^{eval}=\mathbf{z}^{joint}", 12, journal.OUTPUT_LINE),
            (r"k_{CSLS}\in\{3,5,7,10,15,20\}", 11, journal.OUTPUT_LINE),
            ("每个运行由验证 MRR 选择 k_CSLS → 全候选 CSLS 检索 → 测试集一次报告", journal.F_TINY, journal.INK),
        ],
        title_height=48,
        title_font=journal.F_TAG,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True, dpi=(300, 300))


if __name__ == "__main__":
    create_figure(OUTPUT)
    print(OUTPUT)
