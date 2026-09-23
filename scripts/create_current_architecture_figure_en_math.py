from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from PIL import ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "create_new_main_architecture_figure.py"
OUTPUT = ROOT / "outputs" / "paper_assets" / "figure_1_current_protocol_springer_en.png"
ARIAL = "/System/Library/Fonts/Supplemental/Arial.ttf"
ARIAL_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


TRANSLATIONS = {
    "关系感知结构上下文与多尺度语义联合编码": "Relation-Aware Structural Context and Multi-Scale Semantic Fusion",
    "双图输入": "Paired graph input",
    "左知识图谱": "Left knowledge graph",
    "右知识图谱": "Right knowledge graph",
    "关系三元组": "Relation triples",
    "名称 · 关系词\n属性名称 · 属性值": "Names · relation terms\nattribute names · values",
    "结构分支：拓扑初始化与关系感知传播": "Structural branch: topology initialization and relation-aware propagation",
    "拓扑初始化": "Topology initialization",
    "8 维局部统计": "Eight local statistics",
    "度 · 关系多样性\n邻居度 · 方向平衡": "Degree · relation diversity\nneighbor degree · direction",
    "三层关系感知消息传播": "Three-layer relation-aware propagation",
    "聚合传入边 · 共享投影 · 自身/邻居门控": "Incoming edges · shared projections\nself/neighbor gate",
    "节点级层选择": "Node-wise layer selection",
    "按实体融合\n不同传播深度": "Entity-specific fusion\nof propagation depths",
    "语义分支：共享输入上的多尺度编码": "Semantic branch: multi-scale encoding of a shared input sequence",
    "300 维 token 序列\n投影 + 位置编码": "300-d token sequence\nprojection + position encoding",
    "token 视图\n均值 + 注意力": "token view\nmean + attention",
    "phrase 视图\nConv3/5": "phrase view\nConv3/5",
    "global 视图\n2 层 Transformer": "global view\ntwo-layer Transformer",
    "视图门控与语义表示": "View gate and semantic representation",
    "三个视图自适应加权 · MLP · 二范数归一化": "Adaptive view weights · MLP · L2 normalization",
    "一跳出邻居上下文与结构—语义融合": "One-hop outgoing context and structure-semantic fusion",
    "① 变长邻域整理": "1  Variable-size neighborhood batching",
    "沿原始边方向读取全部一跳出邻居": "Read every one-hop outgoing neighbor along original edges",
    "按批次最大邻居数动态补齐 · 无新增参数": "Dynamic padding to the batch maximum · no new parameters",
    "② 稀疏邻居权重": "2  Sparse neighbor weighting",
    "低相关邻居可获得精确零权重": "Low-relevance neighbors may receive exactly zero weight",
    "③ 结构上下文与联合门控": "3  Structural context and joint gate",
    "自身结构/邻域证据门控": "Gate between self-structure and neighbor evidence",
    "共享编码输出": "Shared encoder outputs",
    "左右图共享\n模型参数": "Parameters shared\nacross both graphs",
    "训练目标与验证驱动的检索协议": "Training Objective and Validation-Guided Retrieval",
    "种子对齐上的联合与结构监督": "Joint and structural supervision on seed alignments",
    "两项均为双向 InfoNCE；结构权重随训练进度衰减至 0": "Both terms use bidirectional InfoNCE; structural weight decays to zero",
    "直接联合检索与验证集选择": "Direct joint retrieval and validation selection",
    "每个运行由验证 MRR 选择 k_CSLS → 全候选 CSLS 检索 → 测试集一次报告": "Validation MRR selects k_CSLS per run → all-candidate CSLS → one test report",
}


def translated(text: str) -> str:
    return TRANSLATIONS.get(text, text)


def main() -> None:
    spec = spec_from_file_location("current_architecture_figure", SOURCE)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    journal = module.journal

    journal.F_PANEL = ImageFont.truetype(ARIAL_BOLD, 40)
    journal.F_GROUP = ImageFont.truetype(ARIAL_BOLD, 34)
    journal.F_TITLE = ImageFont.truetype(ARIAL, 29)
    journal.F_BODY = ImageFont.truetype(ARIAL, 27)
    journal.F_SMALL = ImageFont.truetype(ARIAL, 24)
    journal.F_TINY = ImageFont.truetype(ARIAL, 21)
    journal.F_TAG = ImageFont.truetype(ARIAL_BOLD, 22)

    original_box = journal.box
    original_plain_box = journal.plain_box
    original_text = ImageDraw.ImageDraw.text

    def translated_box(draw, xy, **kwargs):
        kwargs["title"] = translated(kwargs.get("title", ""))
        kwargs["lines"] = [
            (translated(text), used_font, color)
            for text, used_font, color in kwargs.get("lines", [])
        ]
        return original_box(draw, xy, **kwargs)

    def translated_plain_box(draw, xy, text, **kwargs):
        return original_plain_box(draw, xy, translated(text), **kwargs)

    def translated_text(self, xy, text, *args, **kwargs):
        return original_text(self, xy, translated(text), *args, **kwargs)

    journal.box = translated_box
    journal.plain_box = translated_plain_box
    ImageDraw.ImageDraw.text = translated_text
    try:
        module.create_figure(OUTPUT)
    finally:
        ImageDraw.ImageDraw.text = original_text

    print(OUTPUT)


if __name__ == "__main__":
    main()
