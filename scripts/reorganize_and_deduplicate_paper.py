from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from docx import Document
from docx.text.paragraph import Paragraph


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mathify_all_document_symbols import set_markup  # noqa: E402


def combined_text(paragraph: Paragraph) -> str:
    return "".join(
        paragraph._p.xpath(".//w:t/text()|.//m:t/text()")
    ).replace("\u00a0", " ").strip()


def find_paragraph(document: Document, prefix: str) -> Paragraph:
    matches = [
        paragraph
        for paragraph in document.paragraphs
        if combined_text(paragraph).startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one paragraph beginning {prefix!r}, found {len(matches)}"
        )
    return matches[0]


def rewrite(document: Document, prefix: str, markup: str) -> None:
    # Recover LaTeX commands that Python interprets as control-character
    # escapes inside long Chinese prose literals.
    markup = (
        markup.replace("\x07lpha", r"\alpha")
        .replace("\x08ar", r"\bar")
        .replace("\t" + "imes", r"\times")
        .replace("\t" + "o", r"\to")
    )
    set_markup(find_paragraph(document, prefix), markup)


def remove_paragraph(document: Document, prefix: str) -> None:
    paragraph = find_paragraph(document, prefix)
    paragraph._p.getparent().remove(paragraph._p)


def move_section_before(
    document: Document,
    section_prefix: str,
    next_section_prefix: str,
    anchor_prefix: str,
) -> None:
    start = find_paragraph(document, section_prefix)._p
    stop = find_paragraph(document, next_section_prefix)._p
    anchor = find_paragraph(document, anchor_prefix)._p

    elements = []
    current = start
    while current is not stop:
        elements.append(current)
        current = current.getnext()

    for element in elements:
        anchor.addprevious(element)


def revise_front_matter(document: Document) -> None:
    rewrite(
        document,
        "本研究的核心目标是检验",
        "本研究旨在检验结构信息与语义信息的融合方式如何影响跨语言实体检索，"
        "并围绕关系结构表征、多粒度语义表征以及早期融合与晚期融合开展受控比较。"
        "具体研究问题与评价协议见第 4.1 节。",
    )


def revise_related_work(document: Document) -> None:
    rewrite(
        document,
        "GraphSAGE 通过采样",
        "GraphSAGE 通过采样并聚合局部邻居学习节点表示[6]；R-GCN 在消息变换中区分关系类型，"
        "并以基分解或块分解控制多关系参数规模[7]。这些工作说明，邻域聚合不仅需要考虑相邻实体，"
        "还需要保留边类型信息。然而，知识图谱中的关系频次和节点所需的结构范围并不均匀，"
        "因而仍需研究如何在有限参数预算下同时表示关系差异和不同传播深度。",
    )
    rewrite(
        document,
        "在实体对齐任务中，RDGCN",
        "在实体对齐任务中，RDGCN 通过实体图与关系对偶图之间的注意力交互引入关系证据[4]，"
        "RREA 通过关系反射变换兼顾关系区分与表示空间的几何保持[5]。RPR-RHGT 则进一步显式生成并筛选"
        "多步关系路径，再利用关系感知异构图 Transformer 编码这些路径[10]。上述研究共同表明关系和多跳结构具有价值，"
        "但显式路径推理与堆叠图消息传播属于不同机制。本文研究后者，并在第 3.3 节给出与代码一致的轻量实现。",
    )
    rewrite(
        document,
        "Transformer 通过多头自注意力",
        "Transformer 通过多头自注意力建模序列中不同位置之间的依赖[8]。HAN 进一步表明，"
        "不同语义粒度对整体表示的贡献可能不同[11]。这些工作为保留细粒度、局部组合和全局上下文提供了方法依据，"
        "但并不能预先保证每一种语义视图都能改善实体对齐；因此，本文在统一编码框架中保留三种粒度，"
        "并通过独立消融检验其实际贡献。",
    )
    rewrite(
        document,
        "已有文本增强实体对齐方法表明",
        "已有文本增强实体对齐研究表明，名称、描述和其他模态能够补充结构证据。RREA 将文本增强结果与纯结构设置分开报告[5]；"
        "MCLEA 通过模态内对比和模态间对齐联合学习多模态表示[12]；多模态知识图谱 Transformer 研究还指出，"
        "异质信息的直接组合可能形成未对齐的表示空间[13]。现有证据支持同时使用结构与语义，"
        "但仍需通过参数受控实验区分性能变化究竟来自融合位置、可见的结构邻居，还是融合模块容量。",
    )


def reorganize_method(document: Document) -> None:
    move_section_before(
        document,
        "3.8 多尺度语义编码器",
        "3.9 邻居上下文、查询信号与联合表示",
        "3.7 固定预算邻居提取",
    )

    rewrite(document, "3 背景与方法", "3 问题定义与方法")
    rewrite(document, "3.6 关系感知轻量结构编码器", "3.3 关系感知轻量结构编码器")
    rewrite(document, "3.8 多尺度语义编码器", "3.4 多尺度语义编码器")
    rewrite(document, "3.7 固定预算邻居提取", "3.5 固定预算邻居提取")
    rewrite(document, "3.9 邻居上下文、查询信号与联合表示", "3.6 邻居上下文与联合表示")
    rewrite(document, "3.10 单阶段双向 InfoNCE 对齐训练", "3.7 单阶段双向 InfoNCE 对齐训练")
    rewrite(document, "3.11 检索与 CSLS", "3.8 检索与 CSLS")

    rewrite(
        document,
        "模型首先在两张知识图谱上进行关系感知消息传递",
        "模型由结构编码、语义编码、结构邻居上下文和对比学习四部分组成。结构编码器在两张知识图谱上执行"
        "关系感知消息传递，并融合不同传播深度的节点状态；语义编码器从名称、关系和属性序列中构造 token、phrase 和 global 三个视图。"
        "随后，模型为当前实体提取固定预算的结构邻居，形成邻域上下文，并将其与语义表示融合为联合表示。"
        "训练仅使用种子对齐上的双向 InfoNCE；验证阶段选择融合权重、有效传播深度和 CSLS 邻域大小。",
    )
    remove_paragraph(document, "结构与语义信息在模型中有两个交互位置")
    rewrite(
        document,
        "图 1 关系感知结构上下文",
        "图 1 关系感知结构上下文与多尺度语义融合方法及训练—检索协议。"
        "（a）结构分支利用关系类型完成三层全图消息传播，并通过节点级层选择获得结构表示；"
        "语义分支由 token、phrase 和 global 视图形成语义表示；固定预算结构邻居经注意力与门控形成结构上下文，"
        "再与语义表示融合为联合表示。（b）种子对齐用于单阶段双向 InfoNCE 训练；验证集选择 "
        "[[\alpha]]、[[L]] 和 [[k_{\mathrm{CSLS}}]]，测试集仅执行一次最终评估。",
    )


def revise_structural_encoder(document: Document) -> None:
    rewrite(
        document,
        "消息传递（message passing）",
        "结构编码器在整张知识图谱上执行有向的关系感知消息传递。对于三元组 [[(j,r,i)]]，"
        "头实体 [[j]] 沿该边向尾实体 [[i]] 发送消息；当前数据处理不自动添加反向边。设 "
        "[[\mathbf{h}_i^{(\ell)}\in\mathbb{R}^{d}]] 为实体 [[i]] 在第 [[\ell]] 层的状态，"
        "[[\mathbf{e}_r\in\mathbb{R}^{d_r}]] 为关系 [[r]] 的嵌入，"
        "[[\mathcal{N}_{\mathrm{in}}(i)]] 为所有指向 [[i]] 的传入边。第 [[\ell]] 层首先对每条传入边计算：",
    )
    rewrite(
        document,
        "其中 Ws",
        "其中 [[\mathbf{W}_s^{(\ell)}\in\mathbb{R}^{d\times d}]] 和 "
        "[[\mathbf{W}_r^{(\ell)}\in\mathbb{R}^{d\times d_r}]] 分别投影源实体状态和关系嵌入。"
        "同一层的全部关系共享这两个投影矩阵，关系差异由 [[\mathbf{e}_r]] 表示，"
        "从而避免为每种关系配置一套独立的大矩阵。"
        "[[\mathbf{m}_{j\to i}^{(\ell)}\in\mathbb{R}^{d}]] 是边 [[(j,r,i)]] 传递给目标实体 [[i]] 的消息。"
        "模型对实体 [[i]] 的全部传入消息取均值：",
    )
    rewrite(
        document,
        "自身状态投影为",
        "实体自身状态先投影为 [[\mathbf{s}_i^{(\ell)}="
        "\mathbf{W}_{\mathrm{self}}^{(\ell)}\mathbf{h}_i^{(\ell)}]]。随后，模型根据自身状态和平均邻居消息计算逐维门控：",
    )
    rewrite(
        document,
        "以三元组“姚明—出生地",
        "式（3）中的 [[\mathbf{g}_i^{(\ell)}]] 为每个表示维度产生一个 0 到 1 之间的权重；"
        "式（4）保留投影后的自身状态，并按该权重加入邻居证据。例如，对三元组“姚明—出生地→上海”，"
        "更新“上海”时，消息同时包含“姚明”的实体状态和“出生地”的关系嵌入。"
        "因此，即使另一个实体也连接到“上海”，模型仍可区分“出生地”和“工作地点”等不同关系。",
    )
    rewrite(
        document,
        "中间层进一步使用 LayerNorm",
        "式（4）产生每一层的原始输出。除最后一层外，该输出在进入下一层之前依次经过 "
        "[[\operatorname{LayerNorm}]]、ReLU 和 dropout：LayerNorm 统一特征尺度，ReLU 引入非线性，"
        "dropout 在训练时以 0.1 的概率随机屏蔽部分特征以减轻过拟合，并在验证和测试时关闭。"
        "这三项操作只处理当前层表示，不会自行增加传播距离；多跳信息来自图卷积层的连续堆叠。",
    )
    rewrite(
        document,
        "本文所称消息路径",
        "主配置堆叠三层图卷积，因此尾实体在第一层接收直接传入邻居的一跳证据，"
        "第二层和第三层继续接收已经聚合过的节点状态，分别覆盖至多两跳和三跳证据。"
        "例如，在有向链“姚明—出生地→上海—所在国家→中国”中，第一层使“上海”吸收“姚明”及“出生地”的信息，"
        "第二层再使“中国”吸收已经包含上述证据的“上海”状态。这里的多跳来自逐层消息传播，"
        "模型不生成、筛选或编码显式关系路径，因此不属于路径级推理。",
    )
    rewrite(
        document,
        "为避免只依赖最深层",
        "仅使用最深层可能丢失实体自身或近邻信息。为此，编码器保留投影后的输入状态和每层原始输出 "
        "[[\mathbf{u}_i^{(0)},\ldots,\mathbf{u}_i^{(L)}]]。其中，"
        "[[\mathbf{u}_i^{(0)}]] 不包含邻居传播，[[\mathbf{u}_i^{(p)}]] 包含至多 [[p]] 跳证据。"
        "模型先计算这些状态的平均上下文 [[\bar{\mathbf{u}}_i]]，再将每个候选状态与平均上下文拼接，"
        "由两层 MLP 输出标量分数，并通过 [[\operatorname{softmax}]] 得到深度权重 [[a_i^{(p)}]]：",
    )
    rewrite(
        document,
        "该 MLP 的输入维度为",
        "层选择 MLP 的输入维度为 [[2d]]，隐藏维度为 [[d]]，采用 GELU 和 dropout，最后输出一个标量。"
        "式（6）按实体分别加权 [[0]] 到 [[L]] 层状态，再进行 LayerNorm 和 L2 归一化，得到结构表示 "
        "[[\mathbf{z}_i^{\mathrm{str}}]]。因此，不同实体可以使用不同的传播深度组合；"
        "关闭层选择器时，代码改用最深层状态与输入残差之和。",
    )


def revise_semantic_and_neighbor_sections(document: Document) -> None:
    rewrite(
        document,
        "加权后的三个向量以拼接形式",
        "加权后的三个视图以拼接形式进入输出 MLP，并保留 global 视图残差；"
        "若 global 视图在消融中关闭，则以其余启用视图的均值作为残差。"
        "输出经过 L2 归一化得到语义表示 [[\mathbf{z}_i^{\mathrm{sem}}]]。"
        "该 MLP 学习实体级的非线性视图权重，而不是使用预先设定的固定比例。",
    )
    rewrite(
        document,
        "整图结构编码只计算一次",
        "整图结构编码每次前向计算一次，随后按当前批次实体编号从结构表示矩阵中索引邻居。"
        "固定邻接表使用三元组的头实体到尾实体方向；对每个实体，系统按确定性顺序提取至多 "
        "[[K_{\mathrm{nbr}}]] 个出邻居，主配置取 [[K_{\mathrm{nbr}}=8]]。"
        "没有出邻居时，以实体自身编号填充并将掩码置为 0；邻居不足预算时循环填充已有邻居，但仍保留有效掩码。"
        "该步骤只将变长邻域整理为固定形状的批次张量，不引入可训练参数或新的邻居评分。",
    )


def revise_fusion_section(document: Document) -> None:
    rewrite(
        document,
        "公式（13）—（20）给出",
        "式（13）—（20）定义主模型的邻居上下文与联合表示。主模型以语义表示作为查询，"
        "在联合表示形成之前读取结构邻居；第 5.6 节通过参数匹配控制实验分别改变查询信号和融合位置。",
    )
    rewrite(
        document,
        "本文设置两个参数受控基线",
        "为区分结构邻居是否可见、查询是否依赖语义以及融合发生的位置，本文设置两个参数匹配基线。"
        "参数匹配晚期融合不读取结构邻居，只在结构分支与语义分支独立编码后进行末端融合；"
        "结构查询邻居基线保留相同邻居集合和门控参数，但以实体自身结构表示代替语义表示查询邻居，"
        "语义信息仅在最终联合门控中进入。",
    )
    rewrite(
        document,
        "参数匹配晚期融合先独立得到",
        "参数匹配晚期融合先分别归一化结构表示与语义表示，再使用式（21）的拼接 MLP：",
    )
    rewrite(
        document,
        "无语义查询邻居基线与主模型共享",
        "结构查询邻居基线与主模型共享式（13）—（20）的参数和邻居输入，"
        "仅将式（13）、式（15）和式（17）的查询条件由语义表示替换为实体自身结构表示；"
        "式（19）—（20）仍融合语义表示与结构上下文。",
    )
    remove_paragraph(document, "按有效前向路径统计")


def renumber_experiments_and_tables(document: Document) -> None:
    rewrite(document, "5.4 不同数据集类别的结果差异", "5.3 不同数据集类别的结果差异")
    rewrite(document, "5.5 结构组件与语义视图消融", "5.4 结构组件与语义视图消融")
    rewrite(document, "5.6 结构分支、语义分支与融合模块", "5.5 结构分支、语义分支与融合模块")
    rewrite(document, "5.7 交互位置、邻居可见范围与容量控制", "5.6 融合位置、邻居可见范围与容量控制")

    caption_rewrites = [
        ("表 2 五数据集统计", "表 1 五个数据集的统计信息与实验划分"),
        ("表 3 五个数据集", "表 2 五个数据集的最终测试结果（均值[[\pm]]样本标准差，[[n=3]]）"),
        ("表 4 与经典方法", "表 3 与经典实体对齐方法的定位性比较"),
        ("表 6 不同数据集", "表 4 不同数据集类别的测试结果汇总（数据集等权平均）"),
        ("表 7 结构组件", "表 5 结构组件与语义视图消融（Hits@1/MRR，均值[[\pm]]样本标准差，[[n=3]]）"),
        ("表 8 分支与融合", "表 6 分支与融合必要性实验（均值[[\pm]]样本标准差，[[n=3]]）"),
        ("表 9 参数匹配", "表 7 参数匹配融合控制的设计差异"),
        ("表 10 参数匹配", "表 8 参数匹配融合控制的测试结果（均值[[\pm]]样本标准差，[[n=3]]）"),
    ]
    for old, new in caption_rewrites:
        rewrite(document, old, new)

    rewrite(
        document,
        "实验围绕三个研究问题展开",
        "实验围绕三个研究问题展开。RQ1：完整模型在不同类型的数据集上能否获得稳定结果？"
        "RQ2：关系类型、节点级层选择以及 token、phrase 和 global 视图分别产生何种贡献？"
        "RQ3：在融合参数量相同的条件下，早期读取结构邻居与两个分支独立编码后的晚期融合有何差异，"
        "且邻居查询是否必须使用语义信息？",
    )
    rewrite(
        document,
        "表 3 汇总完整模型",
        "表 2 汇总完整模型在五个数据集上的最终检索结果。每个数据集均使用随机种子 42、43 和 44 独立训练，"
        "并报告 Hits@1、Hits@10 和 MRR 的均值与样本标准差。DBP15K 从训练对中固定留出 10% 作为验证集；"
        "OpenEA 与 EventEA 使用官方验证集。",
    )
    rewrite(
        document,
        "DBP15K FR–EN 的 Hits@1",
        "DBP15K FR–EN 的 Hits@1 和 MRR 最高，分别为 [[0.9038\pm0.0024]] 和 "
        "[[0.9239\pm0.0017]]；EventEA 的 Hits@1 为 [[0.7563\pm0.0029]]；"
        "OpenEA EN–FR 的 Hits@1 最低，为 [[0.5568\pm0.0096]]。",
    )
    rewrite(
        document,
        "表 4 的 MTransE",
        "表 3 中 DBP15K ZH–EN 的 MTransE、JAPE、BootEA 和 RDGCN 数值取自 RDGCN 的汇总[4]，"
        "RREA-text 取自原论文[5]；OpenEA 经典方法数值取自官方基准的五折平均[14]，"
        "而本文 OpenEA 结果为官方第一折上的三随机种子均值。因此，该表用于方法定位，"
        "不构成完全同协议的严格优劣比较。",
    )
    rewrite(
        document,
        "为比较不同任务类型，表 6",
        "为比较不同任务类型，表 4 将五个数据集划分为 DBP15K、OpenEA EN–FR 和 EventEA 三类，"
        "并对 DBP15K 的三个语言对作等权平均。",
    )
    rewrite(
        document,
        "DBP15K 的选择后平均 Hits@1",
        "DBP15K 的选择后平均 Hits@1 为 [[0.7726]]，EventEA 为 [[0.7563]]，"
        "OpenEA EN–FR-15K-V2 为 [[0.5568]]。",
    )
    rewrite(
        document,
        "表 7 同时报告",
        "表 5 报告关系类型、层选择、单视图删除和累积语义组合的消融结果。"
        "每个配置均训练三个随机种子，并独立使用验证集选择检索配置。",
    )
    rewrite(
        document,
        "去除关系类型后",
        "去除关系类型后，DBP15K 与 OpenEA 的 Hits@1 分别下降 3.40 和 2.35 个百分点；"
        "去除层选择器后分别下降 2.15 和 1.90 个百分点。仅 token 视图的 Hits@1 分别为 0.6913 和 0.4394，"
        "token+phrase 为 0.6863 和 0.5550，完整三视图为 0.6809 和 0.5568。"
        "单项删除中，去除 phrase 使 OpenEA Hits@1 下降 4.24 个百分点；"
        "去除 token 或 global 未产生跨两个数据集方向一致的下降。",
    )
    rewrite(
        document,
        "为回答结构分支和融合模块是否必要",
        "表 6 比较完整模型、结构单分支、语义单分支和无可训练融合。单分支的训练目标仅作用于保留的表示，"
        "验证检索分别固定 [[\alpha=0]] 和 [[\alpha=1]]，防止被删除分支通过后处理重新进入。"
        "无可训练融合仍训练结构与语义编码器，但融合阶段不含 MLP、邻居注意力或门控；"
        "它仅使用验证集选择的 [[\alpha]] 计算固定加权和 "
        "[[\operatorname{Norm}(\alpha\mathbf{z}^{\mathrm{sem}}+(1-\alpha)\mathbf{z}^{\mathrm{str}})]]。",
    )
    rewrite(
        document,
        "相对完整模型，仅结构分支",
        "相对完整模型，结构单分支的 Hits@1 在 DBP15K 和 OpenEA 上分别下降 62.00 和 54.38 个百分点；"
        "语义单分支分别下降 4.50 和 1.51 个百分点。无可训练融合在 DBP15K 上的 Hits@1 下降 0.03 个百分点、"
        "MRR 下降 0.89 个百分点；在 OpenEA 上的 Hits@1 和 MRR 分别下降 1.93 和 1.90 个百分点。",
    )
    rewrite(
        document,
        "表 9 固定结构与语义编码器",
        "表 7 固定结构与语义编码器，并使三种融合路径均具有 197,377 个有效可训练参数，"
        "以控制融合模块容量，同时改变结构邻居可见范围、查询信号和融合位置。",
    )
    rewrite(
        document,
        "与完整模型相比，参数匹配晚期融合",
        "与完整模型相比，参数匹配晚期融合在 DBP15K 和 OpenEA 上的 Hits@1 分别下降 1.32 和 7.52 个百分点。"
        "结构查询邻居基线的 Hits@1 分别变化 +0.23 和 +0.02 个百分点，MRR 分别变化 +0.17 和 -0.02 个百分点；"
        "这些变化均小于相应配置的跨种子标准差。",
    )


def revise_discussion_and_conclusion(document: Document) -> None:
    rewrite(document, "6.1 主要发现", "6.1 研究问题回答")
    rewrite(
        document,
        "主实验与消融实验共同区分了四类结论",
        "针对 RQ1，五个数据集的多随机种子结果表明训练波动小于数据集间差异，但稳定性不等同于在所有任务上均有较高绝对性能。"
        "针对 RQ2，关系类型和节点级层选择在两个代表性数据集上得到方向一致的支持；"
        "多尺度语义的作用则具有数据集依赖性，现有结果只清楚支持 phrase 视图对 OpenEA 的贡献，"
        "不能宣称三个视图均不可缺少。针对 RQ3，参数匹配晚期融合弱于可读取结构邻居的模型，"
        "而结构查询与语义查询的差异小于随机种子波动。因此，当前证据支持在融合前保留结构邻居上下文，"
        "但没有证明邻居查询必须由语义表示驱动。",
    )
    rewrite(document, "6.2 OpenEA 低结果与未获支持的假设", "6.2 OpenEA 结果与语义输入限制")
    rewrite(
        document,
        "OpenEA EN–FR-15K-V2 的 Hits@1",
        "OpenEA EN–FR-15K-V2 对实体 URI 进行编码以降低名称偏置[14]，当前输入又以英文 GloVe 覆盖词表内 token，"
        "并以基于 MD5 种子的确定性随机向量表示词表外 token[15]。本地统计显示，该数据集 token 的 GloVe 覆盖率约为 52.18%，"
        "因此大量专名和编码标识缺少可迁移的跨语言语义。与此同时，结构单分支明显弱于语义单分支，"
        "说明有限种子监督下的结构表示尚不能独立建立可靠的跨图坐标系；phrase 视图的增益则表明局部组合能够部分缓解单 token 信息不足。"
        "这些观察与 OpenEA 的较低结果一致，但仍属于机制解释而非因果证明。",
    )
    rewrite(
        document,
        "内部有效性方面",
        "内部有效性方面，全部消融使用相同数据划分、训练目标和三个随机种子，但 [[n=3]] 仍不足以支持高功效显著性检验。"
        "各变体独立选择配置虽避免沿用对主模型有利的超参数，却也引入验证网格方差。"
        "组件消融只覆盖 DBP15K ZH–EN 和 OpenEA EN–FR-15K-V2，且 OpenEA 主结果只运行一个官方划分。"
        "外部有效性方面，尚未验证十万实体以上规模、开放世界以及存在无对应实体的场景。",
    )
    rewrite(
        document,
        "后续研究应优先解决三个问题",
        "后续研究应优先解决三个问题。第一，以跨语言子词模型或可解释的名称编码替代随机词表外向量，"
        "并在 OpenEA 全部官方划分上复现。第二，重新设计邻居查询或加入针对查询阶段的对比约束，"
        "使语义条件产生可测量且非冗余的作用。第三，在更大规模、开放世界和存在无对应实体的场景中，"
        "进一步评估效率、校准稳定性与泛化能力。",
    )
    rewrite(
        document,
        "本文构建了一套结合轻量关系消息",
        "本文提出并实现一种结合轻量关系消息、节点级多深度选择、固定预算结构邻居上下文和多尺度语义编码的实体对齐方法。"
        "多数据集主实验与参数受控消融将方法贡献限定为：关系类型、多深度结构选择和结构邻居可见性获得当前证据支持；"
        "各语义视图的贡献依赖数据集，且语义提前查询邻居尚未显示独立优势。"
        "上述结论适用于本文的数据集、划分与验证协议，仍需通过更广泛的跨语言表示和大规模评估进一步检验。",
    )
    remove_paragraph(document, "两个代表性数据集上的 66 次新增训练")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source, args.output)
    document = Document(args.output)

    revise_front_matter(document)
    revise_related_work(document)
    reorganize_method(document)
    revise_structural_encoder(document)
    revise_semantic_and_neighbor_sections(document)
    revise_fusion_section(document)
    renumber_experiments_and_tables(document)
    revise_discussion_and_conclusion(document)

    document.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
