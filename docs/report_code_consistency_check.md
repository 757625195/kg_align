# 报告与代码一致性核查

核查对象：

- 报告原文提取自 [tmp_report_extracted.txt](../tmp_report_extracted.txt)
- 代码仓库 `..`

## 结论概览

整体上，报告第 3.6、3.7、3.8 节对语义编码器、跨模态增强、两阶段训练和主动学习主流程的描述，与当前代码主干基本一致。  
真正明显不一致的地方主要集中在三类：

1. 报告写成“已实现/已用于训练”，但代码里只是“有开关或有损失定义，默认并未启用”。
2. 报告把研究计划或理想实验范围写成了当前系统能力，但仓库实际上只支持更窄的范围。
3. 报告某些表述是“人工标注/候选对级别/完整实验结果”，而代码实现的是“模拟标注/每个左实体选一个最佳右实体/仓库内无法直接验证结果数值”。

## 明确不一致项

### 1. 报告写了显式 2-hop/3-hop topology matching 已参与训练，但默认配置下它是关闭的

报告第 3.7.3、3.8.2 节把 `Ltopo` 作为当前训练目标的一部分来介绍。  
代码里确实计算了 topology loss，但默认系数 `lambda_topology` 为 `0.00`，因此默认训练不会真正优化这个项。

证据：

- [train.py](../train.py#L75) 默认 `lambda_topology: float = 0.00`
- [models/losses.py](../models/losses.py#L263) 计算了 `topology_loss`
- [models/losses.py](../models/losses.py#L279) 只有乘上 `lambda_topology` 后才会进入总损失

更准确的表述应是：代码已经实现了显式 2-hop/3-hop 拓扑匹配项，但当前默认实验配置并未启用。

### 2. 报告把实验数据集写成 DBP15K、WK3l、DWY100K，但当前代码只实现了 DBP15K 加载

报告第 1.2、5、8 节反复写到会在 DBP15K、WK3l、DWY100K 上做实验，容易让人理解成当前仓库已经支持这些数据。  
但当前数据加载入口只调用 `torch_geometric.datasets.DBP15K`，没有 WK3l 或 DWY100K 的数据读取与训练分支。

证据：

- [data_utils.py](../data_utils.py#L7) 只有 `load_dbp15k_from_pyg`
- [train.py](../train.py#L24) 默认根目录就是 `data/DBP15K`
- [train.py](../train.py#L25) 当前默认 pair 为 `zh_en`

更准确的表述应是：当前可运行实现只覆盖 DBP15K；WK3l 和 DWY100K 仍属于计划中的扩展实验数据集。

### 3. 报告前文写“人工标注实时反馈”，代码实际是 oracle-style 模拟标注

报告第 1.4 节写的是“模型主动选择样本供人工标注，并实时反馈到训练中”。  
但当前代码并没有真实的人在环标注接口，而是直接用测试集真值来模拟人工反馈。

证据：

- [active_learning.py](../active_learning.py#L206) `simulate_human_annotation`
- [active_learning.py](../active_learning.py#L233) 使用 `gold_test_pairs = set(test_pairs)`
- [active_learning.py](../active_learning.py#L212) 通过真值判断候选是否为正样本

更准确的表述应是：当前实现为“模拟主动学习”，不是“真实人工标注闭环”。

### 4. 报告把主动学习说成对“候选对”统一评分，代码实际上是“每个左实体先取一个最佳右实体，再评分”

报告第 3.4、3.8.3 节的数学写法更像是对任意 candidate pair 逐对打分。  
但当前代码并不是枚举所有 `(left, right)` 候选对来算 `Score(x)`，而是先对每个左实体取 `argmax` 得到一个最佳右实体，再把该配对作为候选。

证据：

- [active_learning.py](../active_learning.py#L101) 先计算完整相似度矩阵
- [active_learning.py](../active_learning.py#L103) `pred_j = sim.argmax(dim=-1)`
- [active_learning.py](../active_learning.py#L127) 每个左实体只拿 `matched_j` 形成一个候选对

所以报告中的公式更像方法说明，而不是代码的精确实现细节。

### 5. 报告把 representativeness 写成“pair-level 的联合估计”，代码只对左侧实体表示打分

报告第 3.4、3.8.3 节给人的直观印象是：代表性是对候选实体对联合计算的。  
当前实现中，`multi_view_representativeness` 只使用 `left_outputs` 的 joint/struct/sem 三个表示，不直接把右侧表示并入 representativeness。

证据：

- [active_learning.py](../active_learning.py#L43) `multi_view_representativeness`
- [active_learning.py](../active_learning.py#L113) 只传入 `left_outputs["z_joint"]`
- [active_learning.py](../active_learning.py#L115) 只传入左侧结构增强表示
- [active_learning.py](../active_learning.py#L116) 只传入左侧语义增强表示

更准确的写法应是：当前实现对“左侧待选实体”的多视图中心性进行估计，再与其最佳匹配右实体组成候选对。

### 6. 报告把“复杂度和可扩展性”写得较完整，但仓库没有对应的 benchmark 或大规模实验脚本

报告第 3.7.4 节已经在正文中承认“仓库尚不包含专门复杂度 benchmark 脚本”，这一点和代码一致。  
但如果读者只看前半段，会以为已经完成了系统复杂度验证。当前仓库里确实没有单独 benchmark、profiling 或工业级部署脚本。

证据：

- 仓库文件列表中没有 benchmark/profiling/distributed training 脚本
- [README.md](../README.md) 也明确把可运行范围限定在当前轻量实现

这一项更适合归类为“表述偏满”，不算代码错误。

### 7. 报告第 11 节的结果数值在文内自相矛盾，而且仓库里没有日志能直接核实

同一节中先写“best Hits@1 = 0.6381”，后面的表格又写“Full Model Hits@1 = 0.6392”。  
这不是“报告与代码不一致”，而是“报告内部不一致”。同时，仓库当前没有保存对应训练日志或结果表，无法仅凭代码验证哪一个数值正确。

证据：

- 报告第 11 节正文：`0.6381`
- 报告第 11 节表 5：`0.6392`
- 仓库只有模型权重文件，没有配套结果日志文件可供交叉核验

建议统一成一个数值，并说明该数值来自哪次运行。

## 基本一致项

### 1. 多尺度 Transformer 语义模块与代码一致

报告第 3.6 节关于以下几点，与代码是一致的：

- 输入先线性投影
- 用非零行构造 valid-token mask
- token-level 用 masked mean pooling
- phrase-level 用 kernel=3 和 kernel=5 的 1D 卷积后平均
- global-level 用 TransformerEncoder
- 三路表示先门控加权再拼接
- 最后经 MLP 投影并做 L2 归一化

证据：

- [models/text_encoder.py](../models/text_encoder.py#L33)
- [models/text_encoder.py](../models/text_encoder.py#L63)
- [models/text_encoder.py](../models/text_encoder.py#L78)
- [models/text_encoder.py](../models/text_encoder.py#L80)
- [models/text_encoder.py](../models/text_encoder.py#L85)
- [models/text_encoder.py](../models/text_encoder.py#L89)

### 2. 轻量级 GNN 的“可选参数共享 / 可选 depthwise separable / 残差归一化”与代码一致

证据：

- [models/full_model.py](../models/full_model.py#L27)
- [models/gnn_encoder.py](../models/gnn_encoder.py#L60)
- [models/gnn_encoder.py](../models/gnn_encoder.py#L80)
- [models/gnn_encoder.py](../models/gnn_encoder.py#L98)
- [models/gnn_encoder.py](../models/gnn_encoder.py#L130)

### 3. 固定预算邻居采样 + 全图结构编码 + 局部邻居增强，这个主设计与代码一致

证据：

- [train.py](../train.py#L55) `num_neighbors`
- [train.py](../train.py#L223) 训练时固定邻居采样
- [models/full_model.py](../models/full_model.py#L77) 全图结构编码
- [models/full_model.py](../models/full_model.py#L103) 再按 batch 提取节点和邻居表示

### 4. 跨模态增强模块与代码一致

报告里写到：

- 结构邻居通过 attention 增强语义
- 语义通过 gating 影响结构
- 增强后拼接，再经 MLP、LayerNorm、L2 normalize 得到 joint embedding

证据：

- [models/fusion.py](../models/fusion.py#L25)
- [models/fusion.py](../models/fusion.py#L56)
- [models/fusion.py](../models/fusion.py#L73)
- [models/fusion.py](../models/fusion.py#L90)

### 5. 两阶段训练、AdamW、early stopping、渐进式启用协同损失，与代码一致

证据：

- [train.py](../train.py#L34) `warmup_epochs = 5`
- [train.py](../train.py#L35) `joint_epochs = 15`
- [train.py](../train.py#L573) `AdamW`
- [train.py](../train.py#L118) `early_stop_patience = 4`
- [models/losses.py](../models/losses.py#L217) 渐进式 ramp-up
- [train.py](../train.py#L699) best model 保存
- [train.py](../train.py#L713) early stopping 触发

### 6. 主动学习中的 uncertainty、margin、entropy、diversity rerank 与代码一致

证据：

- [models/alignment_head.py](../models/alignment_head.py#L29) 置信度统计
- [models/alignment_head.py](../models/alignment_head.py#L40) `0.5 * entropy + 0.5 * (1.0 - margin)`
- [active_learning.py](../active_learning.py#L62) diversity rerank
- [active_learning.py](../active_learning.py#L161) 置信度、margin、不确定性过滤

## 无法直接从当前环境验证的项

以下内容不是“发现代码不符”，而是“当前工作区无法直接核实”：

- 报告第 10 节给出的运行环境版本，如 macOS 26.3.1、PyTorch 2.8.0、PyG 2.6.1
- 报告第 10 节中 `seq_features` 形状 `(38960, 14, 300)` 的运行时数值
- 报告第 11 节的 Hits@1、Hits@10、MRR 数值

原因：

- 当前终端默认 `python3` 环境里没有 `torch`，因此我无法直接加载数据并复现实验配置
- 仓库内没有保存对应训练日志或结果表

## 建议你在报告中修改的表述

如果你要把报告改得和代码严格一致，建议优先改这几句：

1. 把“主动学习中的人工标注实时反馈”改成“当前实现采用 oracle-style 模拟标注，后续可扩展到真人在环”。
2. 把“当前实验使用 DBP15K、WK3l、DWY100K”改成“当前可运行代码已支持 DBP15K，WK3l 与 DWY100K 计划后续扩展”。
3. 把“显式 2-hop/3-hop topology matching objective 已纳入当前训练”改成“代码已实现该损失，但默认配置下未启用”。
4. 把主动学习的 pair-level 数学描述收紧为“当前实现对每个左实体先选一个最优右实体，再对该候选进行排序、过滤和多样性重排”。
5. 把实验结果部分的 `0.6381` 和 `0.6392` 统一成一个确定值，并注明来源运行。
