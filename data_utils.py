from typing import Dict

import torch
from torch_geometric.datasets import DBP15K


def load_dbp15k_from_pyg(root: str = "data/DBP15K", pair: str = "zh_en") -> Dict:
    dataset = DBP15K(root=root, pair=pair)
    data = dataset[0]

    x1 = data.x1.float()                         # [N1, L1, D]
    x2 = data.x2.float()                         # [N2, L2, D]
    edge_index1 = data.edge_index1.long()       # [2, E1]
    edge_index2 = data.edge_index2.long()       # [2, E2]
    train_y = data.train_y.long()               # [2, T]
    test_y = data.test_y.long()                 # [2, U]

    n1 = x1.size(0)
    n2 = x2.size(0)
    total_nodes = n1 + n2

    # 统一图 id：KG2 整体偏移 n1
    edge_index2_global = edge_index2 + n1
    edge_index = torch.cat([edge_index1, edge_index2_global], dim=1)

    # 统一语义特征表 [N1+N2, L, D]
    # PyG 中两个图的序列长度通常一致；这里稳妥起见按最大长度 pad
    l1, d1 = x1.size(1), x1.size(2)
    l2, d2 = x2.size(1), x2.size(2)
    if d1 != d2:
        raise ValueError(f"x1 dim={d1} and x2 dim={d2} mismatch")

    max_len = max(l1, l2)

    if l1 < max_len:
        pad = torch.zeros(n1, max_len - l1, d1, dtype=x1.dtype)
        x1 = torch.cat([x1, pad], dim=1)

    if l2 < max_len:
        pad = torch.zeros(n2, max_len - l2, d2, dtype=x2.dtype)
        x2 = torch.cat([x2, pad], dim=1)

    seq_features = torch.cat([x1, x2], dim=0)   # [N, L, D]

    train_pairs = [
        (int(train_y[0, i].item()), int(train_y[1, i].item() + n1))
        for i in range(train_y.size(1))
    ]
    test_pairs = [
        (int(test_y[0, i].item()), int(test_y[1, i].item() + n1))
        for i in range(test_y.size(1))
    ]

    # 为了兼容旧代码接口，提供全局 id -> 语义序列
    seq_feature_map = {i: seq_features[i] for i in range(total_nodes)}

    return {
        "total_nodes": total_nodes,
        "n1": n1,
        "n2": n2,
        "edge_index": edge_index,
        "seq_features": seq_features,
        "seq_feature_map": seq_feature_map,
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
        "raw_data": data,
    }