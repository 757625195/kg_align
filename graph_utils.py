# graph_utils.py
from typing import Dict, List
import random
import torch


def build_adj_list(edge_index: torch.Tensor, num_nodes: int) -> Dict[int, List[int]]:
    adj = {i: [] for i in range(num_nodes)}
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()

    for s, d in zip(src, dst):
        adj[s].append(d)

    return adj


def sample_neighbors(
    node_ids: torch.Tensor,
    adj_list: Dict[int, List[int]],
    num_neighbors: int,
    device: torch.device,
):
    """
    为每个节点固定采样 K 个邻居。

    这里的设计目的不是显式展开完整多跳子图，而是:
    - 控制每个 batch 的结构上下文规模，避免邻居爆炸
    - 为后续跨模态融合提供局部结构证据
    - 在大图上保持近似线性的邻居访问成本
    """
    batch_ids = node_ids.detach().cpu().tolist()

    neigh_ids = []
    neigh_mask = []

    for nid in batch_ids:
        neighbors = adj_list.get(nid, [])

        if len(neighbors) == 0:
            ids = [nid] * num_neighbors
            mask = [0] * num_neighbors
        else:
            if len(neighbors) >= num_neighbors:
                picked = random.sample(neighbors, num_neighbors)
            else:
                picked = neighbors[:]
                while len(picked) < num_neighbors:
                    picked.append(random.choice(neighbors))
            ids = picked
            mask = [1] * num_neighbors

        neigh_ids.append(ids)
        neigh_mask.append(mask)

    neigh_ids = torch.tensor(neigh_ids, dtype=torch.long, device=device)
    neigh_mask = torch.tensor(neigh_mask, dtype=torch.long, device=device)
    return neigh_ids, neigh_mask
