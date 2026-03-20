# graph_utils.py
from typing import Dict, List
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
    为每个节点稳定选择 K 个邻居。

    这里不再使用随机采样，而是按邻接表中的固定顺序截取。
    这样做的目的在于:
    - 控制每个 batch 的结构上下文规模，避免邻居爆炸
    - 为跨模态融合提供可复现的局部结构证据
    - 减少随机邻居扰动对融合模块训练稳定性的影响
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
                picked = neighbors[:num_neighbors]
            else:
                picked = neighbors[:]
                while len(picked) < num_neighbors:
                    picked.append(neighbors[len(picked) % len(neighbors)])
            ids = picked
            mask = [1] * num_neighbors

        neigh_ids.append(ids)
        neigh_mask.append(mask)

    neigh_ids = torch.tensor(neigh_ids, dtype=torch.long, device=device)
    neigh_mask = torch.tensor(neigh_mask, dtype=torch.long, device=device)
    return neigh_ids, neigh_mask
