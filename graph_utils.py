from typing import Dict, List, Tuple, Union
import math

import torch


NeighborEntry = Union[int, Tuple[int, int, float, int]]


def build_adj_list(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_type: torch.Tensor = None,
    use_relation_aware: bool = False,
    relation_score_alpha: float = 1.0,
    neighbor_degree_alpha: float = 0.25,
) -> Dict[int, List[NeighborEntry]]:
    if edge_type is None or not use_relation_aware:
        adj = {i: [] for i in range(num_nodes)}
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for s, d in zip(src, dst):
            adj[s].append(d)
        return adj

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    rel = edge_type.tolist()

    relation_counts = torch.bincount(edge_type.cpu(), minlength=int(edge_type.max().item()) + 1).float()
    out_degree = torch.bincount(edge_index[0].cpu(), minlength=num_nodes).float()
    in_degree = torch.bincount(edge_index[1].cpu(), minlength=num_nodes).float()
    total_degree = out_degree + in_degree

    adj_maps: List[Dict[int, Tuple[int, float, int]]] = [dict() for _ in range(num_nodes)]
    for order, (s, d, r) in enumerate(zip(src, dst, rel)):
        rel_score = relation_score_alpha / math.sqrt(float(relation_counts[r].item()) + 1.0)
        degree_score = neighbor_degree_alpha / math.sqrt(float(total_degree[d].item()) + 1.0)
        score = rel_score + degree_score

        existing = adj_maps[s].get(d)
        if existing is None or score > existing[1] or (score == existing[1] and order < existing[2]):
            adj_maps[s][d] = (r, score, order)

    adj: Dict[int, List[NeighborEntry]] = {}
    for node_id in range(num_nodes):
        entries = [
            (neighbor_id, relation_id, score, order)
            for neighbor_id, (relation_id, score, order) in adj_maps[node_id].items()
        ]
        entries.sort(key=lambda item: (-item[2], item[3], item[0]))
        adj[node_id] = entries
    return adj


def _pick_relation_aware_neighbors(
    neighbors: List[Tuple[int, int, float, int]],
    num_neighbors: int,
) -> List[int]:
    if len(neighbors) <= num_neighbors:
        return [neighbor_id for neighbor_id, _, _, _ in neighbors]

    relation_buckets: Dict[int, List[Tuple[int, float, int]]] = {}
    for neighbor_id, relation_id, score, order in neighbors:
        relation_buckets.setdefault(relation_id, []).append((neighbor_id, score, order))

    relation_order = sorted(
        relation_buckets.keys(),
        key=lambda relation_id: (
            -relation_buckets[relation_id][0][1],
            relation_buckets[relation_id][0][2],
            relation_id,
        ),
    )

    bucket_pos = {relation_id: 0 for relation_id in relation_order}
    picked: List[int] = []
    while len(picked) < num_neighbors:
        progressed = False
        for relation_id in relation_order:
            pos = bucket_pos[relation_id]
            bucket = relation_buckets[relation_id]
            if pos >= len(bucket):
                continue
            picked.append(bucket[pos][0])
            bucket_pos[relation_id] += 1
            progressed = True
            if len(picked) >= num_neighbors:
                break
        if not progressed:
            break
    return picked


def sample_neighbors(
    node_ids: torch.Tensor,
    adj_list: Dict[int, List[NeighborEntry]],
    num_neighbors: int,
    device: torch.device,
):
    """
    为每个节点稳定选择 K 个邻居。

    当邻接表包含 relation-aware 条目时，先按关系稀有度和邻居非 hub 程度
    预排序，再用 relation-diverse round-robin 方式选取，避免同一种关系占满
    全部邻居预算；否则退化为固定顺序截取。
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
            first_entry = neighbors[0]
            if isinstance(first_entry, tuple):
                picked = _pick_relation_aware_neighbors(neighbors, num_neighbors)
            else:
                picked = neighbors[:num_neighbors]

            if len(picked) >= num_neighbors:
                ids = picked[:num_neighbors]
            else:
                ids = picked[:]
                while len(ids) < num_neighbors:
                    ids.append(picked[len(ids) % len(picked)])
            mask = [1] * num_neighbors

        neigh_ids.append(ids)
        neigh_mask.append(mask)

    neigh_ids = torch.tensor(neigh_ids, dtype=torch.long, device=device)
    neigh_mask = torch.tensor(neigh_mask, dtype=torch.long, device=device)
    return neigh_ids, neigh_mask
