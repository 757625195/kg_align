from typing import Dict, List

import torch


def build_topology_features(edge_index, edge_type, num_nodes, graph_sizes):
    """Build relation-ID-invariant local topology statistics for both KGs."""
    edge_index = edge_index.detach().cpu()
    edge_type = edge_type.detach().cpu()
    src, dst = edge_index
    ones = torch.ones(src.numel(), dtype=torch.float32)
    in_degree = torch.zeros(num_nodes).index_add_(0, dst, ones)
    out_degree = torch.zeros(num_nodes).index_add_(0, src, ones)
    total_degree = in_degree + out_degree

    incoming_relations = [set() for _ in range(num_nodes)]
    outgoing_relations = [set() for _ in range(num_nodes)]
    for source, target, relation in zip(src.tolist(), dst.tolist(), edge_type.tolist()):
        outgoing_relations[source].add(relation)
        incoming_relations[target].add(relation)
    in_rel_diversity = torch.tensor([len(x) for x in incoming_relations], dtype=torch.float32)
    out_rel_diversity = torch.tensor([len(x) for x in outgoing_relations], dtype=torch.float32)

    log_neighbor_degree = torch.log1p(total_degree)
    out_neighbor_mean = (
        torch.zeros(num_nodes).index_add_(0, src, log_neighbor_degree[dst])
        / out_degree.clamp(min=1.0)
    )
    in_neighbor_mean = (
        torch.zeros(num_nodes).index_add_(0, dst, log_neighbor_degree[src])
        / in_degree.clamp(min=1.0)
    )
    direction_balance = (out_degree - in_degree) / total_degree.clamp(min=1.0)
    features = torch.stack([
        torch.log1p(in_degree), torch.log1p(out_degree), torch.log1p(total_degree),
        torch.log1p(in_rel_diversity), torch.log1p(out_rel_diversity),
        in_neighbor_mean, out_neighbor_mean, direction_balance,
    ], dim=-1)

    normalized = torch.empty_like(features)
    offset = 0
    for graph_size in graph_sizes:
        part = features[offset:offset + graph_size]
        normalized[offset:offset + graph_size] = (
            (part - part.mean(dim=0, keepdim=True))
            / part.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-6)
        )
        offset += graph_size
    if offset != num_nodes:
        raise ValueError("graph_sizes must sum to num_nodes")
    return normalized


def build_adj_list(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> Dict[int, List[int]]:
    """Build a deterministic outgoing-neighbor list for each entity."""
    adj = {i: [] for i in range(num_nodes)}
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for source, target in zip(src, dst):
        adj[source].append(target)
    return adj


def sample_neighbors(
    node_ids: torch.Tensor,
    adj_list: Dict[int, List[int]],
    num_neighbors: int,
    device: torch.device,
):
    """Collect deterministic neighbors with fixed or batch-local width.

    A positive ``num_neighbors`` preserves the fixed-budget protocol. A
    non-positive value retains every outgoing adjacency entry and pads only to
    the largest neighborhood in the current batch.
    """
    batch_ids = node_ids.detach().cpu().tolist()

    if num_neighbors <= 0:
        neighborhoods = [adj_list.get(nid, []) for nid in batch_ids]
        batch_width = max(
            1,
            max((len(neighbors) for neighbors in neighborhoods), default=0),
        )
        neigh_ids = []
        neigh_mask = []
        for nid, neighbors in zip(batch_ids, neighborhoods):
            valid_count = len(neighbors)
            padding = batch_width - valid_count
            neigh_ids.append([*neighbors, *([nid] * padding)])
            neigh_mask.append([*([1] * valid_count), *([0] * padding)])

        return (
            torch.tensor(neigh_ids, dtype=torch.long, device=device),
            torch.tensor(neigh_mask, dtype=torch.long, device=device),
        )

    neigh_ids = []
    neigh_mask = []

    for nid in batch_ids:
        neighbors = adj_list.get(nid, [])

        if len(neighbors) == 0:
            ids = [nid] * num_neighbors
            mask = [0] * num_neighbors
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
