from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F

from graph_utils import sample_neighbors


@torch.no_grad()
def encode_entity_outputs(
    model,
    node_ids: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    seq_features: torch.Tensor,
    adj_list: Dict[int, list],
    num_neighbors: int,
    batch_size: int,
    device: torch.device,
):
    model.eval()
    outputs = {}

    for start in range(0, len(node_ids), batch_size):
        batch_ids = node_ids[start:start + batch_size].to(device)
        batch_seq = seq_features[batch_ids.cpu()].to(device)

        neighbor_ids, neighbor_mask = sample_neighbors(
            node_ids=batch_ids,
            adj_list=adj_list,
            num_neighbors=num_neighbors,
            device=device,
        )

        out = model(
            node_ids=batch_ids,
            edge_index=edge_index,
            seq_features=batch_seq,
            neighbor_ids=neighbor_ids,
            neighbor_mask=neighbor_mask,
            edge_type=edge_type,
        )
        for key, value in out.items():
            if not torch.is_tensor(value):
                continue
            outputs.setdefault(key, []).append(value.detach().cpu())

    return {
        key: torch.cat(chunks, dim=0)
        for key, chunks in outputs.items()
    }


@torch.no_grad()
def encode_entities(
    model,
    node_ids: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    seq_features: torch.Tensor,
    adj_list: Dict[int, list],
    num_neighbors: int,
    batch_size: int,
    device: torch.device,
    output_key: str = "z_joint",
):
    outputs = encode_entity_outputs(
        model=model,
        node_ids=node_ids,
        edge_index=edge_index,
        edge_type=edge_type,
        seq_features=seq_features,
        adj_list=adj_list,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        device=device,
    )
    return outputs[output_key]


def fuse_entity_outputs(
    outputs: Dict[str, torch.Tensor],
    weights: Tuple[float, float, float],
) -> torch.Tensor:
    w_joint, w_struct, w_sem = weights
    fused = (
        w_joint * outputs["z_joint"] +
        w_struct * outputs["z_struct_enhanced"] +
        w_sem * outputs["z_sem_enhanced"]
    )
    return F.normalize(fused, p=2, dim=-1)


@torch.no_grad()
def compute_metrics_from_similarity(
    sim: torch.Tensor,
    gt_index: torch.Tensor,
    ks=(1, 10)
):
    gt_score = sim[torch.arange(sim.size(0)), gt_index].unsqueeze(1)
    ranks = (sim > gt_score).sum(dim=1) + 1

    metrics = {}
    for k in ks:
        metrics[f"Hits@{k}"] = (ranks <= k).float().mean().item()
    metrics["MRR"] = (1.0 / ranks.float()).mean().item()
    return metrics


@torch.no_grad()
def evaluate_alignment(
    model,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    seq_features: torch.Tensor,
    adj_list,
    num_neighbors: int,
    test_pairs: List[Tuple[int, int]],
    candidate_right_ids: torch.Tensor,
    batch_size: int,
    device: torch.device,
    fusion_weights: Tuple[float, float, float] = None,
):
    left_ids = torch.tensor([l for l, _ in test_pairs], dtype=torch.long)
    if candidate_right_ids.ndim != 1:
        raise ValueError("candidate_right_ids must be a 1D tensor")

    right_ids = candidate_right_ids.long().cpu()
    right_index = {int(r.item()): idx for idx, r in enumerate(right_ids)}
    gt = torch.tensor(
        [right_index[r] for _, r in test_pairs],
        dtype=torch.long,
    )

    if fusion_weights is None:
        left_emb = encode_entities(
            model=model,
            node_ids=left_ids,
            edge_index=edge_index,
            edge_type=edge_type,
            seq_features=seq_features,
            adj_list=adj_list,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            device=device,
        )

        right_emb = encode_entities(
            model=model,
            node_ids=right_ids,
            edge_index=edge_index,
            edge_type=edge_type,
            seq_features=seq_features,
            adj_list=adj_list,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            device=device,
        )
    else:
        left_outputs = encode_entity_outputs(
            model=model,
            node_ids=left_ids,
            edge_index=edge_index,
            edge_type=edge_type,
            seq_features=seq_features,
            adj_list=adj_list,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            device=device,
        )
        right_outputs = encode_entity_outputs(
            model=model,
            node_ids=right_ids,
            edge_index=edge_index,
            edge_type=edge_type,
            seq_features=seq_features,
            adj_list=adj_list,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            device=device,
        )
        left_emb = fuse_entity_outputs(left_outputs, fusion_weights)
        right_emb = fuse_entity_outputs(right_outputs, fusion_weights)

    left_emb = F.normalize(left_emb, p=2, dim=-1)
    right_emb = F.normalize(right_emb, p=2, dim=-1)

    sim = model.score_pairs(left_emb, right_emb).cpu()

    return compute_metrics_from_similarity(sim, gt)
