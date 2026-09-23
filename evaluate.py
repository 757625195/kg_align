from typing import List, Dict, Tuple, Optional, Sequence, Any

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
    z_struct_all: torch.Tensor = None,
):
    model.eval()
    outputs = {}
    if z_struct_all is None:
        z_struct_all = model.encode_structure_all(
            edge_index=edge_index,
            edge_type=edge_type,
        )

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
            z_struct_all=z_struct_all,
        )
        for key, value in out.items():
            if not torch.is_tensor(value):
                continue
            if key in {"semantic_attention", "structural_gate"}:
                # Neighbor width is batch-dependent in variable-neighborhood
                # mode, so these diagnostics cannot be concatenated directly.
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
    z_struct_all: torch.Tensor = None,
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
        z_struct_all=z_struct_all,
    )
    return outputs[output_key]


@torch.no_grad()
def compute_ranking_metrics_from_similarity(
    sim: torch.Tensor,
    gt_index: torch.Tensor,
    ks=(1, 5, 10),
):
    gt_score = sim[torch.arange(sim.size(0)), gt_index].unsqueeze(1)
    ranks = (sim > gt_score).sum(dim=1) + 1

    metrics = {}
    for k in ks:
        metrics[f"Hits@{k}"] = (ranks <= k).float().mean().item()
    metrics["MRR"] = (1.0 / ranks.float()).mean().item()
    return metrics


@torch.no_grad()
def _normalize_embeddings(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


@torch.no_grad()
def compute_classification_metrics(
    top1_index: torch.Tensor,
    top1_score: torch.Tensor,
    gt_index: torch.Tensor,
    acceptance_threshold: float,
):
    accepted = top1_score >= acceptance_threshold
    correct = top1_index.eq(gt_index)
    true_positive = (accepted & correct).sum().item()
    predicted_positive = accepted.sum().item()
    total_positive = gt_index.numel()

    precision = true_positive / predicted_positive if predicted_positive > 0 else 0.0
    recall = true_positive / total_positive if total_positive > 0 else 0.0
    if precision + recall > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accepted": predicted_positive,
        "AcceptanceRate": predicted_positive / total_positive if total_positive > 0 else 0.0,
        "Threshold": float(acceptance_threshold),
    }


@torch.no_grad()
def find_best_acceptance_threshold(
    top1_index: torch.Tensor,
    top1_score: torch.Tensor,
    gt_index: torch.Tensor,
) -> Dict[str, float]:
    if top1_score.numel() == 0:
        return {
            "threshold": 0.0,
            "Precision": 0.0,
            "Recall": 0.0,
            "F1": 0.0,
        }

    order = torch.argsort(top1_score, descending=True)
    sorted_scores = top1_score[order]
    sorted_correct = top1_index[order].eq(gt_index[order]).to(dtype=torch.float32)

    cum_tp = torch.cumsum(sorted_correct, dim=0)
    pred_pos = torch.arange(1, sorted_scores.numel() + 1, dtype=torch.float32)
    total_positive = float(gt_index.numel())

    best = {
        "threshold": float(sorted_scores.max().item()),
        "Precision": 0.0,
        "Recall": 0.0,
        "F1": 0.0,
    }

    for idx in range(sorted_scores.numel()):
        is_last = idx == sorted_scores.numel() - 1
        if not is_last and sorted_scores[idx].item() == sorted_scores[idx + 1].item():
            continue

        tp = float(cum_tp[idx].item())
        precision = tp / float(pred_pos[idx].item())
        recall = tp / total_positive if total_positive > 0 else 0.0
        f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        threshold = float(sorted_scores[idx].item())

        if (
            f1 > best["F1"] + 1e-12
            or (
                abs(f1 - best["F1"]) <= 1e-12
                and (
                    precision > best["Precision"] + 1e-12
                    or (
                        abs(precision - best["Precision"]) <= 1e-12
                        and (
                            recall > best["Recall"] + 1e-12
                            or (
                                abs(recall - best["Recall"]) <= 1e-12
                                and threshold > best["threshold"]
                            )
                        )
                    )
                )
            )
        ):
            best = {
                "threshold": threshold,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            }

    return best


@torch.no_grad()
def _compute_csls_corrections(
    left_emb: torch.Tensor,
    right_emb: torch.Tensor,
    csls_k: int,
    batch_size: int,
):
    if csls_k <= 0:
        return None, None

    row_correction = []
    for start in range(0, left_emb.size(0), batch_size):
        left_chunk = left_emb[start:start + batch_size]
        sim_chunk = torch.matmul(left_chunk, right_emb.t())
        k = max(1, min(csls_k, sim_chunk.size(1)))
        row_correction.append(sim_chunk.topk(k=k, dim=1).values.mean(dim=1))
    row_correction = torch.cat(row_correction, dim=0)

    col_correction = []
    for start in range(0, right_emb.size(0), batch_size):
        right_chunk = right_emb[start:start + batch_size]
        sim_chunk = torch.matmul(left_emb, right_chunk.t())
        k = max(1, min(csls_k, sim_chunk.size(0)))
        col_correction.append(sim_chunk.topk(k=k, dim=0).values.mean(dim=0))
    col_correction = torch.cat(col_correction, dim=0)

    return row_correction, col_correction


@torch.no_grad()
def _apply_csls_to_similarity(
    sim_chunk: torch.Tensor,
    row_chunk: Optional[torch.Tensor],
    col_correction: Optional[torch.Tensor],
    csls_blend: float,
):
    if row_chunk is None or col_correction is None:
        return sim_chunk

    csls_chunk = 2.0 * sim_chunk - row_chunk - col_correction.unsqueeze(0)
    blend = min(max(float(csls_blend), 0.0), 1.0)
    return (1.0 - blend) * sim_chunk + blend * csls_chunk


@torch.no_grad()
def _compute_similarity_statistics(
    left_emb: torch.Tensor,
    right_emb: torch.Tensor,
    gt_index: torch.Tensor,
    batch_size: int,
    csls_k: int = 0,
    csls_blend: float = 1.0,
):
    row_correction, col_correction = _compute_csls_corrections(
        left_emb=left_emb,
        right_emb=right_emb,
        csls_k=csls_k,
        batch_size=batch_size,
    )

    ranks = []
    top1_index = []
    top1_score = []
    ks = (1, 5, 10)
    hits = {k: 0 for k in ks}

    for start in range(0, left_emb.size(0), batch_size):
        end = start + batch_size
        left_chunk = left_emb[start:end]
        sim_chunk = torch.matmul(left_chunk, right_emb.t())
        row_chunk = None if row_correction is None else row_correction[start:end].unsqueeze(1)
        sim_chunk = _apply_csls_to_similarity(
            sim_chunk=sim_chunk,
            row_chunk=row_chunk,
            col_correction=col_correction,
            csls_blend=csls_blend,
        )
        gt_chunk = gt_index[start:end]
        gt_score = sim_chunk[torch.arange(sim_chunk.size(0)), gt_chunk].unsqueeze(1)
        rank_chunk = (sim_chunk > gt_score).sum(dim=1) + 1

        top_score_chunk, top_index_chunk = sim_chunk.max(dim=1)

        ranks.append(rank_chunk.cpu())
        top1_index.append(top_index_chunk.cpu())
        top1_score.append(top_score_chunk.cpu())

        for k in ks:
            hits[k] += int((rank_chunk <= k).sum().item())

    ranks = torch.cat(ranks, dim=0)
    top1_index = torch.cat(top1_index, dim=0)
    top1_score = torch.cat(top1_score, dim=0)

    total = max(1, gt_index.numel())
    ranking_metrics = {
        f"Hits@{k}": hits[k] / total
        for k in ks
    }
    ranking_metrics["MRR"] = (1.0 / ranks.float()).mean().item()

    return ranking_metrics, top1_index, top1_score


@torch.no_grad()
def select_csls_parameters(
    model,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    seq_features: torch.Tensor,
    adj_list,
    num_neighbors: int,
    validation_pairs: List[Tuple[int, int]],
    candidate_right_ids: torch.Tensor,
    batch_size: int,
    device: torch.device,
    csls_k_candidates: Sequence[int],
    csls_blend_candidates: Sequence[float],
    selection_metric: str = "MRR",
    representation_key: str = "z_joint",
) -> Dict[str, Any]:
    """Select CSLS hyperparameters on validation pairs without re-encoding."""
    if not validation_pairs:
        raise ValueError("validation_pairs must not be empty for CSLS selection")

    k_candidates = sorted({int(k) for k in csls_k_candidates if int(k) > 0})
    blend_candidates = sorted({float(blend) for blend in csls_blend_candidates})
    if not k_candidates:
        raise ValueError("csls_k_candidates must contain at least one positive integer")
    if not blend_candidates:
        raise ValueError("csls_blend_candidates must not be empty")
    if any(blend < 0.0 or blend > 1.0 for blend in blend_candidates):
        raise ValueError("CSLS blend candidates must be within [0, 1]")

    left_ids = torch.tensor([left for left, _ in validation_pairs], dtype=torch.long)
    right_ids = candidate_right_ids.long().cpu()
    right_index = {int(right.item()): idx for idx, right in enumerate(right_ids)}
    gt_index = torch.tensor(
        [right_index[right] for _, right in validation_pairs],
        dtype=torch.long,
    )

    z_struct_all = model.encode_structure_all(
        edge_index=edge_index,
        edge_type=edge_type,
    )
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
        z_struct_all=z_struct_all,
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
        z_struct_all=z_struct_all,
    )

    left_emb = _normalize_embeddings(left_outputs[representation_key]).to(device)
    right_emb = _normalize_embeddings(right_outputs[representation_key]).to(device)

    normalized_metric = selection_metric.strip()
    grid = []
    for csls_k in k_candidates:
        for csls_blend in blend_candidates:
            metrics, _, _ = _compute_similarity_statistics(
                left_emb=left_emb,
                right_emb=right_emb,
                gt_index=gt_index,
                batch_size=batch_size,
                csls_k=csls_k,
                csls_blend=csls_blend,
            )
            if normalized_metric not in metrics:
                raise ValueError(
                    f"Unsupported CSLS selection metric {selection_metric!r}; "
                    f"available metrics are {sorted(metrics)}"
                )
            grid.append({
                "k": csls_k,
                "blend": csls_blend,
                **metrics,
            })

    best = max(
        grid,
        key=lambda row: (
            row[normalized_metric],
            row["Hits@1"],
            row["Hits@10"],
            -row["k"],
            -row["blend"],
        ),
    )
    return {
        "selection_metric": normalized_metric,
        "best_k": int(best["k"]),
        "best_blend": float(best["blend"]),
        "best_metrics": {
            key: float(value)
            for key, value in best.items()
            if key not in {"k", "blend"}
        },
        "grid": grid,
    }


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
    acceptance_threshold: Optional[float] = None,
    calibrate_threshold: bool = False,
    representation_key: str = "z_joint",
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
    z_struct_all = model.encode_structure_all(
        edge_index=edge_index,
        edge_type=edge_type,
    )

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
        z_struct_all=z_struct_all,
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
        z_struct_all=z_struct_all,
    )

    left_emb = _normalize_embeddings(left_outputs[representation_key])
    right_emb = _normalize_embeddings(right_outputs[representation_key])
    left_eval_emb = left_emb.to(device)
    right_eval_emb = right_emb.to(device)

    ranking_metrics, top1_index, top1_score = _compute_similarity_statistics(
        left_emb=left_eval_emb,
        right_emb=right_eval_emb,
        gt_index=gt,
        batch_size=batch_size,
        csls_k=0,
        csls_blend=0.0,
    )

    csls_k = getattr(model.align_head, "csls_k", 0)
    csls_blend = getattr(model.align_head, "csls_blend", 1.0)
    csls_metrics = {}
    if csls_k > 0:
        csls_ranking_metrics, _, _ = _compute_similarity_statistics(
            left_emb=left_eval_emb,
            right_emb=right_eval_emb,
            gt_index=gt,
            batch_size=batch_size,
            csls_k=csls_k,
            csls_blend=csls_blend,
        )
        csls_metrics = {
            f"CSLS{metric_name}": metric_value
            for metric_name, metric_value in csls_ranking_metrics.items()
        }

    if calibrate_threshold:
        threshold_info = find_best_acceptance_threshold(
            top1_index=top1_index,
            top1_score=top1_score,
            gt_index=gt,
        )
        acceptance_threshold = threshold_info["threshold"]
    elif acceptance_threshold is None:
        acceptance_threshold = 0.0

    classification_metrics = compute_classification_metrics(
        top1_index=top1_index,
        top1_score=top1_score,
        gt_index=gt,
        acceptance_threshold=acceptance_threshold,
    )

    return {
        **ranking_metrics,
        **classification_metrics,
        **csls_metrics,
    }
