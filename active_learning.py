from typing import List, Tuple, Set

import torch
import torch.nn.functional as F

from evaluate import encode_entity_outputs


def minmax_norm(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min() + 1e-9)


@torch.no_grad()
def compute_similarity(left_emb: torch.Tensor, right_emb: torch.Tensor) -> torch.Tensor:
    left_emb = F.normalize(left_emb, p=2, dim=-1)
    right_emb = F.normalize(right_emb, p=2, dim=-1)
    return left_emb @ right_emb.t()


@torch.no_grad()
def bidirectional_uncertainty(model, left_joint: torch.Tensor, right_joint: torch.Tensor) -> torch.Tensor:
    sim_lr = model.score_pairs(left_joint, right_joint).cpu()
    sim_rl = sim_lr.t()

    stats_lr = model.align_head.pair_confidence(sim_lr)
    stats_rl = model.align_head.pair_confidence(sim_rl)

    unc_lr = minmax_norm(stats_lr["uncertainty"])
    unc_rl = minmax_norm(stats_rl["uncertainty"])

    paired_unc_rl = unc_rl[sim_lr.argmax(dim=-1)]
    return 0.5 * unc_lr + 0.5 * paired_unc_rl


@torch.no_grad()
def blockwise_pair_stats(
    left_emb: torch.Tensor,
    right_emb: torch.Tensor,
    temperature: float,
    block_size: int = 1024,
):
    left_emb = F.normalize(left_emb, p=2, dim=-1)
    right_emb = F.normalize(right_emb, p=2, dim=-1)
    num_left = left_emb.size(0)
    num_right = right_emb.size(0)
    topk = min(2, num_right)

    top_scores = left_emb.new_full((num_left, topk), -1e9)
    top_indices = torch.full((num_left, topk), -1, dtype=torch.long, device=left_emb.device)

    z = left_emb.new_zeros(num_left)
    s = left_emb.new_zeros(num_left)

    best_left_scores = right_emb.new_full((num_right,), -1e9)
    best_left_indices = torch.full((num_right,), -1, dtype=torch.long, device=right_emb.device)

    for start in range(0, num_right, block_size):
        end = min(num_right, start + block_size)
        right_block = right_emb[start:end]
        sim_block = left_emb @ right_block.t()
        logits = sim_block / temperature

        exp_logits = torch.exp(logits)
        z += exp_logits.sum(dim=1)
        s += (exp_logits * logits).sum(dim=1)

        block_top_scores, block_top_idx = torch.topk(sim_block, k=min(topk, sim_block.size(1)), dim=1)
        block_top_idx = block_top_idx + start
        merged_scores = torch.cat([top_scores, block_top_scores], dim=1)
        merged_idx = torch.cat([top_indices, block_top_idx], dim=1)
        new_scores, new_pos = torch.topk(merged_scores, k=topk, dim=1)
        new_idx = torch.gather(merged_idx, 1, new_pos)
        top_scores, top_indices = new_scores, new_idx

        block_max_scores, block_argmax_left = sim_block.max(dim=0)
        update_mask = block_max_scores > best_left_scores[start:end]
        best_left_scores[start:end][update_mask] = block_max_scores[update_mask]
        best_left_indices[start:end][update_mask] = block_argmax_left[update_mask]

    if topk == 1:
        margin = torch.ones_like(top_scores[:, 0])
    else:
        margin = top_scores[:, 0] - top_scores[:, 1]

    max_logit = top_scores[:, 0] / temperature
    confidence = torch.exp(max_logit) / (z + 1e-9)
    entropy = torch.log(z + 1e-9) - (s / (z + 1e-9))
    uncertainty = 0.5 * entropy + 0.5 * (1.0 - margin)

    return {
        "confidence": confidence,
        "margin": margin,
        "uncertainty": uncertainty,
        "pred_j": top_indices[:, 0],
        "pred_i": best_left_indices,
    }


@torch.no_grad()
def centrality_score(emb: torch.Tensor) -> torch.Tensor:
    emb = F.normalize(emb, p=2, dim=-1)
    center = F.normalize(emb.mean(dim=0, keepdim=True), p=2, dim=-1)
    score = (emb * center).sum(dim=-1)
    return minmax_norm(score)


@torch.no_grad()
def pair_embedding(
    left_emb: torch.Tensor,
    right_emb: torch.Tensor,
) -> torch.Tensor:
    return F.normalize(torch.cat([left_emb, right_emb], dim=-1), p=2, dim=-1)


@torch.no_grad()
def multi_view_pair_representativeness(
    left_outputs: dict,
    right_outputs: dict,
    matched_right_idx: torch.Tensor,
    alpha_joint: float = 0.5,
    alpha_struct: float = 0.3,
    alpha_sem: float = 0.2,
) -> torch.Tensor:
    """
    Estimate representativeness for candidate pairs rather than only the left
    entities. The matched right entity participates in joint/structural/semantic
    centrality estimation, which better matches the unified alignment-space view
    described in the guidance.
    """
    matched_right_idx = matched_right_idx.to(
        device=right_outputs["z_joint"].device,
        dtype=torch.long,
    )

    left_joint_score = centrality_score(left_outputs["z_joint"])
    right_joint_score = centrality_score(right_outputs["z_joint"])[matched_right_idx]

    left_struct_score = centrality_score(left_outputs["z_struct_enhanced"])
    right_struct_score = centrality_score(right_outputs["z_struct_enhanced"])[matched_right_idx]

    left_sem_score = centrality_score(left_outputs["z_sem_enhanced"])
    right_sem_score = centrality_score(right_outputs["z_sem_enhanced"])[matched_right_idx]

    joint_score = 0.5 * (left_joint_score + right_joint_score)
    struct_score = 0.5 * (left_struct_score + right_struct_score)
    sem_score = 0.5 * (left_sem_score + right_sem_score)
    return (
        alpha_joint * joint_score +
        alpha_struct * struct_score +
        alpha_sem * sem_score
    )


@torch.no_grad()
def diversity_rerank(selected_indices: List[int], candidate_emb: torch.Tensor, max_keep: int):
    if len(selected_indices) <= max_keep:
        return selected_indices

    candidate_emb = F.normalize(candidate_emb, p=2, dim=-1)
    kept = [selected_indices[0]]
    remain = selected_indices[1:]

    while len(kept) < max_keep and remain:
        kept_emb = candidate_emb[kept]
        best_idx = None
        best_score = None

        for idx in remain:
            sim = (candidate_emb[idx:idx + 1] @ kept_emb.t()).max().item()
            score = -sim
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx

        kept.append(best_idx)
        remain.remove(best_idx)

    return kept


@torch.no_grad()
def select_active_learning_candidates(
    model,
    left_ids: torch.Tensor,
    right_ids: torch.Tensor,
    left_outputs: dict,
    right_outputs: dict,
    known_pairs: Set[Tuple[int, int]],
    queried_pairs: Set[Tuple[int, int]],
    budget: int = 100,
    alpha_uncertainty: float = 0.35,
    alpha_repr: float = 0.65,
    block_size: int = 1024,
    use_blockwise: bool = False,
    use_diversity: bool = True,
):
    if use_blockwise:
        stats_lr = blockwise_pair_stats(
            left_outputs["z_joint"],
            right_outputs["z_joint"],
            temperature=model.align_head.temperature,
            block_size=block_size,
        )
        stats_rl = blockwise_pair_stats(
            right_outputs["z_joint"],
            left_outputs["z_joint"],
            temperature=model.align_head.temperature,
            block_size=block_size,
        )
        pred_j = stats_lr["pred_j"]
        pred_i = stats_lr["pred_i"]

        unc_lr = minmax_norm(stats_lr["uncertainty"])
        unc_rl = minmax_norm(stats_rl["uncertainty"])
        paired_unc_rl = unc_rl[pred_j]
        unc = 0.5 * unc_lr + 0.5 * paired_unc_rl
    else:
        sim = model.score_pairs(left_outputs["z_joint"], right_outputs["z_joint"]).cpu()
        sim_rl = sim.t()
        pred_j = sim.argmax(dim=-1)
        pred_i = sim_rl.argmax(dim=-1)
        stats_lr = model.align_head.pair_confidence(sim)
        stats_rl = model.align_head.pair_confidence(sim_rl)

        unc = bidirectional_uncertainty(
            model=model,
            left_joint=left_outputs["z_joint"],
            right_joint=right_outputs["z_joint"],
        )
    rep = multi_view_pair_representativeness(
        left_outputs=left_outputs,
        right_outputs=right_outputs,
        matched_right_idx=pred_j,
    )
    final_score = alpha_uncertainty * unc + alpha_repr * rep

    rank_idx = torch.argsort(final_score, descending=True).tolist()

    candidates = []
    for i in rank_idx:
        l = int(left_ids[i].item())
        matched_j = int(pred_j[i].item())
        r = int(right_ids[matched_j].item())
        if (l, r) in known_pairs or (l, r) in queried_pairs:
            continue
        reciprocal = int(pred_i[matched_j].item()) == i
        candidates.append({
            "pair": (l, r),
            "left_index": i,
            "right_index": matched_j,
            "score": float(final_score[i].item()),
            "uncertainty": float(unc[i].item()),
            "confidence_lr": float(stats_lr["confidence"][i].item()),
            "confidence_rl": float(stats_rl["confidence"][matched_j].item()),
            "margin_lr": float(stats_lr["margin"][i].item()),
            "margin_rl": float(stats_rl["margin"][matched_j].item()),
            "reciprocal": reciprocal,
        })
        if len(candidates) >= budget * 3:
            break

    if not candidates:
        return []

    if not use_diversity:
        return candidates[:min(budget, len(candidates))]

    left_device = left_outputs["z_joint"].device
    right_device = right_outputs["z_joint"].device
    cand_left_idx = torch.tensor(
        [item["left_index"] for item in candidates],
        dtype=torch.long,
        device=left_device,
    )
    cand_right_idx = torch.tensor(
        [item["right_index"] for item in candidates],
        dtype=torch.long,
        device=right_device,
    )
    cand_joint = pair_embedding(
        left_outputs["z_joint"][cand_left_idx],
        right_outputs["z_joint"][cand_right_idx],
    )
    selected_local = diversity_rerank(
        selected_indices=list(range(len(candidates))),
        candidate_emb=cand_joint,
        max_keep=min(budget, len(candidates)),
    )

    return [candidates[i] for i in selected_local]


def filter_active_learning_candidates(
    candidates,
    min_confidence: float,
    min_margin: float,
    min_uncertainty: float,
    max_uncertainty: float,
    require_bidirectional: bool,
):
    filtered = []
    rejected = {
        "bidirectional": 0,
        "confidence": 0,
        "margin": 0,
        "low_uncertainty": 0,
        "uncertainty": 0,
    }

    for item in candidates:
        if require_bidirectional and not item["reciprocal"]:
            rejected["bidirectional"] += 1
            continue

        confidence = min(item["confidence_lr"], item["confidence_rl"])
        if confidence < min_confidence:
            rejected["confidence"] += 1
            continue

        margin = min(item["margin_lr"], item["margin_rl"])
        if margin < min_margin:
            rejected["margin"] += 1
            continue

        if item["uncertainty"] < min_uncertainty:
            rejected["low_uncertainty"] += 1
            continue

        if item["uncertainty"] > max_uncertainty:
            rejected["uncertainty"] += 1
            continue

        filtered.append(item)

    return filtered, rejected


def adaptive_filter_active_learning_candidates(
    candidates,
    budget: int,
    min_confidence: float,
    min_margin: float,
    min_uncertainty: float,
    max_uncertainty: float,
    require_bidirectional: bool,
    max_stage_name: str = "score_only",
    min_fill_ratio: float = 0.25,
):
    """
    Human-in-the-loop active learning should not get stuck because pseudo-label
    style filters are too strict. We therefore try a small ladder of increasingly
    permissive filters until we obtain a usable number of candidates.

    Important principle: we should relax confidence/margin requirements before
    we give up bidirectional consistency, because reciprocal matches are usually
    higher-quality annotation requests.
    """
    min_keep = min(budget, max(1, int(round(budget * min_fill_ratio))))
    stages = [
        {
            "name": "strict",
            "min_confidence": min_confidence,
            "min_margin": min_margin,
            "min_uncertainty": min_uncertainty,
            "max_uncertainty": max_uncertainty,
            "require_bidirectional": require_bidirectional,
        },
        {
            "name": "no_confidence_gate",
            "min_confidence": 0.0,
            "min_margin": min_margin,
            "min_uncertainty": min_uncertainty,
            "max_uncertainty": max_uncertainty,
            "require_bidirectional": require_bidirectional,
        },
        {
            "name": "reciprocal_relaxed_margin",
            "min_confidence": 0.0,
            "min_margin": 0.0,
            "min_uncertainty": min_uncertainty,
            "max_uncertainty": max_uncertainty,
            "require_bidirectional": require_bidirectional,
        },
        {
            "name": "allow_nonreciprocal",
            "min_confidence": 0.0,
            "min_margin": 0.0,
            "min_uncertainty": min_uncertainty,
            "max_uncertainty": max(0.98, max_uncertainty),
            "require_bidirectional": False,
        },
        {
            "name": "score_only",
            "min_confidence": 0.0,
            "min_margin": 0.0,
            "min_uncertainty": 0.0,
            "max_uncertainty": 1.0,
            "require_bidirectional": False,
        },
    ]
    stage_names = [stage["name"] for stage in stages]
    if max_stage_name not in stage_names:
        raise ValueError(
            f"Unsupported active-learning filter stage: {max_stage_name}. "
            f"Expected one of {stage_names}."
        )
    stages = stages[: stage_names.index(max_stage_name) + 1]

    strict_filtered = []
    strict_rejected = {
        "bidirectional": 0,
        "confidence": 0,
        "margin": 0,
        "low_uncertainty": 0,
        "uncertainty": 0,
    }

    for stage_idx, stage in enumerate(stages):
        filtered, rejected = filter_active_learning_candidates(
            candidates=candidates,
            min_confidence=stage["min_confidence"],
            min_margin=stage["min_margin"],
            min_uncertainty=stage["min_uncertainty"],
            max_uncertainty=stage["max_uncertainty"],
            require_bidirectional=stage["require_bidirectional"],
        )
        if stage_idx == 0:
            strict_filtered = filtered
            strict_rejected = rejected
        if len(filtered) >= min_keep or stage["name"] == "score_only":
            return filtered[:budget], strict_filtered[:budget], strict_rejected, stage["name"]

    return [], strict_filtered[:budget], strict_rejected, "strict"


def simulate_human_annotation(candidate_items, gold_pool_pairs):
    new_positive = []
    new_negative = []

    for item in candidate_items:
        p = item["pair"]
        if p in gold_pool_pairs:
            new_positive.append(p)
        else:
            new_negative.append(p)

    return new_positive, new_negative


@torch.no_grad()
def run_active_learning_round(
    model,
    cfg,
    edge_index,
    seq_features,
    adj_list,
    train_pairs,
    pool_pairs,
    gold_pairs,
    pool_left_ids,
    pool_right_ids,
    queried_pairs,
    device,
):
    model.eval()

    known_pairs = set(train_pairs)
    gold_pool_pairs = set(gold_pairs) if gold_pairs is not None else set(pool_pairs)

    if len(pool_pairs) == 0 and not pool_left_ids:
        return {
            "candidates": 0,
            "strict_filtered_candidates": 0,
            "filtered_candidates": 0,
            "new_positive": 0,
            "new_negative": 0,
            "added_to_train": 0,
            "rejected_bidirectional": 0,
            "rejected_confidence": 0,
            "rejected_margin": 0,
            "rejected_low_uncertainty": 0,
            "rejected_uncertainty": 0,
            "remaining_pool": 0,
            "negative_pairs": [],
            "filter_stage": "strict",
        }

    if pool_left_ids and pool_right_ids:
        left_ids = torch.tensor(pool_left_ids, dtype=torch.long)
        right_ids = torch.tensor(pool_right_ids, dtype=torch.long)
    else:
        left_ids = torch.tensor(sorted({l for l, _ in pool_pairs}), dtype=torch.long)
        right_ids = torch.tensor(sorted({r for _, r in pool_pairs}), dtype=torch.long)

    left_outputs = encode_entity_outputs(
        model=model,
        node_ids=left_ids,
        edge_index=edge_index,
        seq_features=seq_features,
        adj_list=adj_list,
        num_neighbors=cfg.num_neighbors,
        batch_size=cfg.eval_batch_size,
        device=device,
    )
    right_outputs = encode_entity_outputs(
        model=model,
        node_ids=right_ids,
        edge_index=edge_index,
        seq_features=seq_features,
        adj_list=adj_list,
        num_neighbors=cfg.num_neighbors,
        batch_size=cfg.eval_batch_size,
        device=device,
    )

    candidates = select_active_learning_candidates(
        model=model,
        left_ids=left_ids,
        right_ids=right_ids,
        left_outputs=left_outputs,
        right_outputs=right_outputs,
        known_pairs=known_pairs,
        queried_pairs=queried_pairs,
        budget=cfg.al_budget,
        alpha_uncertainty=cfg.al_alpha_uncertainty,
        alpha_repr=cfg.al_alpha_repr,
        block_size=cfg.al_block_size,
        use_blockwise=bool(pool_left_ids),
        use_diversity=cfg.al_use_diversity,
    )

    filtered_candidates, strict_filtered_candidates, rejected, filter_stage = adaptive_filter_active_learning_candidates(
        candidates=candidates,
        budget=cfg.al_budget,
        min_confidence=cfg.al_min_confidence,
        min_margin=cfg.al_min_margin,
        min_uncertainty=cfg.al_min_uncertainty,
        max_uncertainty=cfg.al_max_uncertainty,
        require_bidirectional=cfg.al_require_bidirectional,
        max_stage_name=cfg.al_max_filter_stage,
    )

    new_pos, new_neg = simulate_human_annotation(filtered_candidates, gold_pool_pairs)
    queried_pairs.update(item["pair"] for item in filtered_candidates)

    added = 0
    for p in new_pos:
        if p not in known_pairs:
            train_pairs.append(p)
            known_pairs.add(p)
            added += 1

    if new_pos and pool_pairs:
        pool_pairs[:] = [p for p in pool_pairs if p not in set(new_pos)]

    return {
        "candidates": len(candidates),
        "strict_filtered_candidates": len(strict_filtered_candidates),
        "filtered_candidates": len(filtered_candidates),
        "new_positive": len(new_pos),
        "new_negative": len(new_neg),
        "added_to_train": added,
        "rejected_bidirectional": rejected["bidirectional"],
        "rejected_confidence": rejected["confidence"],
        "rejected_margin": rejected["margin"],
        "rejected_low_uncertainty": rejected["low_uncertainty"],
        "rejected_uncertainty": rejected["uncertainty"],
        "remaining_pool": len(pool_pairs) if pool_pairs else len(left_ids) * len(right_ids),
        "negative_pairs": list(new_neg),
        "filter_stage": filter_stage,
    }
