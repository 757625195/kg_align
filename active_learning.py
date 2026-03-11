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
def centrality_score(emb: torch.Tensor) -> torch.Tensor:
    emb = F.normalize(emb, p=2, dim=-1)
    center = F.normalize(emb.mean(dim=0, keepdim=True), p=2, dim=-1)
    score = (emb * center).sum(dim=-1)
    return minmax_norm(score)


@torch.no_grad()
def multi_view_representativeness(
    joint_emb: torch.Tensor,
    struct_emb: torch.Tensor,
    sem_emb: torch.Tensor,
    alpha_joint: float = 0.5,
    alpha_struct: float = 0.3,
    alpha_sem: float = 0.2,
) -> torch.Tensor:
    joint_score = centrality_score(joint_emb)
    struct_score = centrality_score(struct_emb)
    sem_score = centrality_score(sem_emb)
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
    budget: int = 100,
    alpha_uncertainty: float = 0.35,
    alpha_repr: float = 0.65,
):
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
    rep = multi_view_representativeness(
        joint_emb=left_outputs["z_joint"],
        struct_emb=left_outputs["z_struct_enhanced"],
        sem_emb=left_outputs["z_sem_enhanced"],
    )
    final_score = alpha_uncertainty * unc + alpha_repr * rep

    rank_idx = torch.argsort(final_score, descending=True).tolist()

    candidates = []
    kept_rank_positions = []

    for i in rank_idx:
        l = int(left_ids[i].item())
        matched_j = int(pred_j[i].item())
        r = int(right_ids[matched_j].item())
        if (l, r) in known_pairs:
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
        kept_rank_positions.append(i)
        if len(candidates) >= budget * 3:
            break

    if not candidates:
        return []

    cand_joint = left_outputs["z_joint"][torch.tensor(kept_rank_positions)]
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


def simulate_human_annotation(candidate_items, gold_test_pairs):
    new_positive = []
    new_negative = []

    for item in candidate_items:
        p = item["pair"]
        if p in gold_test_pairs:
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
    test_pairs,
    device,
):
    model.eval()

    known_pairs = set(train_pairs)
    gold_test_pairs = set(test_pairs)

    left_ids = torch.tensor(sorted({l for l, _ in test_pairs}), dtype=torch.long)
    right_ids = torch.tensor(sorted({r for _, r in test_pairs}), dtype=torch.long)

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
        budget=cfg.al_budget,
    )

    filtered_candidates, rejected = filter_active_learning_candidates(
        candidates=candidates,
        min_confidence=cfg.al_min_confidence,
        min_margin=cfg.al_min_margin,
        min_uncertainty=cfg.al_min_uncertainty,
        max_uncertainty=cfg.al_max_uncertainty,
        require_bidirectional=cfg.al_require_bidirectional,
    )

    filtered_candidates = filtered_candidates[:cfg.al_budget]

    new_pos, new_neg = simulate_human_annotation(filtered_candidates, gold_test_pairs)

    added = 0
    for p in new_pos:
        if p not in known_pairs:
            train_pairs.append(p)
            known_pairs.add(p)
            added += 1

    return {
        "candidates": len(candidates),
        "filtered_candidates": len(filtered_candidates),
        "new_positive": len(new_pos),
        "new_negative": len(new_neg),
        "added_to_train": added,
        "rejected_bidirectional": rejected["bidirectional"],
        "rejected_confidence": rejected["confidence"],
        "rejected_margin": rejected["margin"],
        "rejected_low_uncertainty": rejected["low_uncertainty"],
        "rejected_uncertainty": rejected["uncertainty"],
    }
