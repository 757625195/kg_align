import random
from typing import Dict, List, Tuple, Set

import torch
import torch.nn.functional as F


def build_pair_dict(train_pairs: List[Tuple[int, int]]) -> Dict[int, int]:
    return {l: r for l, r in train_pairs}


def build_reverse_pair_dict(train_pairs: List[Tuple[int, int]]) -> Dict[int, int]:
    return {r: l for l, r in train_pairs}


def collect_entity_sets(train_pairs: List[Tuple[int, int]]):
    left_entities = sorted({l for l, _ in train_pairs})
    right_entities = sorted({r for _, r in train_pairs})
    return left_entities, right_entities


def random_negative_sampling(
    batch_pairs: List[Tuple[int, int]],
    all_left_entities: List[int],
    all_right_entities: List[int],
    known_pairs: Set[Tuple[int, int]],
    num_random_neg: int = 1,
):
    """
    对每个正样本 (l, r)，构造:
      (l, r_neg), (l_neg, r)
    """
    negatives = []

    for l, r in batch_pairs:
        for _ in range(num_random_neg):
            # negative on right
            rr = random.choice(all_right_entities)
            trial = 0
            while (l, rr) in known_pairs and trial < 20:
                rr = random.choice(all_right_entities)
                trial += 1
            negatives.append((l, rr))

            # negative on left
            ll = random.choice(all_left_entities)
            trial = 0
            while (ll, r) in known_pairs and trial < 20:
                ll = random.choice(all_left_entities)
                trial += 1
            negatives.append((ll, r))

    return negatives


@torch.no_grad()
def hard_negative_sampling(
    left_emb: torch.Tensor,
    right_emb: torch.Tensor,
    batch_pairs: List[Tuple[int, int]],
    batch_left_ids: torch.Tensor,
    batch_right_ids: torch.Tensor,
    all_left_ids: torch.Tensor,
    all_right_ids: torch.Tensor,
    known_pairs: Set[Tuple[int, int]],
    topk: int = 5,
    num_hard_neg: int = 1,
):
    """
    用当前 batch 的联合表示在 batch 内构造难负样本。
    先做一个轻量可跑版本：
    - 对每个 left，找 batch 内最像但不是对应 right 的若干候选
    - 对每个 right，找 batch 内最像但不是对应 left 的若干候选
    """
    negatives = []

    sim = F.normalize(left_emb, p=2, dim=-1) @ F.normalize(right_emb, p=2, dim=-1).t()

    batch_left_ids = batch_left_ids.detach().cpu().tolist()
    batch_right_ids = batch_right_ids.detach().cpu().tolist()

    for i, (l, r) in enumerate(batch_pairs):
        # hard negative on right
        row = sim[i].clone()
        pos_j = i
        row[pos_j] = -1e9
        cand_js = torch.topk(row, k=min(topk, row.size(0))).indices.tolist()

        picked = 0
        for j in cand_js:
            rr = batch_right_ids[j]
            if (l, rr) not in known_pairs:
                negatives.append((l, rr))
                picked += 1
                if picked >= num_hard_neg:
                    break

        # hard negative on left
        col = sim[:, i].clone()
        pos_i = i
        col[pos_i] = -1e9
        cand_is = torch.topk(col, k=min(topk, col.size(0))).indices.tolist()

        picked = 0
        for ii in cand_is:
            ll = batch_left_ids[ii]
            if (ll, r) not in known_pairs:
                negatives.append((ll, r))
                picked += 1
                if picked >= num_hard_neg:
                    break

    return negatives


@torch.no_grad()
def global_hard_negative_sampling(
    batch_pairs: List[Tuple[int, int]],
    batch_left_ids: torch.Tensor,
    batch_right_ids: torch.Tensor,
    global_left_ids: torch.Tensor,
    global_right_ids: torch.Tensor,
    global_left_emb: torch.Tensor,
    global_right_emb: torch.Tensor,
    known_pairs: Set[Tuple[int, int]],
    topk: int = 20,
    num_hard_neg: int = 1,
):
    """
    基于整个训练实体池构造难负样本：
    - 对每个 left，在全局 right 池中找最相似的非配对实体
    - 对每个 right，在全局 left 池中找最相似的非配对实体
    """
    negatives = []

    global_left_ids_list = global_left_ids.tolist()
    global_right_ids_list = global_right_ids.tolist()
    right_pos = {rid: idx for idx, rid in enumerate(global_right_ids_list)}
    left_pos = {lid: idx for idx, lid in enumerate(global_left_ids_list)}

    batch_left_emb = global_left_emb[[left_pos[int(l.item())] for l in batch_left_ids.cpu()]]
    batch_right_emb = global_right_emb[[right_pos[int(r.item())] for r in batch_right_ids.cpu()]]

    sim_lr = F.normalize(batch_left_emb, p=2, dim=-1) @ F.normalize(global_right_emb, p=2, dim=-1).t()
    sim_rl = F.normalize(batch_right_emb, p=2, dim=-1) @ F.normalize(global_left_emb, p=2, dim=-1).t()

    batch_left_ids_list = batch_left_ids.detach().cpu().tolist()
    batch_right_ids_list = batch_right_ids.detach().cpu().tolist()

    for i, (l, r) in enumerate(batch_pairs):
        row = sim_lr[i].clone()
        if r in right_pos:
            row[right_pos[r]] = -1e9
        cand_js = torch.topk(row, k=min(topk, row.size(0))).indices.tolist()

        picked = 0
        for j in cand_js:
            rr = global_right_ids_list[j]
            if (l, rr) not in known_pairs:
                negatives.append((l, rr))
                picked += 1
                if picked >= num_hard_neg:
                    break

        col = sim_rl[i].clone()
        if l in left_pos:
            col[left_pos[l]] = -1e9
        cand_is = torch.topk(col, k=min(topk, col.size(0))).indices.tolist()

        picked = 0
        for ii in cand_is:
            ll = global_left_ids_list[ii]
            if (ll, r) not in known_pairs:
                negatives.append((ll, r))
                picked += 1
                if picked >= num_hard_neg:
                    break

    return negatives


def _select_topk_candidates(
    scores: torch.Tensor,
    candidate_ids: List[int],
    fixed_id: int,
    known_pairs: Set[Tuple[int, int]],
    pick_right: bool,
    num_samples: int,
):
    selected = []
    for idx in scores.indices.tolist():
        candidate_id = candidate_ids[idx]
        pair = (fixed_id, candidate_id) if pick_right else (candidate_id, fixed_id)
        if pair in known_pairs:
            continue
        selected.append(pair)
        if len(selected) >= num_samples:
            break
    return selected


@torch.no_grad()
def queue_hard_negative_sampling(
    batch_pairs: List[Tuple[int, int]],
    batch_left_ids: torch.Tensor,
    batch_right_ids: torch.Tensor,
    batch_left_outputs: Dict[str, torch.Tensor],
    batch_right_outputs: Dict[str, torch.Tensor],
    bank: Dict[str, torch.Tensor],
    known_pairs: Set[Tuple[int, int]],
    joint_topk: int,
    num_global_hard_neg: int,
    num_conflict_neg: int,
):
    negatives = []

    bank_left_ids = bank["left_ids"].tolist()
    bank_right_ids = bank["right_ids"].tolist()
    left_pos = {entity_id: idx for idx, entity_id in enumerate(bank_left_ids)}
    right_pos = {entity_id: idx for idx, entity_id in enumerate(bank_right_ids)}

    batch_left_joint = F.normalize(batch_left_outputs["z_joint"].detach().cpu(), p=2, dim=-1)
    batch_right_joint = F.normalize(batch_right_outputs["z_joint"].detach().cpu(), p=2, dim=-1)
    batch_left_struct = F.normalize(batch_left_outputs["z_struct"].detach().cpu(), p=2, dim=-1)
    batch_right_struct = F.normalize(batch_right_outputs["z_struct"].detach().cpu(), p=2, dim=-1)
    batch_left_sem = F.normalize(batch_left_outputs["z_sem"].detach().cpu(), p=2, dim=-1)
    batch_right_sem = F.normalize(batch_right_outputs["z_sem"].detach().cpu(), p=2, dim=-1)

    bank_right_joint = F.normalize(bank["right_joint"], p=2, dim=-1)
    bank_left_joint = F.normalize(bank["left_joint"], p=2, dim=-1)
    bank_right_struct = F.normalize(bank["right_struct"], p=2, dim=-1)
    bank_left_struct = F.normalize(bank["left_struct"], p=2, dim=-1)
    bank_right_sem = F.normalize(bank["right_sem"], p=2, dim=-1)
    bank_left_sem = F.normalize(bank["left_sem"], p=2, dim=-1)

    batch_left_ids_list = batch_left_ids.detach().cpu().tolist()
    batch_right_ids_list = batch_right_ids.detach().cpu().tolist()

    for i, (l, r) in enumerate(batch_pairs):
        joint_scores_right = batch_left_joint[i] @ bank_right_joint.t()
        if r in right_pos:
            joint_scores_right[right_pos[r]] = -1e9
        joint_top_right = torch.topk(joint_scores_right, k=min(joint_topk, joint_scores_right.size(0)))
        negatives.extend(
            _select_topk_candidates(
                scores=joint_top_right,
                candidate_ids=bank_right_ids,
                fixed_id=l,
                known_pairs=known_pairs,
                pick_right=True,
                num_samples=num_global_hard_neg,
            )
        )

        joint_scores_left = batch_right_joint[i] @ bank_left_joint.t()
        if l in left_pos:
            joint_scores_left[left_pos[l]] = -1e9
        joint_top_left = torch.topk(joint_scores_left, k=min(joint_topk, joint_scores_left.size(0)))
        negatives.extend(
            _select_topk_candidates(
                scores=joint_top_left,
                candidate_ids=bank_left_ids,
                fixed_id=r,
                known_pairs=known_pairs,
                pick_right=False,
                num_samples=num_global_hard_neg,
            )
        )

        if num_conflict_neg <= 0:
            continue

        conflict_right_struct = batch_left_struct[i] @ bank_right_struct.t()
        conflict_right_sem = batch_left_sem[i] @ bank_right_sem.t()
        conflict_right_a = conflict_right_struct - conflict_right_sem
        conflict_right_b = conflict_right_sem - conflict_right_struct
        if r in right_pos:
            conflict_right_a[right_pos[r]] = -1e9
            conflict_right_b[right_pos[r]] = -1e9

        half_conflict = max(1, num_conflict_neg // 2)
        conflict_top_right_a = torch.topk(conflict_right_a, k=min(joint_topk, conflict_right_a.size(0)))
        conflict_top_right_b = torch.topk(conflict_right_b, k=min(joint_topk, conflict_right_b.size(0)))
        negatives.extend(
            _select_topk_candidates(
                scores=conflict_top_right_a,
                candidate_ids=bank_right_ids,
                fixed_id=l,
                known_pairs=known_pairs,
                pick_right=True,
                num_samples=half_conflict,
            )
        )
        negatives.extend(
            _select_topk_candidates(
                scores=conflict_top_right_b,
                candidate_ids=bank_right_ids,
                fixed_id=l,
                known_pairs=known_pairs,
                pick_right=True,
                num_samples=num_conflict_neg - half_conflict,
            )
        )

        conflict_left_struct = batch_right_struct[i] @ bank_left_struct.t()
        conflict_left_sem = batch_right_sem[i] @ bank_left_sem.t()
        conflict_left_a = conflict_left_struct - conflict_left_sem
        conflict_left_b = conflict_left_sem - conflict_left_struct
        if l in left_pos:
            conflict_left_a[left_pos[l]] = -1e9
            conflict_left_b[left_pos[l]] = -1e9

        conflict_top_left_a = torch.topk(conflict_left_a, k=min(joint_topk, conflict_left_a.size(0)))
        conflict_top_left_b = torch.topk(conflict_left_b, k=min(joint_topk, conflict_left_b.size(0)))
        negatives.extend(
            _select_topk_candidates(
                scores=conflict_top_left_a,
                candidate_ids=bank_left_ids,
                fixed_id=r,
                known_pairs=known_pairs,
                pick_right=False,
                num_samples=half_conflict,
            )
        )
        negatives.extend(
            _select_topk_candidates(
                scores=conflict_top_left_b,
                candidate_ids=bank_left_ids,
                fixed_id=r,
                known_pairs=known_pairs,
                pick_right=False,
                num_samples=num_conflict_neg - half_conflict,
            )
        )

    deduped = []
    seen = set()
    for pair in negatives:
        if pair in seen:
            continue
        seen.add(pair)
        deduped.append(pair)
    return deduped


def pair_list_to_tensors(pairs: List[Tuple[int, int]], device: torch.device):
    if len(pairs) == 0:
        return None, None
    left_ids = torch.tensor([l for l, _ in pairs], dtype=torch.long, device=device)
    right_ids = torch.tensor([r for _, r in pairs], dtype=torch.long, device=device)
    return left_ids, right_ids
