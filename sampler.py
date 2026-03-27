import random
from typing import Callable, Dict, List, Optional, Set, Tuple

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
    return_grouped: bool = False,
):
    """
    对每个正样本 (l, r)，构造:
      (l, r_neg), (l_neg, r)
    """
    negatives = []
    grouped_negatives = []

    for l, r in batch_pairs:
        pair_negatives = []
        for _ in range(num_random_neg):
            # negative on right
            rr = random.choice(all_right_entities)
            trial = 0
            while (l, rr) in known_pairs and trial < 20:
                rr = random.choice(all_right_entities)
                trial += 1
            pair_negatives.append((l, rr))

            # negative on left
            ll = random.choice(all_left_entities)
            trial = 0
            while (ll, r) in known_pairs and trial < 20:
                ll = random.choice(all_left_entities)
                trial += 1
            pair_negatives.append((ll, r))

        if return_grouped:
            grouped_negatives.append(pair_negatives)
        else:
            negatives.extend(pair_negatives)

    return grouped_negatives if return_grouped else negatives


def _build_pair(fixed_id: int, candidate_id: int, pick_right: bool) -> Tuple[int, int]:
    return (fixed_id, candidate_id) if pick_right else (candidate_id, fixed_id)


def _select_ranked_candidates(
    scores: torch.Tensor,
    candidate_ids: List[int],
    fixed_id: int,
    known_pairs: Set[Tuple[int, int]],
    pick_right: bool,
    num_samples: int,
    excluded_id: int = None,
    validator: Optional[Callable[[int, int], bool]] = None,
) -> List[Tuple[int, int]]:
    selected = []
    used_pairs = set()

    for idx in torch.argsort(scores, descending=True).tolist():
        candidate_id = candidate_ids[idx]
        if excluded_id is not None and candidate_id == excluded_id:
            continue
        if validator is not None and not validator(candidate_id, idx):
            continue
        pair = _build_pair(fixed_id, candidate_id, pick_right)
        if pair in known_pairs or pair in used_pairs:
            continue
        selected.append(pair)
        used_pairs.add(pair)
        if len(selected) >= num_samples:
            return selected

    if len(selected) >= num_samples:
        return selected

    shuffled_candidates = list(candidate_ids)
    random.shuffle(shuffled_candidates)
    for candidate_id in shuffled_candidates:
        if excluded_id is not None and candidate_id == excluded_id:
            continue
        idx = candidate_ids.index(candidate_id)
        if validator is not None and not validator(candidate_id, idx):
            continue
        pair = _build_pair(fixed_id, candidate_id, pick_right)
        if pair in known_pairs or pair in used_pairs:
            continue
        selected.append(pair)
        used_pairs.add(pair)
        if len(selected) >= num_samples:
            break

    return selected


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
    return_grouped: bool = False,
):
    """
    用当前 batch 的联合表示在 batch 内构造难负样本。
    先做一个轻量可跑版本：
    - 对每个 left，找 batch 内最像但不是对应 right 的若干候选
    - 对每个 right，找 batch 内最像但不是对应 left 的若干候选
    """
    negatives = []
    grouped_negatives = []

    sim = F.normalize(left_emb, p=2, dim=-1) @ F.normalize(right_emb, p=2, dim=-1).t()

    batch_left_ids = batch_left_ids.detach().cpu().tolist()
    batch_right_ids = batch_right_ids.detach().cpu().tolist()

    for i, (l, r) in enumerate(batch_pairs):
        pair_negatives = []
        # hard negative on right
        row = sim[i].clone()
        pair_negatives.extend(
            _select_ranked_candidates(
                scores=row,
                candidate_ids=batch_right_ids,
                fixed_id=l,
                known_pairs=known_pairs,
                pick_right=True,
                num_samples=num_hard_neg,
                excluded_id=r,
            )
        )

        # hard negative on left
        col = sim[:, i].clone()
        pair_negatives.extend(
            _select_ranked_candidates(
                scores=col,
                candidate_ids=batch_left_ids,
                fixed_id=r,
                known_pairs=known_pairs,
                pick_right=False,
                num_samples=num_hard_neg,
                excluded_id=l,
            )
        )

        if return_grouped:
            grouped_negatives.append(pair_negatives)
        else:
            negatives.extend(pair_negatives)

    return grouped_negatives if return_grouped else negatives


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
    reciprocal_guard_topk: int = 0,
    return_grouped: bool = False,
):
    negatives = []
    grouped_negatives = []

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
    right_guard_cache: Dict[int, Set[int]] = {}
    left_guard_cache: Dict[int, Set[int]] = {}

    def right_candidate_is_safe(left_entity_id: int, right_entity_id: int) -> bool:
        if reciprocal_guard_topk <= 0:
            return True
        cached = right_guard_cache.get(right_entity_id)
        if cached is None:
            reverse_scores = bank_right_joint[right_pos[right_entity_id]] @ bank_left_joint.t()
            guard_k = min(reciprocal_guard_topk, reverse_scores.size(0))
            cached = {
                bank_left_ids[idx]
                for idx in torch.topk(reverse_scores, k=guard_k).indices.tolist()
            }
            right_guard_cache[right_entity_id] = cached
        return left_entity_id not in cached

    def left_candidate_is_safe(right_entity_id: int, left_entity_id: int) -> bool:
        if reciprocal_guard_topk <= 0:
            return True
        cached = left_guard_cache.get(left_entity_id)
        if cached is None:
            reverse_scores = bank_left_joint[left_pos[left_entity_id]] @ bank_right_joint.t()
            guard_k = min(reciprocal_guard_topk, reverse_scores.size(0))
            cached = {
                bank_right_ids[idx]
                for idx in torch.topk(reverse_scores, k=guard_k).indices.tolist()
            }
            left_guard_cache[left_entity_id] = cached
        return right_entity_id not in cached

    for i, (l, r) in enumerate(batch_pairs):
        pair_negatives = []

        joint_scores_right = batch_left_joint[i] @ bank_right_joint.t()
        pair_negatives.extend(
            _select_ranked_candidates(
                scores=joint_scores_right,
                candidate_ids=bank_right_ids,
                fixed_id=l,
                known_pairs=known_pairs,
                pick_right=True,
                num_samples=num_global_hard_neg,
                excluded_id=r,
                validator=lambda candidate_id, _idx, left_entity_id=l: right_candidate_is_safe(
                    left_entity_id, candidate_id
                ),
            )
        )

        joint_scores_left = batch_right_joint[i] @ bank_left_joint.t()
        pair_negatives.extend(
            _select_ranked_candidates(
                scores=joint_scores_left,
                candidate_ids=bank_left_ids,
                fixed_id=r,
                known_pairs=known_pairs,
                pick_right=False,
                num_samples=num_global_hard_neg,
                excluded_id=l,
                validator=lambda candidate_id, _idx, right_entity_id=r: left_candidate_is_safe(
                    right_entity_id, candidate_id
                ),
            )
        )

        if num_conflict_neg <= 0:
            if return_grouped:
                grouped_negatives.append(pair_negatives)
            else:
                negatives.extend(pair_negatives)
            continue

        conflict_right_struct = batch_left_struct[i] @ bank_right_struct.t()
        conflict_right_sem = batch_left_sem[i] @ bank_right_sem.t()
        conflict_right_a = conflict_right_struct - conflict_right_sem
        conflict_right_b = conflict_right_sem - conflict_right_struct
        if r in right_pos:
            conflict_right_a[right_pos[r]] = -1e9
            conflict_right_b[right_pos[r]] = -1e9

        half_conflict = max(1, num_conflict_neg // 2)
        pair_negatives.extend(
            _select_ranked_candidates(
                scores=conflict_right_a,
                candidate_ids=bank_right_ids,
                fixed_id=l,
                known_pairs=known_pairs,
                pick_right=True,
                num_samples=half_conflict,
                excluded_id=r,
            )
        )
        pair_negatives.extend(
            _select_ranked_candidates(
                scores=conflict_right_b,
                candidate_ids=bank_right_ids,
                fixed_id=l,
                known_pairs=known_pairs,
                pick_right=True,
                num_samples=num_conflict_neg - half_conflict,
                excluded_id=r,
            )
        )

        conflict_left_struct = batch_right_struct[i] @ bank_left_struct.t()
        conflict_left_sem = batch_right_sem[i] @ bank_left_sem.t()
        conflict_left_a = conflict_left_struct - conflict_left_sem
        conflict_left_b = conflict_left_sem - conflict_left_struct
        if l in left_pos:
            conflict_left_a[left_pos[l]] = -1e9
            conflict_left_b[left_pos[l]] = -1e9

        pair_negatives.extend(
            _select_ranked_candidates(
                scores=conflict_left_a,
                candidate_ids=bank_left_ids,
                fixed_id=r,
                known_pairs=known_pairs,
                pick_right=False,
                num_samples=half_conflict,
                excluded_id=l,
            )
        )
        pair_negatives.extend(
            _select_ranked_candidates(
                scores=conflict_left_b,
                candidate_ids=bank_left_ids,
                fixed_id=r,
                known_pairs=known_pairs,
                pick_right=False,
                num_samples=num_conflict_neg - half_conflict,
                excluded_id=l,
            )
        )

        if return_grouped:
            grouped_negatives.append(pair_negatives)
        else:
            negatives.extend(pair_negatives)

    return grouped_negatives if return_grouped else negatives


def pair_list_to_tensors(pairs: List[Tuple[int, int]], device: torch.device):
    if len(pairs) == 0:
        return None, None
    left_ids = torch.tensor([l for l, _ in pairs], dtype=torch.long, device=device)
    right_ids = torch.tensor([r for _, r in pairs], dtype=torch.long, device=device)
    return left_ids, right_ids
