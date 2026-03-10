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


def pair_list_to_tensors(pairs: List[Tuple[int, int]], device: torch.device):
    if len(pairs) == 0:
        return None, None
    left_ids = torch.tensor([l for l, _ in pairs], dtype=torch.long, device=device)
    right_ids = torch.tensor([r for _, r in pairs], dtype=torch.long, device=device)
    return left_ids, right_ids