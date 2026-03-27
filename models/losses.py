import torch
import torch.nn.functional as F


def cosine_sim(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    x1 = F.normalize(x1, p=2, dim=-1)
    x2 = F.normalize(x2, p=2, dim=-1)
    return (x1 * x2).sum(dim=-1)


def reshape_sampled_negative_sim(
    pos_left: torch.Tensor,
    neg_left: torch.Tensor = None,
    neg_right: torch.Tensor = None,
) -> torch.Tensor:
    if neg_left is None or neg_right is None:
        return None

    batch_size = pos_left.size(0)
    if batch_size <= 0:
        return None

    neg_sim = cosine_sim(neg_left, neg_right)
    if neg_sim.numel() < batch_size:
        return None

    usable = (neg_sim.numel() // batch_size) * batch_size
    if usable <= 0:
        return None

    return neg_sim[:usable].view(batch_size, -1)


def stable_topk_mean(values: torch.Tensor, topk: int, dim: int = -1) -> torch.Tensor:
    if values.numel() == 0:
        raise ValueError("stable_topk_mean requires a non-empty tensor")

    size = values.size(dim)
    if size <= 0:
        raise ValueError("stable_topk_mean requires a positive-sized dimension")

    k = max(1, min(topk, size))
    safe_values = values.masked_fill(~torch.isfinite(values), float("-inf"))
    topk_values = safe_values.topk(k=k, dim=dim).values
    finite_mask = torch.isfinite(topk_values)
    denom = finite_mask.sum(dim=dim).clamp_min(1)
    return topk_values.masked_fill(~finite_mask, 0.0).sum(dim=dim) / denom


def info_nce_loss(
    x1: torch.Tensor,
    x2: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    双向 InfoNCE
    x1, x2: [B, D]
    默认第 i 个 x1 和第 i 个 x2 为正样本对
    """
    x1 = F.normalize(x1, p=2, dim=-1)
    x2 = F.normalize(x2, p=2, dim=-1)

    logits = torch.matmul(x1, x2.t()) / temperature
    labels = torch.arange(x1.size(0), device=x1.device)

    loss_12 = F.cross_entropy(logits, labels)
    loss_21 = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_12 + loss_21)


def smooth_alignment_loss(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    x1 = F.normalize(x1, p=2, dim=-1)
    x2 = F.normalize(x2, p=2, dim=-1)
    return F.smooth_l1_loss(x1, x2)


def smooth_consistency_loss(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    结构/语义一致性约束:
    - 余弦项约束两个空间的方向一致
    - SmoothL1 项约束两个空间的几何距离不过度偏离
    这比单独使用 cosine 或 L2 更稳定，也更贴合联合对齐空间的训练需求。
    """
    x1 = F.normalize(x1, p=2, dim=-1)
    x2 = F.normalize(x2, p=2, dim=-1)
    cos_term = 1.0 - (x1 * x2).sum(dim=-1).mean()
    smooth_term = F.smooth_l1_loss(x1, x2)
    return 0.5 * cos_term + 0.5 * smooth_term


def branch_alignment_loss(x1: torch.Tensor, x2: torch.Tensor, temperature: float) -> torch.Tensor:
    # Softer than pure InfoNCE: keep pair discrimination while stabilizing geometry.
    return 0.5 * info_nce_loss(x1, x2, temperature=temperature) + 0.5 * smooth_alignment_loss(x1, x2)


def topology_matching_loss(
    left_hop2: torch.Tensor,
    right_hop2: torch.Tensor,
    left_hop3: torch.Tensor,
    right_hop3: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Explicit 2/3-hop topology matching objective.
    The loss directly aligns higher-order structural summaries of positive pairs.
    """
    hop2_loss = branch_alignment_loss(left_hop2, right_hop2, temperature=temperature)
    hop3_loss = branch_alignment_loss(left_hop3, right_hop3, temperature=temperature)
    return 0.5 * (hop2_loss + hop3_loss)


def rampup_weight(progress: float, start: float, end: float) -> float:
    if progress <= start:
        return 0.0
    if progress >= end:
        return 1.0
    return (progress - start) / max(1e-6, end - start)


def margin_based_negative_loss(
    pos_left: torch.Tensor,
    pos_right: torch.Tensor,
    neg_left: torch.Tensor = None,
    neg_right: torch.Tensor = None,
    margin: float = 0.2,
    hard_weight: float = 1.0,
    per_pair_hard: bool = False,
    stable_topk: int = 1,
) -> torch.Tensor:
    """
    一个轻量可运行版负样本损失：
    - 正样本使用 batch 内对应 pair
    - 负样本使用外部采样得到的 neg pairs
    """
    if neg_left is None or neg_right is None:
        return pos_left.new_tensor(0.0)

    pos_sim = cosine_sim(pos_left, pos_right)   # [B]
    neg_per = reshape_sampled_negative_sim(
        pos_left=pos_left,
        neg_left=neg_left,
        neg_right=neg_right,
    )

    if neg_per is not None:
        if per_pair_hard:
            hard_neg = stable_topk_mean(neg_per, topk=stable_topk, dim=1)
            loss = F.relu(margin - pos_sim + hard_neg).mean()
        else:
            pos_ref = pos_sim.unsqueeze(1)
            loss = F.relu(margin - pos_ref + neg_per).mean()
    else:
        # Fallback to global reference when the shapes do not align.
        neg_sim = cosine_sim(neg_left, neg_right)   # [K]
        hard_neg = stable_topk_mean(neg_sim, topk=stable_topk, dim=0)
        pos_ref = pos_sim.mean()
        loss = F.relu(margin - pos_ref + hard_neg).mean()
    return hard_weight * loss


def explicit_negative_pair_loss(
    neg_left: torch.Tensor = None,
    neg_right: torch.Tensor = None,
    margin: float = 0.2,
) -> torch.Tensor:
    """
    Penalize explicitly confirmed negative pairs from active learning.
    Their similarity should stay below the decision margin.
    """
    if neg_left is None or neg_right is None:
        if neg_left is not None:
            return neg_left.new_tensor(0.0)
        if neg_right is not None:
            return neg_right.new_tensor(0.0)
        return torch.tensor(0.0)

    neg_sim = cosine_sim(neg_left, neg_right)
    return F.relu(neg_sim - margin).mean()


def hardest_negative_ranking_loss(
    pos_left: torch.Tensor,
    pos_right: torch.Tensor,
    margin: float = 0.2,
    sampled_neg_left: torch.Tensor = None,
    sampled_neg_right: torch.Tensor = None,
    batch_topk: int = 1,
    sampled_topk: int = 1,
) -> torch.Tensor:
    """
    Encourage the gold pair to outrank the hardest in-batch impostor on both
    left-to-right and right-to-left retrieval, and optionally the hardest
    sampled negative pairs collected by the mining pipeline.
    """
    if pos_left.size(0) <= 1 and (sampled_neg_left is None or sampled_neg_right is None):
        return pos_left.new_tensor(0.0)

    left = F.normalize(pos_left, p=2, dim=-1)
    right = F.normalize(pos_right, p=2, dim=-1)
    sim = left @ right.t()
    pos_sim = sim.diag()

    hardest_any = pos_sim.new_full(pos_sim.shape, -1e9)
    if pos_left.size(0) > 1:
        mask = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
        neg_sim = sim.masked_fill(mask, float("-inf"))
        hardest_right = stable_topk_mean(neg_sim, topk=batch_topk, dim=1)
        hardest_left = stable_topk_mean(neg_sim, topk=batch_topk, dim=0)
        hardest_any = torch.maximum(hardest_right, hardest_left)

    sampled_neg_per = reshape_sampled_negative_sim(
        pos_left=pos_left,
        neg_left=sampled_neg_left,
        neg_right=sampled_neg_right,
    )
    if sampled_neg_per is not None:
        sampled_hardest = stable_topk_mean(sampled_neg_per, topk=sampled_topk, dim=1)
        hardest_any = torch.maximum(hardest_any, sampled_hardest)

    valid_mask = hardest_any > -1e8
    if not valid_mask.any():
        return pos_left.new_tensor(0.0)
    return F.relu(margin - pos_sim[valid_mask] + hardest_any[valid_mask]).mean()


def structure_consistency_loss(
    z_struct: torch.Tensor,
    z_joint: torch.Tensor
) -> torch.Tensor:
    return smooth_consistency_loss(z_struct, z_joint)


def semantic_consistency_loss(
    z_sem: torch.Tensor,
    z_joint: torch.Tensor
) -> torch.Tensor:
    return smooth_consistency_loss(z_sem, z_joint)


def cross_modal_consistency_loss(
    z_struct: torch.Tensor,
    z_sem: torch.Tensor,
) -> torch.Tensor:
    return smooth_consistency_loss(z_struct, z_sem)


def joint_branch_consistency_loss(
    z_joint: torch.Tensor,
    z_branch: torch.Tensor,
) -> torch.Tensor:
    return smooth_consistency_loss(z_joint, z_branch)


def structural_self_supervised_loss(z_struct: torch.Tensor) -> torch.Tensor:
    """
    warmup 阶段的结构自监督稳定项。
    通过约束 embedding 各维度保持足够方差，避免结构空间在联合训练前塌缩。
    """
    z = F.normalize(z_struct, p=2, dim=-1)
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return F.relu(1.0 - std).mean()


def total_loss(
    left_outputs: dict,
    right_outputs: dict,
    lambda_branch_align: float = 0.2,
    lambda_cross_modal: float = 0.05,
    lambda_joint_branch: float = 0.05,
    lambda_struct: float = 0.2,
    lambda_sem: float = 0.2,
    lambda_neg: float = 0.2,
    lambda_ranking: float = 0.0,
    lambda_topology: float = 0.15,
    current_joint_epoch: int = 1,
    total_joint_epochs: int = 1,
    branch_align_start: float = 0.15,
    branch_align_end: float = 0.50,
    cross_modal_start: float = 0.25,
    cross_modal_end: float = 0.65,
    joint_branch_start: float = 0.40,
    joint_branch_end: float = 0.80,
    topology_start: float = 0.55,
    topology_end: float = 0.90,
    temperature: float = 0.07,
    neg_left_joint: torch.Tensor = None,
    neg_right_joint: torch.Tensor = None,
    al_neg_left_joint: torch.Tensor = None,
    al_neg_right_joint: torch.Tensor = None,
    margin: float = 0.2,
    hard_negative_weight: float = 1.0,
    al_negative_weight: float = 1.0,
    use_pairwise_hard_neg: bool = False,
    per_pair_hard_neg: bool = False,
    stable_neg_topk: int = 1,
    stable_ranking_topk: int = 1,
    warmup_mode: bool = False,
):
    """
    left_outputs / right_outputs:
    {
        "z_struct": ...,
        "z_sem": ...,
        "z_joint": ...
    }

    训练逻辑:
    - warmup: 先稳定结构空间
    - joint: 再优化联合对齐、结构一致性、语义一致性和负样本分离
    """

    # =========================
    # Stage 1: warmup
    # =========================
    if warmup_mode:
        struct_loss = 0.5 * (
            structural_self_supervised_loss(left_outputs["z_struct"]) +
            structural_self_supervised_loss(right_outputs["z_struct"])
        )

        zero = struct_loss.new_tensor(0.0)
        return {
            "loss": struct_loss,
            "align_loss": zero,
            "branch_align_loss": zero,
            "cross_modal_loss": zero,
            "joint_branch_loss": zero,
            "branch_align_scale": zero,
            "cross_modal_scale": zero,
            "joint_branch_scale": zero,
            "topology_scale": zero,
            "struct_loss": struct_loss,
            "sem_loss": zero,
            "neg_loss": zero,
            "al_neg_loss": zero,
            "ranking_loss": zero,
            "topology_loss": zero,
        }

    # =========================
    # Stage 2: joint training
    # =========================
    align = info_nce_loss(
        left_outputs["z_joint"],
        right_outputs["z_joint"],
        temperature=temperature
    )

    progress = current_joint_epoch / max(1, total_joint_epochs)
    branch_align_scale = rampup_weight(progress, branch_align_start, branch_align_end)
    cross_modal_scale = rampup_weight(progress, cross_modal_start, cross_modal_end)
    joint_branch_scale = rampup_weight(progress, joint_branch_start, joint_branch_end)
    topology_scale = rampup_weight(progress, topology_start, topology_end)

    left_branch = left_outputs.get("z_struct_enhanced", left_outputs["z_struct"])
    right_branch = right_outputs.get("z_struct_enhanced", right_outputs["z_struct"])
    left_sem_branch = left_outputs.get("z_sem_enhanced", left_outputs["z_sem"])
    right_sem_branch = right_outputs.get("z_sem_enhanced", right_outputs["z_sem"])

    branch_align = 0.5 * (
        branch_alignment_loss(left_branch, right_branch, temperature=temperature) +
        branch_alignment_loss(left_sem_branch, right_sem_branch, temperature=temperature)
    )

    cross_modal = 0.5 * (
        cross_modal_consistency_loss(left_branch, left_sem_branch) +
        cross_modal_consistency_loss(right_branch, right_sem_branch)
    )

    joint_branch = 0.25 * (
        joint_branch_consistency_loss(left_outputs["z_joint"], left_branch) +
        joint_branch_consistency_loss(left_outputs["z_joint"], left_sem_branch) +
        joint_branch_consistency_loss(right_outputs["z_joint"], right_branch) +
        joint_branch_consistency_loss(right_outputs["z_joint"], right_sem_branch)
    )

    struct_reg = 0.5 * (
        structure_consistency_loss(left_branch, left_outputs["z_joint"]) +
        structure_consistency_loss(right_branch, right_outputs["z_joint"])
    )

    sem_reg = 0.5 * (
        semantic_consistency_loss(left_sem_branch, left_outputs["z_joint"]) +
        semantic_consistency_loss(right_sem_branch, right_outputs["z_joint"])
    )

    if use_pairwise_hard_neg:
        neg_loss = hard_negative_weight * hardest_negative_ranking_loss(
            pos_left=left_outputs["z_joint"],
            pos_right=right_outputs["z_joint"],
            margin=margin,
        )
    else:
        neg_loss = margin_based_negative_loss(
            pos_left=left_outputs["z_joint"],
            pos_right=right_outputs["z_joint"],
            neg_left=neg_left_joint,
            neg_right=neg_right_joint,
            margin=margin,
            hard_weight=hard_negative_weight,
            per_pair_hard=per_pair_hard_neg,
            stable_topk=stable_neg_topk,
        )

    al_neg_loss = explicit_negative_pair_loss(
        neg_left=al_neg_left_joint,
        neg_right=al_neg_right_joint,
        margin=margin,
    )

    ranking_loss = hardest_negative_ranking_loss(
        pos_left=left_outputs["z_joint"],
        pos_right=right_outputs["z_joint"],
        margin=margin,
        sampled_neg_left=neg_left_joint,
        sampled_neg_right=neg_right_joint,
        batch_topk=stable_ranking_topk,
        sampled_topk=stable_ranking_topk,
    )

    topology_loss = topology_matching_loss(
        left_hop2=left_outputs["z_hop2"],
        right_hop2=right_outputs["z_hop2"],
        left_hop3=left_outputs["z_hop3"],
        right_hop3=right_outputs["z_hop3"],
        temperature=temperature,
    )

    loss = (
        align +
        (lambda_branch_align * branch_align_scale) * branch_align +
        (lambda_cross_modal * cross_modal_scale) * cross_modal +
        (lambda_joint_branch * joint_branch_scale) * joint_branch +
        lambda_struct * struct_reg +
        lambda_sem * sem_reg +
        lambda_neg * neg_loss +
        lambda_neg * al_negative_weight * al_neg_loss +
        lambda_ranking * ranking_loss +
        (lambda_topology * topology_scale) * topology_loss
    )

    return {
        "loss": loss,
        "align_loss": align,
        "branch_align_loss": branch_align,
        "cross_modal_loss": cross_modal,
        "joint_branch_loss": joint_branch,
        "branch_align_scale": align.new_tensor(branch_align_scale),
        "cross_modal_scale": align.new_tensor(cross_modal_scale),
        "joint_branch_scale": align.new_tensor(joint_branch_scale),
        "topology_scale": align.new_tensor(topology_scale),
        "struct_loss": struct_reg,
        "sem_loss": sem_reg,
        "neg_loss": neg_loss,
        "al_neg_loss": al_neg_loss,
        "ranking_loss": ranking_loss,
        "topology_loss": topology_loss,
    }
