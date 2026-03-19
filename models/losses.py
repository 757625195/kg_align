import torch
import torch.nn.functional as F


def cosine_sim(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    x1 = F.normalize(x1, p=2, dim=-1)
    x2 = F.normalize(x2, p=2, dim=-1)
    return (x1 * x2).sum(dim=-1)


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
) -> torch.Tensor:
    """
    一个轻量可运行版负样本损失：
    - 正样本使用 batch 内对应 pair
    - 负样本使用外部采样得到的 neg pairs
    """
    if neg_left is None or neg_right is None:
        return pos_left.new_tensor(0.0)

    pos_sim = cosine_sim(pos_left, pos_right)   # [B]
    neg_sim = cosine_sim(neg_left, neg_right)   # [K]

    # 用正样本平均相似度作为参考
    pos_ref = pos_sim.mean()

    loss = F.relu(margin - pos_ref + neg_sim).mean()
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
    lambda_topology: float = 0.15,
    current_joint_epoch: int = 1,
    total_joint_epochs: int = 1,
    branch_align_start: float = 0.15,
    branch_align_end: float = 0.50,
    cross_modal_start: float = 0.25,
    cross_modal_end: float = 0.65,
    joint_branch_start: float = 0.40,
    joint_branch_end: float = 0.80,
    temperature: float = 0.07,
    neg_left_joint: torch.Tensor = None,
    neg_right_joint: torch.Tensor = None,
    al_neg_left_joint: torch.Tensor = None,
    al_neg_right_joint: torch.Tensor = None,
    margin: float = 0.2,
    hard_negative_weight: float = 1.0,
    al_negative_weight: float = 1.0,
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
            "struct_loss": struct_loss,
            "sem_loss": zero,
            "neg_loss": zero,
            "al_neg_loss": zero,
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

    neg_loss = margin_based_negative_loss(
        pos_left=left_outputs["z_joint"],
        pos_right=right_outputs["z_joint"],
        neg_left=neg_left_joint,
        neg_right=neg_right_joint,
        margin=margin,
        hard_weight=hard_negative_weight,
    )

    al_neg_loss = explicit_negative_pair_loss(
        neg_left=al_neg_left_joint,
        neg_right=al_neg_right_joint,
        margin=margin,
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
        lambda_topology * topology_loss
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
        "struct_loss": struct_reg,
        "sem_loss": sem_reg,
        "neg_loss": neg_loss,
        "al_neg_loss": al_neg_loss,
        "topology_loss": topology_loss,
    }
