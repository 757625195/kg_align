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


def structure_consistency_loss(
    z_struct: torch.Tensor,
    z_joint: torch.Tensor
) -> torch.Tensor:
    z_struct = F.normalize(z_struct, p=2, dim=-1)
    z_joint = F.normalize(z_joint, p=2, dim=-1)
    return 1.0 - (z_struct * z_joint).sum(dim=-1).mean()


def semantic_consistency_loss(
    z_sem: torch.Tensor,
    z_joint: torch.Tensor
) -> torch.Tensor:
    z_sem = F.normalize(z_sem, p=2, dim=-1)
    z_joint = F.normalize(z_joint, p=2, dim=-1)
    return 1.0 - (z_sem * z_joint).sum(dim=-1).mean()


def branch_consistency_loss(
    z_struct_shared: torch.Tensor,
    z_sem_shared: torch.Tensor,
) -> torch.Tensor:
    z_struct_shared = F.normalize(z_struct_shared, p=2, dim=-1)
    z_sem_shared = F.normalize(z_sem_shared, p=2, dim=-1)
    return 1.0 - (z_struct_shared * z_sem_shared).sum(dim=-1).mean()


def structural_self_supervised_loss(z_struct: torch.Tensor) -> torch.Tensor:
    """
    warmup 阶段占位版结构自监督
    目的：避免结构表征塌缩
    """
    z = F.normalize(z_struct, p=2, dim=-1)
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return F.relu(1.0 - std).mean()


def total_loss(
    left_outputs: dict,
    right_outputs: dict,
    lambda_struct: float = 0.2,
    lambda_sem: float = 0.2,
    lambda_branch_align: float = 0.5,
    lambda_cross_modal: float = 0.2,
    lambda_neg: float = 0.2,
    temperature: float = 0.07,
    neg_left_joint: torch.Tensor = None,
    neg_right_joint: torch.Tensor = None,
    margin: float = 0.2,
    hard_negative_weight: float = 1.0,
    warmup_mode: bool = False,
):
    """
    left_outputs / right_outputs:
    {
        "z_struct": ...,
        "z_sem": ...,
        "z_joint": ...
    }
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
            "struct_loss": struct_loss,
            "sem_loss": zero,
            "branch_align_loss": zero,
            "cross_modal_loss": zero,
            "neg_loss": zero,
        }

    # =========================
    # Stage 2: joint training
    # =========================
    align = info_nce_loss(
        left_outputs["z_joint"],
        right_outputs["z_joint"],
        temperature=temperature
    )

    branch_align = 0.5 * (
        info_nce_loss(
            left_outputs["z_struct_shared"],
            right_outputs["z_struct_shared"],
            temperature=temperature,
        ) +
        info_nce_loss(
            left_outputs["z_sem_shared"],
            right_outputs["z_sem_shared"],
            temperature=temperature,
        )
    )

    struct_reg = 0.5 * (
        structure_consistency_loss(left_outputs["z_struct_shared"], left_outputs["z_joint"]) +
        structure_consistency_loss(right_outputs["z_struct_shared"], right_outputs["z_joint"])
    )

    sem_reg = 0.5 * (
        semantic_consistency_loss(left_outputs["z_sem_shared"], left_outputs["z_joint"]) +
        semantic_consistency_loss(right_outputs["z_sem_shared"], right_outputs["z_joint"])
    )

    cross_modal = 0.5 * (
        branch_consistency_loss(left_outputs["z_struct_shared"], left_outputs["z_sem_shared"]) +
        branch_consistency_loss(right_outputs["z_struct_shared"], right_outputs["z_sem_shared"])
    )

    neg_loss = margin_based_negative_loss(
        pos_left=left_outputs["z_joint"],
        pos_right=right_outputs["z_joint"],
        neg_left=neg_left_joint,
        neg_right=neg_right_joint,
        margin=margin,
        hard_weight=hard_negative_weight,
    )

    loss = (
        align +
        lambda_branch_align * branch_align +
        lambda_struct * struct_reg +
        lambda_sem * sem_reg +
        lambda_cross_modal * cross_modal +
        lambda_neg * neg_loss
    )

    return {
        "loss": loss,
        "align_loss": align,
        "struct_loss": struct_reg,
        "sem_loss": sem_reg,
        "branch_align_loss": branch_align,
        "cross_modal_loss": cross_modal,
        "neg_loss": neg_loss,
    }
