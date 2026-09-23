import torch
import torch.nn.functional as F


def info_nce_loss(
    x1: torch.Tensor,
    x2: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Bidirectional in-batch InfoNCE for aligned entity pairs."""
    x1 = F.normalize(x1, p=2, dim=-1)
    x2 = F.normalize(x2, p=2, dim=-1)

    logits = x1 @ x2.t() / temperature
    labels = torch.arange(x1.size(0), device=x1.device)
    loss_12 = F.cross_entropy(logits, labels)
    loss_21 = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_12 + loss_21)


def total_loss(
    left_outputs: dict,
    right_outputs: dict,
    temperature: float = 0.07,
    structure_loss_weight: float = 0.0,
) -> dict:
    """Compute joint InfoNCE with optional direct structural supervision."""
    align = info_nce_loss(
        left_outputs["z_joint"],
        right_outputs["z_joint"],
        temperature=temperature,
    )
    if structure_loss_weight > 0.0:
        structure_align = info_nce_loss(
            left_outputs["z_struct"],
            right_outputs["z_struct"],
            temperature=temperature,
        )
    else:
        structure_align = align.new_zeros(())
    return {
        "loss": align + structure_loss_weight * structure_align,
        "align_loss": align,
        "structure_align_loss": structure_align,
    }
