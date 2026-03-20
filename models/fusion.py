import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalFusion(nn.Module):
    """
    轻量稳定版跨模态增强模块

    设计目标:
    1) 先让结构和语义彼此增强，而不是直接重写 joint 表示
    2) 使用稳定的邻居均值上下文，避免随机/局部注意力带来的高方差
    3) 最终只做轻量门控融合，避免重 MLP 偏移基础表示
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        residual_ratio: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.residual_ratio = residual_ratio

        self.struct_to_sem = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.sem_to_struct = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

        self.sem_gate = nn.Linear(dim * 3, dim)
        self.struct_gate = nn.Linear(dim * 3, dim)
        self.joint_gate = nn.Linear(dim * 3, dim)

    def forward(
        self,
        s: torch.Tensor,
        t_self: torch.Tensor,
        t_nei: torch.Tensor,
        nei_mask: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor:
        mask = nei_mask.unsqueeze(-1).float()
        neigh_sum = (t_nei * mask).sum(dim=1)
        neigh_denom = mask.sum(dim=1).clamp(min=1e-6)
        neigh_mean = neigh_sum / neigh_denom

        has_neighbor = (nei_mask.sum(dim=1, keepdim=True) > 0).float()
        struct_context = has_neighbor * (0.5 * (t_self + neigh_mean)) + (1.0 - has_neighbor) * t_self

        sem_input = torch.cat([s, struct_context], dim=-1)
        struct_input = torch.cat([t_self, s], dim=-1)

        sem_delta = torch.tanh(self.struct_to_sem(sem_input))
        struct_delta = torch.tanh(self.sem_to_struct(struct_input))

        sem_gate = torch.sigmoid(self.sem_gate(torch.cat([s, struct_context, s - struct_context], dim=-1)))
        struct_gate = torch.sigmoid(self.struct_gate(torch.cat([t_self, s, t_self - s], dim=-1)))

        ratio = self.residual_ratio
        s_enhanced = F.normalize(s + ratio * sem_gate * sem_delta, p=2, dim=-1)
        t_enhanced = F.normalize(t_self + ratio * struct_gate * struct_delta, p=2, dim=-1)

        joint_alpha = torch.sigmoid(
            self.joint_gate(torch.cat([s_enhanced, t_enhanced, s_enhanced - t_enhanced], dim=-1))
        )
        z_joint = F.normalize(joint_alpha * s_enhanced + (1.0 - joint_alpha) * t_enhanced, p=2, dim=-1)

        if return_components:
            semantic_attention = nei_mask.float()
            structural_gate = joint_alpha.mean(dim=-1, keepdim=True).expand_as(nei_mask.float())
            return {
                "z_joint": z_joint,
                "z_sem_enhanced": s_enhanced,
                "z_struct_enhanced": t_enhanced,
                "semantic_attention": semantic_attention,
                "structural_gate": structural_gate,
            }

        return z_joint
