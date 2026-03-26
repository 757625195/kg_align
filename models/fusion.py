import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalFusion(nn.Module):
    """
    轻量稳定版跨模态增强模块

    设计目标:
    1) 先让结构和语义彼此增强，而不是直接重写 joint 表示
    2) 使用跨模态互注意力，让语义直接参与结构邻居选择
    3) 将增强后的结构/语义向量拼接后，再通过两层 MLP 做最终融合
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

        self.sem_query = nn.Linear(dim, dim, bias=False)
        self.struct_query = nn.Linear(dim, dim, bias=False)
        self.struct_key = nn.Linear(dim, dim, bias=False)
        self.struct_value = nn.Linear(dim, dim, bias=False)
        self.sem_key = nn.Linear(dim, dim, bias=False)
        self.neighbor_sem_gate = nn.Linear(dim * 3, 1)

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
        self.conf_gate = nn.Linear(dim * 3, 1)
        self.joint_mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(
        self,
        s: torch.Tensor,
        t_self: torch.Tensor,
        t_nei: torch.Tensor,
        nei_mask: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor:
        # Cross-modal attention: the semantic branch queries structural
        # neighbors, so neighbor importance is conditioned on meaning rather
        # than structure alone.
        sem_q = self.sem_query(s).unsqueeze(1)  # [B, 1, D]
        nei_k = self.struct_key(t_nei)          # [B, K, D]
        nei_v = self.struct_value(t_nei)        # [B, K, D]
        attn_scores = (nei_k * sem_q).sum(dim=-1) / (self.dim ** 0.5)  # [B, K]

        # Semantic gate on each neighbor: keep neighbors that are more aligned
        # with the semantic branch and suppress semantically inconsistent ones.
        s_expand = s.unsqueeze(1).expand_as(t_nei)
        neighbor_sem_gate = torch.sigmoid(
            self.neighbor_sem_gate(torch.cat([t_nei, s_expand, t_nei - s_expand], dim=-1))
        ).squeeze(-1)

        gated_scores = attn_scores + torch.log(neighbor_sem_gate.clamp(min=1e-6))
        gated_scores = gated_scores.masked_fill(nei_mask == 0, -1e9)
        attn_weights = F.softmax(gated_scores, dim=1) * nei_mask.float()
        attn_weights = attn_weights / attn_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        neigh_attn = (attn_weights.unsqueeze(-1) * nei_v).sum(dim=1)

        has_neighbor = (nei_mask.sum(dim=1, keepdim=True) > 0).float()
        struct_context = has_neighbor * (0.5 * (t_self + neigh_attn)) + (1.0 - has_neighbor) * t_self

        # Reverse direction of the mutual attention: structural context decides
        # how strongly it should influence the semantic branch.
        struct_q = self.struct_query(struct_context)
        sem_k = self.sem_key(s)
        struct_to_sem_weight = torch.sigmoid(
            (struct_q * sem_k).sum(dim=-1, keepdim=True) / (self.dim ** 0.5)
        )
        struct_influence = struct_to_sem_weight * struct_context
        sem_influence = struct_to_sem_weight * s

        sem_input = torch.cat([s, struct_influence], dim=-1)
        struct_input = torch.cat([t_self, sem_influence], dim=-1)

        sem_delta = torch.tanh(self.struct_to_sem(sem_input))
        struct_delta = torch.tanh(self.sem_to_struct(struct_input))

        sem_gate = torch.sigmoid(self.sem_gate(torch.cat([s, struct_context, s - struct_context], dim=-1)))
        struct_gate = torch.sigmoid(self.struct_gate(torch.cat([t_self, s, t_self - s], dim=-1)))

        # Confidence gate: trust enhancement more when structure/semantic agree.
        conf = torch.sigmoid(self.conf_gate(torch.cat([s, struct_context, s - struct_context], dim=-1)))
        # Margin-based confidence: if semantic is closer to self-structure than neighbors, boost confidence.
        nei_sim = (t_nei * s_expand).sum(dim=-1).masked_fill(nei_mask == 0, -1e9)
        hard_nei_sim = nei_sim.max(dim=1).values
        self_sim = (s * t_self).sum(dim=-1)
        margin_conf = torch.sigmoid((self_sim - hard_nei_sim).unsqueeze(-1) * 5.0)
        conf = conf * margin_conf
        ratio = self.residual_ratio * conf
        s_enhanced = F.normalize(s + ratio * sem_gate * sem_delta, p=2, dim=-1)
        t_enhanced = F.normalize(t_self + ratio * struct_gate * struct_delta, p=2, dim=-1)

        # Guidance-aligned final fusion:
        # concatenate the enhanced semantic/structural vectors, then use a
        # two-layer MLP to produce the joint alignment embedding.
        joint_input = torch.cat([s_enhanced, t_enhanced], dim=-1)
        joint_delta = torch.tanh(self.joint_mlp(joint_input))
        base_joint = 0.5 * (s_enhanced + t_enhanced)
        z_joint = F.normalize(base_joint + conf * joint_delta, p=2, dim=-1)

        if return_components:
            semantic_attention = attn_weights
            structural_gate = neighbor_sem_gate * nei_mask.float()
            return {
                "z_joint": z_joint,
                "z_sem_enhanced": s_enhanced,
                "z_struct_enhanced": t_enhanced,
                "semantic_attention": semantic_attention,
                "structural_gate": structural_gate,
                "confidence_gate": conf,
            }

        return z_joint
