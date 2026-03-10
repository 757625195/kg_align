import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalFusion(nn.Module):
    """
    邻居感知跨模态融合模块

    输入:
        s:        [B, D]     当前实体语义表示
        t_self:   [B, D]     当前实体自身结构表示
        t_nei:    [B, K, D]  当前实体邻居结构表示
        nei_mask: [B, K]     邻居有效mask, 1表示有效，0表示padding

    输出:
        u:        [B, D]     融合后的联合表示
    """

    def __init__(self, dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dim = dim

        # structure -> semantic attention
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # semantic -> structure gating
        self.gate_proj = nn.Linear(dim, dim, bias=False)

        # self structural residual transform
        self.self_proj = nn.Linear(dim, dim)

        # final fusion
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        s: torch.Tensor,         # [B, D]
        t_self: torch.Tensor,    # [B, D]
        t_nei: torch.Tensor,     # [B, K, D]
        nei_mask: torch.Tensor,  # [B, K]
        return_components: bool = False,
    ) -> torch.Tensor:
        B, K, D = t_nei.shape

        # =========================
        # 1) structure -> semantic
        # =========================
        q = self.q_proj(s).unsqueeze(1)          # [B, 1, D]
        k = self.k_proj(t_nei)                   # [B, K, D]
        v = self.v_proj(t_nei)                   # [B, K, D]

        attn_scores = torch.matmul(q, k.transpose(1, 2)).squeeze(1) / math.sqrt(D)  # [B, K]
        attn_scores = attn_scores.masked_fill(nei_mask == 0, -1e9)

        alpha = F.softmax(attn_scores, dim=-1)   # [B, K]
        alpha = alpha * nei_mask.float()
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        delta_s = torch.sum(alpha.unsqueeze(-1) * v, dim=1)  # [B, D]
        s_plus = s + delta_s                                 # [B, D]

        # =========================
        # 2) semantic -> structure
        # =========================
        gate_logits = torch.sum(
            s.unsqueeze(1) * self.gate_proj(t_nei),
            dim=-1
        )  # [B, K]

        gate = torch.sigmoid(gate_logits) * nei_mask.float()
        gate = gate / gate.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        t_nei_agg = torch.sum(gate.unsqueeze(-1) * t_nei, dim=1)  # [B, D]
        t_plus = self.self_proj(t_self) + t_nei_agg               # [B, D]

        # =========================
        # 3) concat + MLP
        # =========================
        u = torch.cat([s_plus, t_plus], dim=-1)   # [B, 2D]
        u = self.mlp(u)
        u = self.norm(u)
        u = F.normalize(u, p=2, dim=-1)

        if return_components:
            return {
                "z_joint": u,
                "z_sem_enhanced": F.normalize(s_plus, p=2, dim=-1),
                "z_struct_enhanced": F.normalize(t_plus, p=2, dim=-1),
                "semantic_attention": alpha,
                "structural_gate": gate,
            }

        return u
