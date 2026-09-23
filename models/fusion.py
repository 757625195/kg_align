import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_entmax15(
    scores: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
    iterations: int = 24,
) -> torch.Tensor:
    """Compute masked 1.5-entmax with a differentiable threshold search."""
    valid = mask.to(dtype=torch.bool)
    has_valid = valid.any(dim=dim, keepdim=True)
    scaled = (scores / 2.0).masked_fill(~valid, -1e4)
    maximum = scaled.max(dim=dim, keepdim=True).values
    maximum = torch.where(has_valid, maximum, torch.zeros_like(maximum))

    lower = maximum - 1.0
    upper = maximum
    for _ in range(iterations):
        threshold = (lower + upper) / 2.0
        probability_mass = (
            torch.relu(scaled - threshold).square() * valid.to(scores.dtype)
        ).sum(dim=dim, keepdim=True)
        lower = torch.where(probability_mass > 1.0, threshold, lower)
        upper = torch.where(probability_mass > 1.0, upper, threshold)

    threshold = (lower + upper) / 2.0
    weights = torch.relu(scaled - threshold).square() * valid.to(scores.dtype)
    normalizer = weights.sum(dim=dim, keepdim=True)
    return torch.where(
        has_valid,
        weights / normalizer.clamp(min=1e-12),
        torch.zeros_like(weights),
    )


class LateConcatFusion(nn.Module):
    """Late-fusion baseline: merge independently encoded branches at the output."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.projection = nn.Sequential(
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
        return_components: bool = False,
    ) -> torch.Tensor:
        s_norm = F.normalize(s, p=2, dim=-1)
        t_norm = F.normalize(t_self, p=2, dim=-1)
        z_joint = F.normalize(
            self.projection(torch.cat([t_norm, s_norm], dim=-1)),
            p=2,
            dim=-1,
        )

        if return_components:
            zeros = torch.zeros(
                z_joint.size(0),
                t_nei.size(1),
                device=z_joint.device,
                dtype=z_joint.dtype,
            )
            return {
                "z_joint": z_joint,
                "z_sem_enhanced": s_norm,
                "z_struct_enhanced": t_norm,
                "semantic_attention": zeros,
                "structural_gate": zeros,
            }

        return z_joint


class ParameterMatchedLateFusion(nn.Module):
    """Late fusion with the same trainable parameter count as early fusion."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.feature_scale = nn.Parameter(torch.zeros(dim))
        self.residual_mix_logit = nn.Parameter(torch.tensor(4.0))

    def forward(
        self,
        s: torch.Tensor,
        t_self: torch.Tensor,
        t_nei: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor:
        s_norm = F.normalize(s, p=2, dim=-1)
        t_norm = F.normalize(t_self, p=2, dim=-1)
        transformed = self.projection(torch.cat([t_norm, s_norm], dim=-1))
        calibrated = transformed * (1.0 + 0.1 * torch.tanh(self.feature_scale))
        base = 0.5 * (t_norm + s_norm)
        mix = torch.sigmoid(self.residual_mix_logit)
        z_joint = F.normalize(mix * calibrated + (1.0 - mix) * base, p=2, dim=-1)

        if return_components:
            zeros = torch.zeros(
                z_joint.size(0),
                t_nei.size(1),
                device=z_joint.device,
                dtype=z_joint.dtype,
            )
            return {
                "z_joint": z_joint,
                "z_sem_enhanced": s_norm,
                "z_struct_enhanced": t_norm,
                "semantic_attention": zeros,
                "structural_gate": zeros,
            }

        return z_joint


class CrossModalFusion(nn.Module):
    """
    Semantic-conditioned structural-context fusion:
    - use semantics to select structural neighbors
    - filter neighbors with a semantic gate
    - build a structural context
    - directly fuse semantics and structural context with a simple gate
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        residual_ratio: float = 0.1,
        neighbor_query_mode: str = "semantic",
        neighbor_gate_mode: str = "semantic",
        neighbor_query_semantic_weight: float = 0.5,
        neighbor_attention: str = "softmax",
        neighbor_attention_temperature: float = 1.0,
        reciprocal_add: bool = False,
        reciprocal_residual_add: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.residual_ratio = min(max(float(residual_ratio), 0.0), 1.0)
        self.reciprocal_add = bool(reciprocal_add)
        self.reciprocal_residual_add = bool(reciprocal_residual_add)
        self.neighbor_query_mode = neighbor_query_mode.strip().lower()
        if self.neighbor_query_mode not in {"semantic", "structural", "hybrid"}:
            raise ValueError(
                "neighbor_query_mode must be 'semantic', 'structural', or 'hybrid', got "
                f"{neighbor_query_mode!r}"
            )
        self.neighbor_gate_mode = neighbor_gate_mode.strip().lower()
        if self.neighbor_gate_mode not in {"semantic", "structural", "hybrid"}:
            raise ValueError(
                "neighbor_gate_mode must be 'semantic', 'structural', or 'hybrid', got "
                f"{neighbor_gate_mode!r}"
            )
        self.neighbor_query_semantic_weight = float(neighbor_query_semantic_weight)
        if not 0.0 <= self.neighbor_query_semantic_weight <= 1.0:
            raise ValueError("neighbor_query_semantic_weight must be within [0, 1]")
        self.neighbor_attention = neighbor_attention.strip().lower()
        if self.neighbor_attention not in {"softmax", "entmax15"}:
            raise ValueError(
                "neighbor_attention must be 'softmax' or 'entmax15', got "
                f"{neighbor_attention!r}"
            )
        self.neighbor_attention_temperature = float(neighbor_attention_temperature)
        if self.neighbor_attention_temperature <= 0.0:
            raise ValueError("neighbor_attention_temperature must be positive")

        self.sem_query = nn.Linear(dim, dim, bias=False)
        self.struct_key = nn.Linear(dim, dim, bias=False)
        self.struct_value = nn.Linear(dim, dim, bias=False)
        self.neighbor_sem_gate = nn.Linear(dim * 4, 1)
        self.struct_context_gate = nn.Linear(dim * 5, dim)
        self.joint_gate = nn.Linear(dim * 4, dim)

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
        # Retain these legacy checkpoint keys without treating the unused paths
        # as trainable capacity in controlled comparisons.
        self.struct_to_sem.requires_grad_(False)
        self.sem_to_struct.requires_grad_(False)

    @staticmethod
    def _pair_features(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, y, x * y, torch.abs(x - y)], dim=-1)

    def _conditioning_source(
        self,
        mode: str,
        semantic: torch.Tensor,
        structural: torch.Tensor,
    ) -> torch.Tensor:
        if mode == "semantic":
            return semantic
        if mode == "structural":
            return structural
        beta = self.neighbor_query_semantic_weight
        return beta * semantic + (1.0 - beta) * structural

    def forward(
        self,
        s: torch.Tensor,
        t_self: torch.Tensor,
        t_nei: torch.Tensor,
        nei_mask: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor:
        query_source = self._conditioning_source(
            self.neighbor_query_mode,
            s,
            t_self,
        )
        gate_source = self._conditioning_source(
            self.neighbor_gate_mode,
            s,
            t_self,
        )
        sem_q = self.sem_query(query_source).unsqueeze(1)
        nei_k = self.struct_key(t_nei)
        nei_v = self.struct_value(t_nei)
        attn_scores = (nei_k * sem_q).sum(dim=-1) / (self.dim ** 0.5)

        gate_expand = gate_source.unsqueeze(1).expand_as(t_nei)
        neighbor_sem_gate = torch.sigmoid(
            self.neighbor_sem_gate(self._pair_features(t_nei, gate_expand))
        ).squeeze(-1)

        gated_scores = attn_scores + torch.log(neighbor_sem_gate.clamp(min=1e-6))
        if self.neighbor_attention == "entmax15":
            attn_weights = masked_entmax15(
                gated_scores / self.neighbor_attention_temperature,
                nei_mask,
                dim=1,
            )
        else:
            gated_scores = (
                gated_scores / self.neighbor_attention_temperature
            ).masked_fill(nei_mask == 0, -1e9)
            attn_weights = F.softmax(gated_scores, dim=1) * nei_mask.float()
            attn_weights = attn_weights / attn_weights.sum(
                dim=1, keepdim=True
            ).clamp(min=1e-6)
        neigh_attn = (attn_weights.unsqueeze(-1) * nei_v).sum(dim=1)

        has_neighbor = (nei_mask.sum(dim=1, keepdim=True) > 0).float()
        # Keep the self-versus-neighbor context gate fixed across the 2x2
        # query/gate experiment so only the two tested neighbor-scoring factors vary.
        context_gate_input = torch.cat(
            [self._pair_features(t_self, neigh_attn), s],
            dim=-1,
        )
        context_gate = torch.sigmoid(self.struct_context_gate(context_gate_input))
        gated_struct_context = context_gate * t_self + (1.0 - context_gate) * neigh_attn
        struct_context = has_neighbor * gated_struct_context + (1.0 - has_neighbor) * t_self

        s_initial = F.normalize(s, p=2, dim=-1)
        t_initial = F.normalize(t_self, p=2, dim=-1)
        t_enhanced = F.normalize(struct_context, p=2, dim=-1)

        if self.reciprocal_add or self.reciprocal_residual_add:
            # Parallel reciprocal conditioning: semantics forms the structural
            # context above, while the initial structure gates initial semantics.
            semantic_gate = torch.sigmoid(
                self.joint_gate(self._pair_features(s_initial, t_initial))
            )
            s_enhanced = F.normalize(
                semantic_gate * s_initial + (1.0 - semantic_gate) * t_initial,
                p=2,
                dim=-1,
            )
            if self.reciprocal_residual_add:
                z_joint = F.normalize(
                    s_initial + t_initial + s_enhanced + t_enhanced,
                    p=2,
                    dim=-1,
                )
            else:
                z_joint = F.normalize(s_enhanced + t_enhanced, p=2, dim=-1)
        else:
            s_enhanced = s_initial
            joint_features = self._pair_features(s_enhanced, t_enhanced)
            joint_mix_gate = torch.sigmoid(self.joint_gate(joint_features))
            gated_joint = F.normalize(
                joint_mix_gate * t_enhanced + (1.0 - joint_mix_gate) * s_enhanced,
                p=2,
                dim=-1,
            )
            if self.residual_ratio > 0.0:
                base_joint = F.normalize(
                    0.5 * (t_enhanced + s_enhanced),
                    p=2,
                    dim=-1,
                )
                z_joint = F.normalize(
                    (1.0 - self.residual_ratio) * gated_joint
                    + self.residual_ratio * base_joint,
                    p=2,
                    dim=-1,
                )
            else:
                z_joint = gated_joint

        if return_components:
            return {
                "z_joint": z_joint,
                "z_sem_enhanced": s_enhanced,
                "z_struct_enhanced": t_enhanced,
                "semantic_attention": attn_weights,
                "structural_gate": neighbor_sem_gate * nei_mask.float(),
                "struct_context_gate": context_gate,
                "neighbor_support_size": (attn_weights > 1e-8).sum(dim=1),
                "neighbor_valid_count": nei_mask.sum(dim=1),
            }

        return z_joint
