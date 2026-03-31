import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentHead(nn.Module):
    """
    对齐头：
    - 输出实体表示
    - 支持两图之间相似度矩阵计算
    两个图谱里所有实体的联合 embedding 拿出来，
    两两计算余弦相似度，形成对齐分数矩阵，并用温度缩放让训练更容易拉开正负样本差距。
    """

    def __init__(
        self,
        temperature: float = 0.07,
        csls_k: int = 0,
        csls_blend: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.csls_k = max(0, int(csls_k))
        self.csls_blend = float(csls_blend)

    @staticmethod
    def _cosine_similarity(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = F.normalize(x1, p=2, dim=-1)
        x2 = F.normalize(x2, p=2, dim=-1)
        return torch.matmul(x1, x2.transpose(0, 1))

    @staticmethod
    def _topk_mean(sim: torch.Tensor, k: int, dim: int) -> torch.Tensor:
        k = max(1, min(k, sim.size(dim)))
        return sim.topk(k=k, dim=dim).values.mean(dim=dim, keepdim=True)

    def _apply_csls(self, sim: torch.Tensor) -> torch.Tensor:
        if self.csls_k <= 0 or sim.size(0) <= 1 or sim.size(1) <= 1:
            return sim

        row_corr = self._topk_mean(sim, self.csls_k, dim=1)
        col_corr = self._topk_mean(sim, self.csls_k, dim=0)
        csls_sim = 2.0 * sim - row_corr - col_corr
        blend = min(max(self.csls_blend, 0.0), 1.0)
        return (1.0 - blend) * sim + blend * csls_sim

    def pairwise_similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        x1: [N1, D]
        x2: [N2, D]
        return: [N1, N2]
        """
        sim = self._cosine_similarity(x1, x2)
        return self._apply_csls(sim)

    def pair_confidence(self, sim: torch.Tensor) -> dict:
        probs = F.softmax(sim / self.temperature, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)

        top2 = torch.topk(sim, k=min(2, sim.size(1)), dim=-1).values
        if top2.size(1) == 1:
            margin = torch.ones_like(top2[:, 0])
        else:
            margin = top2[:, 0] - top2[:, 1]

        confidence = probs.max(dim=-1).values
        uncertainty = 0.5 * entropy + 0.5 * (1.0 - margin)

        return {
            "probabilities": probs,
            "confidence": confidence,
            "entropy": entropy,
            "margin": margin,
            "uncertainty": uncertainty,
        }

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.pairwise_similarity(x1, x2) / self.temperature
