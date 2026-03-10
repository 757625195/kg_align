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

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def pairwise_similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        x1: [N1, D]
        x2: [N2, D]
        return: [N1, N2]
        """
        x1 = F.normalize(x1, p=2, dim=-1)
        x2 = F.normalize(x2, p=2, dim=-1)
        return torch.matmul(x1, x2.transpose(0, 1))

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
