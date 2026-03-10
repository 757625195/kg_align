import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class LightweightGNNEncoder(nn.Module):
    """
    轻量化多跳 GNN 编码器
    - 使用 GraphSAGE 做邻域聚合
    - 支持多层传播
    - 通过残差 + 层归一化提升稳定性
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        assert num_layers >= 1

        self.num_layers = num_layers
        self.dropout = dropout

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            self.convs.append(SAGEConv(dims[i], dims[i + 1]))
            if i != num_layers - 1:
                self.norms.append(nn.LayerNorm(dims[i + 1]))

        self.res_proj = None
        if in_dim != out_dim:
            self.res_proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: [num_nodes, in_dim]
        edge_index: [2, num_edges]
        """
        h0 = x
        h = x

        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)

            if i != self.num_layers - 1:
                h = self.norms[i](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        residual = h0 if self.res_proj is None else self.res_proj(h0)
        h = h + residual
        h = F.normalize(h, p=2, dim=-1)
        return h