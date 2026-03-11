import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


def mean_neighbor_aggregate(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Mean aggregation over incoming neighbors with self-loop preservation.
    This helper is used both by the depthwise-separable graph layer and by the
    explicit 2/3-hop topology matching objective.
    """
    src, dst = edge_index
    out = x.new_zeros(x.size(0), x.size(1))
    out.index_add_(0, dst, x[src])

    deg = x.new_zeros(x.size(0))
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
    deg = deg.clamp(min=1.0).unsqueeze(-1)

    neigh_mean = out / deg
    return 0.5 * (neigh_mean + x)


class DepthwiseSeparableGraphConv(nn.Module):
    """
    Graph analogue of depthwise-separable convolution:
    1) parameter-free neighborhood mean aggregation
    2) per-channel scaling (depthwise)
    3) pointwise projection
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.depthwise_weight = nn.Parameter(torch.ones(in_dim))
        self.pointwise = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = mean_neighbor_aggregate(x, edge_index)
        h = h * self.depthwise_weight
        return self.pointwise(h)


class LightweightGNNEncoder(nn.Module):
    """
    轻量化多跳 GNN 编码器
    - 默认使用 GraphSAGE 做稀疏邻域聚合
    - 可选跨层参数共享，减少卷积核参数规模
    - 可选深度可分离图卷积，进一步降低参数和计算成本
    - 通过残差 + 层归一化提升稳定性
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        share_parameters: bool = False,
        use_depthwise_separable: bool = False,
    ):
        super().__init__()
        assert num_layers >= 1

        self.num_layers = num_layers
        self.dropout = dropout
        self.share_parameters = share_parameters
        self.use_depthwise_separable = use_depthwise_separable

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layer_dims = list(zip(dims[:-1], dims[1:]))
        self.norms = nn.ModuleList()

        def make_conv(src_dim: int, dst_dim: int):
            if use_depthwise_separable:
                return DepthwiseSeparableGraphConv(src_dim, dst_dim)
            return SAGEConv(src_dim, dst_dim)

        if share_parameters:
            unique_pairs = []
            for pair in self.layer_dims:
                if pair not in unique_pairs:
                    unique_pairs.append(pair)
            self.shared_convs = nn.ModuleDict({
                f"{src}_{dst}": make_conv(src, dst)
                for src, dst in unique_pairs
            })
        else:
            self.convs = nn.ModuleList([
                make_conv(src, dst)
                for src, dst in self.layer_dims
            ])

        for i in range(num_layers - 1):
            self.norms.append(nn.LayerNorm(dims[i + 1]))

        self.res_proj = None
        if in_dim != out_dim:
            self.res_proj = nn.Linear(in_dim, out_dim)

    def _get_conv(self, layer_idx: int):
        if not self.share_parameters:
            return self.convs[layer_idx]
        src_dim, dst_dim = self.layer_dims[layer_idx]
        return self.shared_convs[f"{src_dim}_{dst_dim}"]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: [num_nodes, in_dim]
        edge_index: [2, num_edges]

        实现要点:
        1) 每层卷积都在稀疏边上做一次邻域聚合
        2) 多层堆叠后，节点表示会逐步吸收多跳邻居信息
        3) 最后一层输出与初始输入做残差相加，保留节点自身身份
        """
        h0 = x
        h = x

        for i in range(self.num_layers):
            conv = self._get_conv(i)
            h = conv(h, edge_index)

            if i != self.num_layers - 1:
                h = self.norms[i](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        residual = h0 if self.res_proj is None else self.res_proj(h0)
        h = h + residual
        h = F.normalize(h, p=2, dim=-1)
        return h

    def collect_k_hop_features(
        self,
        z_struct_all: torch.Tensor,
        edge_index: torch.Tensor,
        max_hop: int = 3,
    ) -> dict:
        """
        Build explicit 1/2/3-hop topology summaries from the learned structure
        space. This is used for the topology matching objective.
        """
        hop_features = {}
        h = z_struct_all
        for hop in range(1, max_hop + 1):
            h = mean_neighbor_aggregate(h, edge_index)
            hop_features[hop] = F.normalize(h, p=2, dim=-1)
        return hop_features
