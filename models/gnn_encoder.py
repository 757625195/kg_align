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


class RelationAwareGraphConv(nn.Module):
    """
    Lightweight relation-aware message passing:
    - build messages from source node features plus relation embeddings
    - mean aggregate on incoming edges
    - adaptively mix self and neighbor information with a learned gate
    """

    def __init__(self, in_dim: int, out_dim: int, num_relations: int):
        super().__init__()
        self.rel_emb = nn.Embedding(num_relations, in_dim)
        self.src_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.rel_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.self_proj = nn.Linear(in_dim, out_dim)
        self.msg_norm = nn.LayerNorm(out_dim)
        self.mix_gate = nn.Linear(out_dim * 2, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        rel = self.rel_emb(edge_type)
        msg = self.src_proj(x[src]) + self.rel_proj(rel)
        msg = self.msg_norm(msg)

        neigh = x.new_zeros(x.size(0), msg.size(-1))
        neigh.index_add_(0, dst, msg)

        deg = x.new_zeros(x.size(0))
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        neigh = neigh / deg.clamp(min=1.0).unsqueeze(-1)

        self_feat = self.self_proj(x)
        gate = torch.sigmoid(self.mix_gate(torch.cat([self_feat, neigh], dim=-1)))
        return self_feat + gate * neigh


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


class RelationAwareGNNEncoder(nn.Module):
    """
    Relation-aware structural encoder.
    Compared with plain GraphSAGE, this encoder distinguishes edges by relation
    type and injects relation embeddings directly into the propagated messages.
    In addition, it keeps multiple hop-depth representations and fuses them with
    a node-wise layer selector instead of relying only on the deepest layer.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_relations: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        share_parameters: bool = False,
        use_layer_fusion: bool = True,
    ):
        super().__init__()
        assert num_layers >= 1

        self.num_layers = num_layers
        self.dropout = dropout
        self.share_parameters = share_parameters
        self.use_layer_fusion = use_layer_fusion and num_layers > 1
        self.out_dim = out_dim

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layer_dims = list(zip(dims[:-1], dims[1:]))
        self.norms = nn.ModuleList()

        def make_conv(src_dim: int, dst_dim: int):
            return RelationAwareGraphConv(
                in_dim=src_dim,
                out_dim=dst_dim,
                num_relations=num_relations,
            )

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

        self.input_proj = None
        if in_dim != out_dim:
            self.input_proj = nn.Linear(in_dim, out_dim)

        self.layer_out_projs = nn.ModuleList()
        for _, dst_dim in self.layer_dims:
            if dst_dim == out_dim:
                self.layer_out_projs.append(nn.Identity())
            else:
                self.layer_out_projs.append(nn.Linear(dst_dim, out_dim))

        self.res_proj = None
        if in_dim != out_dim:
            self.res_proj = nn.Linear(in_dim, out_dim)

        self.output_norm = nn.LayerNorm(out_dim)
        if self.use_layer_fusion:
            self.layer_score = nn.Sequential(
                nn.Linear(out_dim * 2, out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim, 1),
            )
        else:
            self.layer_score = None

    def _get_conv(self, layer_idx: int):
        if not self.share_parameters:
            return self.convs[layer_idx]
        src_dim, dst_dim = self.layer_dims[layer_idx]
        return self.shared_convs[f"{src_dim}_{dst_dim}"]

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        h0 = x
        h = x
        residual = h0 if self.res_proj is None else self.res_proj(h0)
        layer_states = []
        base_state = residual if self.input_proj is None else self.input_proj(h0)
        layer_states.append(base_state)

        for i in range(self.num_layers):
            conv = self._get_conv(i)
            h = conv(h, edge_index, edge_type)
            layer_states.append(self.layer_out_projs[i](h))

            if i != self.num_layers - 1:
                h = self.norms[i](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        if self.use_layer_fusion:
            stacked = torch.stack(layer_states, dim=1)  # [N, L+1, D]
            global_context = stacked.mean(dim=1, keepdim=True).expand_as(stacked)
            layer_logits = self.layer_score(torch.cat([stacked, global_context], dim=-1)).squeeze(-1)
            layer_weights = torch.softmax(layer_logits, dim=1).unsqueeze(-1)
            h = (layer_weights * stacked).sum(dim=1)
        else:
            h = layer_states[-1] + residual

        h = self.output_norm(h)
        h = F.normalize(h, p=2, dim=-1)
        return h

    def collect_k_hop_features(
        self,
        z_struct_all: torch.Tensor,
        edge_index: torch.Tensor,
        max_hop: int = 3,
    ) -> dict:
        hop_features = {}
        h = z_struct_all
        for hop in range(1, max_hop + 1):
            h = mean_neighbor_aggregate(h, edge_index)
            hop_features[hop] = F.normalize(h, p=2, dim=-1)
        return hop_features
