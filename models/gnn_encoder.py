import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import softmax


def mean_neighbor_aggregate(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Mean aggregation over incoming neighbors with self-loop preservation.
    This helper is used by the depthwise-separable graph layer.
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
    Stable relation-aware message passing:
    - relation embeddings are injected directly into propagated messages
    - incoming neighbor evidence is mixed with an explicit self feature via a
      learned gate, preserving the behavior of the strongest prior model
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_relations: int,
        use_relation_types: bool = True,
    ):
        super().__init__()
        self.use_relation_types = use_relation_types
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
        if self.use_relation_types:
            rel = self.rel_emb(edge_type)
        else:
            # Keep the relation channel and parameter budget fixed while
            # removing edge-type identity from every propagated message.
            rel = self.rel_emb.weight.mean(dim=0, keepdim=True).expand(
                edge_type.numel(), -1
            )
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
        use_relation_types: bool = True,
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
                use_relation_types=use_relation_types,
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


class RDGCNStructuralEncoder(nn.Module):
    """Adapted RDGCN entity/relation dual-graph structural encoder."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_relations, num_layers=3, dropout=0.1):
        super().__init__()
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_proj = nn.Linear(in_dim, out_dim)
        self.relation_init = nn.Linear(out_dim * 2, out_dim)
        self.relation_key = nn.Linear(out_dim, out_dim, bias=False)
        self.relation_value = nn.Linear(out_dim, out_dim, bias=False)
        self.relation_query = nn.Linear(out_dim, out_dim, bias=False)
        self.entity_src = nn.ModuleList(nn.Linear(out_dim, out_dim, bias=False) for _ in range(num_layers))
        self.entity_rel = nn.ModuleList(nn.Linear(out_dim, out_dim, bias=False) for _ in range(num_layers))
        self.entity_self = nn.ModuleList(nn.Linear(out_dim, out_dim) for _ in range(num_layers))
        self.entity_gate = nn.ModuleList(nn.Linear(out_dim * 2, out_dim) for _ in range(num_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(out_dim) for _ in range(num_layers))
        self.output_norm = nn.LayerNorm(out_dim)
        self._dual_graph_cache = {}

    def _relation_means(self, h, edge_index, edge_type):
        src, dst = edge_index
        head_sum = h.new_zeros(self.num_relations, h.size(-1))
        tail_sum = h.new_zeros(self.num_relations, h.size(-1))
        count = h.new_zeros(self.num_relations)
        head_sum.index_add_(0, edge_type, h[src])
        tail_sum.index_add_(0, edge_type, h[dst])
        count.index_add_(0, edge_type, torch.ones_like(edge_type, dtype=h.dtype))
        denom = count.clamp(min=1.0).unsqueeze(-1)
        return self.relation_init(torch.cat([head_sum / denom, tail_sum / denom], dim=-1))

    def _dual_graph(self, edge_index, edge_type):
        key = (edge_index.data_ptr(), edge_type.data_ptr(), edge_index.size(1))
        cached = self._dual_graph_cache.get(key)
        if cached is not None:
            return cached
        node_relations = {}
        src_cpu, dst_cpu = edge_index.detach().cpu()
        rel_cpu = edge_type.detach().cpu()
        for src, dst, rel in zip(src_cpu.tolist(), dst_cpu.tolist(), rel_cpu.tolist()):
            node_relations.setdefault(src, set()).add(rel)
            node_relations.setdefault(dst, set()).add(rel)
        dual_edges = set()
        for relations in node_relations.values():
            for left in relations:
                dual_edges.add((left, left))
                for right in relations:
                    dual_edges.add((left, right))
        if not dual_edges:
            dual_edges = {(rel, rel) for rel in range(self.num_relations)}
        dual = torch.tensor(sorted(dual_edges), dtype=torch.long, device=edge_index.device).t()
        self._dual_graph_cache = {key: dual}
        return dual

    def _update_relations(self, relation_state, dual_edge_index):
        src, dst = dual_edge_index
        query = self.relation_query(relation_state[dst])
        key = self.relation_key(relation_state[src])
        score = (query * key).sum(dim=-1) / (relation_state.size(-1) ** 0.5)
        weight = softmax(score, dst, num_nodes=self.num_relations)
        message = weight.unsqueeze(-1) * self.relation_value(relation_state[src])
        aggregate = relation_state.new_zeros(relation_state.shape)
        aggregate.index_add_(0, dst, message)
        return F.normalize(relation_state + aggregate, p=2, dim=-1)

    def forward(self, x, edge_index, edge_type):
        h = self.input_proj(x)
        residual = h
        dual_edge_index = self._dual_graph(edge_index, edge_type)
        relation_state = self._relation_means(h, edge_index, edge_type)
        src, dst = edge_index
        for layer in range(self.num_layers):
            relation_state = self._update_relations(relation_state, dual_edge_index)
            message = self.entity_src[layer](h[src]) + self.entity_rel[layer](relation_state[edge_type])
            neighbor = h.new_zeros(h.shape)
            neighbor.index_add_(0, dst, message)
            degree = h.new_zeros(h.size(0))
            degree.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
            neighbor = neighbor / degree.clamp(min=1.0).unsqueeze(-1)
            self_state = self.entity_self[layer](h)
            gate = torch.sigmoid(self.entity_gate[layer](torch.cat([self_state, neighbor], dim=-1)))
            h = self.norms[layer](self_state + gate * neighbor)
            if layer + 1 < self.num_layers:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            relation_state = self._relation_means(h, edge_index, edge_type)
        return F.normalize(self.output_norm(h + residual), p=2, dim=-1)


class RREAStructuralEncoder(nn.Module):
    """RREA-style relation-reflection encoder adapted to the shared EA pipeline."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_relations, num_layers=3, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_proj = nn.Linear(in_dim, out_dim)
        self.relation_emb = nn.Embedding(num_relations, out_dim)
        self.attention = nn.ModuleList(
            nn.Linear(out_dim * 3, 1, bias=False) for _ in range(num_layers)
        )
        self.self_proj = nn.ModuleList(
            nn.Linear(out_dim, out_dim, bias=False) for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(out_dim) for _ in range(num_layers))
        # RREA concatenates layer-wise states. Projecting them restores the fixed
        # output width required by the existing semantic interaction module.
        self.layer_fusion = nn.Linear(out_dim * (num_layers + 1), out_dim)
        self.relation_aspect = nn.Linear(out_dim, out_dim, bias=False)
        self.output_norm = nn.LayerNorm(out_dim)

    @staticmethod
    def _reflect(entity_state: torch.Tensor, relation_normal: torch.Tensor) -> torch.Tensor:
        # (I - 2 rr^T)e, evaluated without materializing one matrix per edge.
        projection = (entity_state * relation_normal).sum(dim=-1, keepdim=True)
        return entity_state - 2.0 * projection * relation_normal

    def forward(self, x, edge_index, edge_type):
        h = self.input_proj(x)
        layer_states = [h]
        src, dst = edge_index
        relation_normal = F.normalize(self.relation_emb.weight, p=2, dim=-1)

        for layer in range(self.num_layers):
            edge_relation = relation_normal[edge_type]
            reflected = self._reflect(h[src], edge_relation)
            logits = self.attention[layer](
                torch.cat([h[dst], reflected, edge_relation], dim=-1)
            ).squeeze(-1)
            weights = softmax(F.leaky_relu(logits, negative_slope=0.2), dst, num_nodes=h.size(0))
            messages = weights.unsqueeze(-1) * reflected
            neighbor = h.new_zeros(h.shape)
            neighbor.index_add_(0, dst, messages)
            h = self.norms[layer](self.self_proj[layer](h) + neighbor)
            if layer + 1 < self.num_layers:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            layer_states.append(h)

        structural = self.layer_fusion(torch.cat(layer_states, dim=-1))
        relation_sum = h.new_zeros(h.shape)
        relation_sum.index_add_(0, dst, relation_normal[edge_type])
        degree = h.new_zeros(h.size(0))
        degree.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
        relation_aspect = relation_sum / degree.clamp(min=1.0).unsqueeze(-1)
        output = structural + self.relation_aspect(relation_aspect)
        return F.normalize(self.output_norm(output), p=2, dim=-1)
