import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn_encoder import LightweightGNNEncoder
from .text_encoder import MultiScaleTransformerEncoder, SimpleTextEncoder
from .fusion import CrossModalFusion
from .alignment_head import AlignmentHead


class JointEAModel(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        text_input_dim: int = 300,
        node_input_dim: int = 128,
        gnn_hidden_dim: int = 128,
        text_hidden_dim: int = 128,
        fusion_dim: int = 128,
        gnn_layers: int = 2,
        text_heads: int = 4,
        text_layers: int = 2,
        dropout: float = 0.1,
        use_mst: bool = True,
        use_light_gnn: bool = True,
        use_cross_modal_enhancement: bool = True,
        gnn_share_parameters: bool = False,
        gnn_use_depthwise_separable: bool = False,
        use_explicit_topology_matching: bool = True,
    ):
        super().__init__()
        self.use_mst = use_mst
        self.use_light_gnn = use_light_gnn
        self.use_cross_modal_enhancement = use_cross_modal_enhancement
        self.use_explicit_topology_matching = use_explicit_topology_matching

        self.node_emb = nn.Embedding(num_nodes, node_input_dim)
        self.node_proj = nn.Linear(node_input_dim, fusion_dim)

        if use_light_gnn:
            self.gnn_encoder = LightweightGNNEncoder(
                in_dim=node_input_dim,
                hidden_dim=gnn_hidden_dim,
                out_dim=fusion_dim,
                num_layers=gnn_layers,
                dropout=dropout,
                share_parameters=gnn_share_parameters,
                use_depthwise_separable=gnn_use_depthwise_separable,
            )
        else:
            self.gnn_encoder = None

        if use_mst:
            self.text_encoder = MultiScaleTransformerEncoder(
                in_dim=text_input_dim,
                embed_dim=text_hidden_dim,
                hidden_dim=fusion_dim,
                num_heads=text_heads,
                num_layers=text_layers,
                dropout=dropout,
            )
        else:
            self.text_encoder = SimpleTextEncoder(
                in_dim=text_input_dim,
                hidden_dim=fusion_dim,
                dropout=dropout,
            )

        self.fusion = CrossModalFusion(
            dim=fusion_dim,
            hidden_dim=fusion_dim,
            dropout=dropout,
        )

        self.align_head = AlignmentHead()

    def encode_structure_all(self, edge_index: torch.Tensor) -> torch.Tensor:
        # 结构分支先对整图编码，再按当前 batch 取实体及其邻居表示。
        x = self.node_emb.weight
        if self.gnn_encoder is None:
            return F.normalize(self.node_proj(x), p=2, dim=-1)
        z_struct_all = self.gnn_encoder(x, edge_index)
        return z_struct_all

    def encode_semantics(self, seq_features: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(seq_features)           # [B, D]

    def score_pairs(self, left_emb: torch.Tensor, right_emb: torch.Tensor) -> torch.Tensor:
        return self.align_head(left_emb, right_emb)

    def forward(
        self,
        node_ids: torch.Tensor,       # [B]
        edge_index: torch.Tensor,     # [2, E]
        seq_features: torch.Tensor,   # [B, L, Din]
        neighbor_ids: torch.Tensor,   # [B, K]
        neighbor_mask: torch.Tensor,  # [B, K]
    ):
        # 全图结构编码 + batch 级局部邻居提取，是当前轻量化设计的核心：
        # 用一次稀疏消息传递覆盖多跳上下文，再用固定邻居预算控制后续交互成本。
        z_struct_all = self.encode_structure_all(edge_index)    # [N, D]

        z_struct = z_struct_all[node_ids]                       # [B, D]
        z_neighbor = z_struct_all[neighbor_ids]                # [B, K, D]
        hop_features = {}
        if self.use_explicit_topology_matching and self.gnn_encoder is not None:
            hop_features = self.gnn_encoder.collect_k_hop_features(
                z_struct_all=z_struct_all,
                edge_index=edge_index,
                max_hop=3,
            )

        z_sem = self.encode_semantics(seq_features)            # [B, D]

        if self.use_cross_modal_enhancement:
            fusion_out = self.fusion(
                s=z_sem,
                t_self=z_struct,
                t_nei=z_neighbor,
                nei_mask=neighbor_mask,
                return_components=True,
            )
        else:
            z_joint = F.normalize(0.5 * (z_struct + z_sem), p=2, dim=-1)
            zeros = torch.zeros(
                z_joint.size(0),
                z_neighbor.size(1),
                device=z_joint.device,
                dtype=z_joint.dtype,
            )
            fusion_out = {
                "z_joint": z_joint,
                "z_sem_enhanced": z_sem,
                "z_struct_enhanced": z_struct,
                "semantic_attention": zeros,
                "structural_gate": zeros,
            }

        return {
            "z_struct": z_struct,
            "z_sem": z_sem,
            "z_joint": fusion_out["z_joint"],
            "z_sem_enhanced": fusion_out["z_sem_enhanced"],
            "z_struct_enhanced": fusion_out["z_struct_enhanced"],
            "semantic_attention": fusion_out["semantic_attention"],
            "structural_gate": fusion_out["structural_gate"],
            "z_hop1": hop_features[1][node_ids] if 1 in hop_features else z_struct,
            "z_hop2": hop_features[2][node_ids] if 2 in hop_features else z_struct,
            "z_hop3": hop_features[3][node_ids] if 3 in hop_features else z_struct,
        }
