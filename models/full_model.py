import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn_encoder import (
    LightweightGNNEncoder,
    RDGCNStructuralEncoder,
    RREAStructuralEncoder,
    RelationAwareGNNEncoder,
)
from .text_encoder import MultiScaleTransformerEncoder, SimpleTextEncoder
from .fusion import CrossModalFusion, LateConcatFusion, ParameterMatchedLateFusion
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
        ce_residual_ratio: float = 0.3,
        use_mst: bool = True,
        use_light_gnn: bool = True,
        use_cross_modal_enhancement: bool = True,
        gnn_share_parameters: bool = False,
        gnn_use_depthwise_separable: bool = False,
        use_relation_gnn: bool = True,
        num_relations: int = 0,
        relation_layer_fusion: bool = True,
        use_relation_types: bool = True,
        alignment_csls_k: int = 0,
        alignment_csls_blend: float = 1.0,
        sem_use_token_view: bool = True,
        sem_use_phrase_view: bool = True,
        sem_use_global_view: bool = True,
        sem_residual_mode: str = "gated",
        fusion_mode: str = "early_interaction",
        neighbor_query_mode: str = "semantic",
        neighbor_gate_mode: str = "semantic",
        neighbor_query_semantic_weight: float = 0.5,
        neighbor_attention: str = "softmax",
        neighbor_attention_temperature: float = 1.0,
        structure_encoder: str = "relation_gnn",
        structure_initialization: str = "learned",
        share_seed_structure_embeddings: bool = False,
        topology_entity_residual_ratio: float = 0.1,
    ):
        super().__init__()
        self.use_mst = use_mst
        self.use_light_gnn = use_light_gnn
        normalized_fusion_mode = fusion_mode.strip().lower()
        if not use_cross_modal_enhancement and normalized_fusion_mode == "early_interaction":
            normalized_fusion_mode = "mean"
        valid_fusion_modes = {
            "early_interaction",
            "reciprocal_add",
            "reciprocal_residual_add",
            "late_concat_mlp",
            "late_concat_matched",
            "mean",
            "structure_only",
            "semantic_only",
        }
        if normalized_fusion_mode not in valid_fusion_modes:
            raise ValueError(
                f"Unsupported fusion_mode={fusion_mode!r}; "
                f"expected one of {sorted(valid_fusion_modes)}"
            )
        self.fusion_mode = normalized_fusion_mode
        self.use_cross_modal_enhancement = self.fusion_mode in {
            "early_interaction",
            "reciprocal_add",
            "reciprocal_residual_add",
        }
        self.use_relation_gnn = use_relation_gnn and num_relations > 0
        self.num_relations = num_relations
        self.structure_encoder = structure_encoder.strip().lower()
        if self.structure_encoder not in {"relation_gnn", "rdgcn", "rrea"}:
            raise ValueError("structure_encoder must be 'relation_gnn', 'rdgcn', or 'rrea'")
        self.structure_initialization = structure_initialization.strip().lower()
        if self.structure_initialization not in {"learned", "name", "topology"}:
            raise ValueError("structure_initialization must be 'learned', 'name', or 'topology'")

        self.node_emb = nn.Embedding(num_nodes, node_input_dim)
        self.share_seed_structure_embeddings = share_seed_structure_embeddings
        self.register_buffer("structure_seed_left", None, persistent=False)
        self.register_buffer("structure_seed_right", None, persistent=False)
        self.name_node_proj = nn.Linear(text_input_dim, node_input_dim)
        self.topology_node_proj = nn.Linear(8, node_input_dim)
        self.topology_entity_residual_ratio = topology_entity_residual_ratio
        self.register_buffer("structure_name_features", None, persistent=False)
        self.register_buffer("structure_topology_features", None, persistent=False)
        self.node_proj = nn.Linear(node_input_dim, fusion_dim)

        if use_light_gnn and self.use_relation_gnn and self.structure_encoder == "rdgcn":
            self.gnn_encoder = RDGCNStructuralEncoder(
                in_dim=node_input_dim,
                hidden_dim=gnn_hidden_dim,
                out_dim=fusion_dim,
                num_relations=num_relations,
                num_layers=gnn_layers,
                dropout=dropout,
            )
        elif use_light_gnn and self.use_relation_gnn and self.structure_encoder == "rrea":
            self.gnn_encoder = RREAStructuralEncoder(
                in_dim=node_input_dim,
                hidden_dim=gnn_hidden_dim,
                out_dim=fusion_dim,
                num_relations=num_relations,
                num_layers=gnn_layers,
                dropout=dropout,
            )
        elif use_light_gnn and self.use_relation_gnn:
            self.gnn_encoder = RelationAwareGNNEncoder(
                in_dim=node_input_dim,
                hidden_dim=gnn_hidden_dim,
                out_dim=fusion_dim,
                num_relations=num_relations,
                num_layers=gnn_layers,
                dropout=dropout,
                share_parameters=gnn_share_parameters,
                use_layer_fusion=relation_layer_fusion,
                use_relation_types=use_relation_types,
            )
        elif use_light_gnn:
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
                use_token_view=sem_use_token_view,
                use_phrase_view=sem_use_phrase_view,
                use_global_view=sem_use_global_view,
                residual_mode=sem_residual_mode,
            )
        else:
            self.text_encoder = SimpleTextEncoder(
                in_dim=text_input_dim,
                hidden_dim=fusion_dim,
                dropout=dropout,
            )

        self.fusion = None
        self.late_fusion = None
        if self.fusion_mode in {
            "early_interaction",
            "reciprocal_add",
            "reciprocal_residual_add",
        }:
            self.fusion = CrossModalFusion(
                dim=fusion_dim,
                hidden_dim=fusion_dim,
                dropout=dropout,
                residual_ratio=ce_residual_ratio,
                neighbor_query_mode=neighbor_query_mode,
                neighbor_gate_mode=neighbor_gate_mode,
                neighbor_query_semantic_weight=neighbor_query_semantic_weight,
                neighbor_attention=neighbor_attention,
                neighbor_attention_temperature=neighbor_attention_temperature,
                reciprocal_add=self.fusion_mode == "reciprocal_add",
                reciprocal_residual_add=(
                    self.fusion_mode == "reciprocal_residual_add"
                ),
            )
        elif self.fusion_mode == "late_concat_mlp":
            self.late_fusion = LateConcatFusion(
                dim=fusion_dim,
                hidden_dim=fusion_dim,
                dropout=dropout,
            )
        elif self.fusion_mode == "late_concat_matched":
            self.late_fusion = ParameterMatchedLateFusion(
                dim=fusion_dim,
                hidden_dim=fusion_dim * 4,
                dropout=dropout,
            )

        self.align_head = AlignmentHead(
            csls_k=alignment_csls_k,
            csls_blend=alignment_csls_blend,
        )

    def encode_structure_all(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor = None,
    ) -> torch.Tensor:
        # 结构分支先对整图编码，再按当前 batch 取实体及其邻居表示。
        if self.structure_initialization == "name":
            if self.structure_name_features is None:
                raise ValueError("Name-based structure initialization requires name features")
            x = self.name_node_proj(self.structure_name_features)
        elif self.structure_initialization == "topology":
            if self.structure_topology_features is None:
                raise ValueError("Topology initialization requires topology features")
            x = self.topology_node_proj(self.structure_topology_features)
            x = x + self.topology_entity_residual_ratio * self.node_emb.weight
        else:
            x = self.node_emb.weight
        if self.share_seed_structure_embeddings:
            if self.structure_seed_left is None or self.structure_seed_right is None:
                raise ValueError("Seed-shared structure initialization requires training seed pairs")
            shared = 0.5 * (
                x[self.structure_seed_left] + x[self.structure_seed_right]
            )
            x = x.clone()
            x.index_copy_(0, self.structure_seed_left, shared)
            x.index_copy_(0, self.structure_seed_right, shared)
        if self.gnn_encoder is None:
            return F.normalize(self.node_proj(x), p=2, dim=-1)
        if self.use_relation_gnn:
            if edge_type is None:
                raise ValueError("Relation-aware GNN requires edge_type")
            z_struct_all = self.gnn_encoder(x, edge_index, edge_type)
        else:
            z_struct_all = self.gnn_encoder(x, edge_index)
        return z_struct_all

    def set_structure_name_features(self, features: torch.Tensor) -> None:
        if features.size(0) != self.node_emb.num_embeddings:
            raise ValueError("Name feature count must match the number of entities")
        self.structure_name_features = features

    def set_structure_topology_features(self, features: torch.Tensor) -> None:
        if features.shape != (self.node_emb.num_embeddings, 8):
            raise ValueError("Topology features must have shape [num_entities, 8]")
        self.structure_topology_features = features

    def set_structure_seed_pairs(self, pairs) -> None:
        pair_tensor = torch.as_tensor(pairs, dtype=torch.long, device=self.node_emb.weight.device)
        if pair_tensor.ndim != 2 or pair_tensor.size(1) != 2:
            raise ValueError("Structure seed pairs must have shape [N, 2]")
        self.structure_seed_left = pair_tensor[:, 0]
        self.structure_seed_right = pair_tensor[:, 1]

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
        edge_type: torch.Tensor = None,  # [E]
        z_struct_all: torch.Tensor = None,  # [N, D]
    ):
        # The structural graph is encoded once, then local neighbors are
        # gathered for the current batch using the configured collection mode.
        if z_struct_all is None:
            z_struct_all = self.encode_structure_all(
                edge_index=edge_index,
                edge_type=edge_type,
            )    # [N, D]

        z_struct = z_struct_all[node_ids]                       # [B, D]
        z_neighbor = z_struct_all[neighbor_ids]                # [B, K, D]
        z_sem = self.encode_semantics(seq_features)            # [B, D]

        if self.fusion_mode in {
            "early_interaction",
            "reciprocal_add",
            "reciprocal_residual_add",
        }:
            fusion_out = self.fusion(
                s=z_sem,
                t_self=z_struct,
                t_nei=z_neighbor,
                nei_mask=neighbor_mask,
                return_components=True,
            )
        elif self.fusion_mode in {"late_concat_mlp", "late_concat_matched"}:
            fusion_out = self.late_fusion(
                s=z_sem,
                t_self=z_struct,
                t_nei=z_neighbor,
                return_components=True,
            )
        elif self.fusion_mode == "mean":
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
        else:
            z_joint = z_struct if self.fusion_mode == "structure_only" else z_sem
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

        outputs = {
            "z_struct": z_struct,
            "z_sem": z_sem,
            "z_joint": fusion_out["z_joint"],
            "z_sem_enhanced": fusion_out["z_sem_enhanced"],
            "z_struct_enhanced": fusion_out["z_struct_enhanced"],
            "semantic_attention": fusion_out["semantic_attention"],
            "structural_gate": fusion_out["structural_gate"],
        }
        for key in ("neighbor_support_size", "neighbor_valid_count"):
            if key in fusion_out:
                outputs[key] = fusion_out[key]
        return outputs
