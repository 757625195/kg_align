import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn_encoder import LightweightGNNEncoder
from .text_encoder import MultiScaleTransformerEncoder
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
    ):
        super().__init__()

        self.node_emb = nn.Embedding(num_nodes, node_input_dim)

        self.gnn_encoder = LightweightGNNEncoder(
            in_dim=node_input_dim,
            hidden_dim=gnn_hidden_dim,
            out_dim=fusion_dim,
            num_layers=gnn_layers,
            dropout=dropout,
        )

        self.text_encoder = MultiScaleTransformerEncoder(
            in_dim=text_input_dim,
            embed_dim=text_hidden_dim,
            hidden_dim=fusion_dim,
            num_heads=text_heads,
            num_layers=text_layers,
            dropout=dropout,
        )

        self.fusion = CrossModalFusion(
            dim=fusion_dim,
            hidden_dim=fusion_dim,
            dropout=dropout,
        )

        self.struct_projector = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )
        self.sem_projector = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )

        self.align_head = AlignmentHead()

    def encode_structure_all(self, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.node_emb.weight
        z_struct_all = self.gnn_encoder(x, edge_index)   # [N, D]
        return z_struct_all

    def encode_semantics(self, seq_features: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(seq_features)           # [B, D]

    def score_pairs(self, left_emb: torch.Tensor, right_emb: torch.Tensor) -> torch.Tensor:
        return self.align_head(left_emb, right_emb)

    @staticmethod
    def to_shared_space(projector: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(projector(x), p=2, dim=-1)

    def forward(
        self,
        node_ids: torch.Tensor,       # [B]
        edge_index: torch.Tensor,     # [2, E]
        seq_features: torch.Tensor,   # [B, L, Din]
        neighbor_ids: torch.Tensor,   # [B, K]
        neighbor_mask: torch.Tensor,  # [B, K]
    ):
        z_struct_all = self.encode_structure_all(edge_index)    # [N, D]

        z_struct = z_struct_all[node_ids]                      # [B, D]
        z_neighbor = z_struct_all[neighbor_ids]                # [B, K, D]

        z_sem = self.encode_semantics(seq_features)            # [B, D]
        z_struct_shared = self.to_shared_space(self.struct_projector, z_struct)
        z_sem_shared = self.to_shared_space(self.sem_projector, z_sem)
        z_neighbor_shared = self.to_shared_space(self.struct_projector, z_neighbor)

        fusion_out = self.fusion(
            s=z_sem_shared,
            t_self=z_struct_shared,
            t_nei=z_neighbor_shared,
            nei_mask=neighbor_mask,
            return_components=True,
        )

        return {
            "z_struct": z_struct,
            "z_sem": z_sem,
            "z_struct_shared": z_struct_shared,
            "z_sem_shared": z_sem_shared,
            "z_joint": fusion_out["z_joint"],
            "z_sem_enhanced": fusion_out["z_sem_enhanced"],
            "z_struct_enhanced": fusion_out["z_struct_enhanced"],
            "semantic_attention": fusion_out["semantic_attention"],
            "structural_gate": fusion_out["structural_gate"],
        }
