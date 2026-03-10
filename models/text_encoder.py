import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMeanPooling(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).float()          # [B, L, 1]
        x = x * mask
        denom = mask.sum(dim=1).clamp(min=1e-6)   # [B, 1]
        return x.sum(dim=1) / denom


class MultiScaleTransformerEncoder(nn.Module):
    """
    输入:
      seq_x: [B, L, in_dim]   例如 PyG DBP15K 的 300 维词向量序列
    输出:
      out:   [B, hidden_dim]
    """

    def __init__(
        self,
        in_dim: int = 300,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, embed_dim)

        self.phrase_conv_3 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.phrase_conv_5 = nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = MaskedMeanPooling()

        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3),
        )

        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def build_mask(seq_x: torch.Tensor) -> torch.Tensor:
        """
        seq_x: [B, L, D]
        非零行视为有效 token
        """
        return (seq_x.abs().sum(dim=-1) > 0).long()

    def forward(self, seq_x: torch.Tensor) -> torch.Tensor:
        """
        seq_x: [B, L, in_dim]
        """
        mask = self.build_mask(seq_x)                       # [B, L]
        x = self.input_proj(seq_x)                         # [B, L, E]

        token_repr = self.pool(x, mask)                    # [B, E]

        x_conv = x.transpose(1, 2)                         # [B, E, L]
        phrase3 = self.phrase_conv_3(x_conv).transpose(1, 2)
        phrase5 = self.phrase_conv_5(x_conv).transpose(1, 2)
        phrase_repr = self.pool((phrase3 + phrase5) / 2.0, mask)

        key_padding_mask = (mask == 0)
        trans_out = self.transformer(x, src_key_padding_mask=key_padding_mask)
        global_repr = self.pool(trans_out, mask)

        concat = torch.cat([token_repr, phrase_repr, global_repr], dim=-1)
        gate = F.softmax(self.gate(concat), dim=-1)

        fused = torch.cat([
            token_repr * gate[:, 0:1],
            phrase_repr * gate[:, 1:2],
            global_repr * gate[:, 2:3],
        ], dim=-1)

        out = self.out_proj(fused)
        out = F.normalize(out, p=2, dim=-1)
        return out


class SimpleTextEncoder(nn.Module):
    """
    用于 w/o MST 的轻量语义基线：
    仅做输入投影 + masked mean pooling。
    """

    def __init__(self, in_dim: int = 300, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pool = MaskedMeanPooling()

    @staticmethod
    def build_mask(seq_x: torch.Tensor) -> torch.Tensor:
        return (seq_x.abs().sum(dim=-1) > 0).long()

    def forward(self, seq_x: torch.Tensor) -> torch.Tensor:
        mask = self.build_mask(seq_x)
        x = self.input_proj(seq_x)
        out = self.pool(x, mask)
        return F.normalize(out, p=2, dim=-1)
