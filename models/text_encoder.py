import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 256):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.register_buffer("pe", self._build_encoding(max_len, dim), persistent=False)

    @staticmethod
    def _build_encoding(length: int, dim: int) -> torch.Tensor:
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / dim)
        )
        pe = torch.zeros(length, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            pe = self._build_encoding(seq_len, self.dim).to(device=x.device, dtype=x.dtype)
        else:
            pe = self.pe[:, :seq_len].to(device=x.device, dtype=x.dtype)
        return x + pe


class MaskedMeanPooling(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).float()          # [B, L, 1]
        x = x * mask
        denom = mask.sum(dim=1).clamp(min=1e-6)   # [B, 1]
        return x.sum(dim=1) / denom


class MaskedAttentionPooling(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.score(x).squeeze(-1)
        logits = logits.masked_fill(mask == 0, -1e9)
        weights = F.softmax(logits, dim=-1)
        weights = weights * mask.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


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
        self.input_norm = nn.LayerNorm(embed_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.positional_encoding = SinusoidalPositionalEncoding(embed_dim)

        self.phrase_conv_3 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.phrase_conv_5 = nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2)
        self.phrase_norm = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mean_pool = MaskedMeanPooling()
        self.token_pool = MaskedAttentionPooling(embed_dim, dropout=dropout)
        self.phrase_pool = MaskedAttentionPooling(embed_dim, dropout=dropout)
        self.global_pool = MaskedAttentionPooling(embed_dim, dropout=dropout)

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
        self.res_proj = nn.Identity() if embed_dim == hidden_dim else nn.Linear(embed_dim, hidden_dim)

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
        x = self.input_norm(x)
        x = self.positional_encoding(x)
        x = self.input_dropout(x)

        token_mean = self.mean_pool(x, mask)
        token_attn = self.token_pool(x, mask)
        token_repr = 0.5 * (token_mean + token_attn)       # [B, E]

        x_conv = x.transpose(1, 2)                         # [B, E, L]
        phrase3 = self.phrase_conv_3(x_conv).transpose(1, 2)
        phrase5 = self.phrase_conv_5(x_conv).transpose(1, 2)
        phrase_hidden = self.phrase_norm(x + 0.5 * (phrase3 + phrase5))
        phrase_mean = self.mean_pool(phrase_hidden, mask)
        phrase_attn = self.phrase_pool(phrase_hidden, mask)
        phrase_repr = 0.5 * (phrase_mean + phrase_attn)

        key_padding_mask = (mask == 0)
        trans_out = self.transformer(x, src_key_padding_mask=key_padding_mask)
        global_mean = self.mean_pool(trans_out, mask)
        global_attn = self.global_pool(trans_out, mask)
        global_repr = 0.5 * (global_mean + global_attn)

        concat = torch.cat([token_repr, phrase_repr, global_repr], dim=-1)
        gate = F.softmax(self.gate(concat), dim=-1)

        fused = torch.cat([
            token_repr * gate[:, 0:1],
            phrase_repr * gate[:, 1:2],
            global_repr * gate[:, 2:3],
        ], dim=-1)

        out = self.out_proj(fused)
        out = out + self.res_proj(global_repr)
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
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.positional_encoding = SinusoidalPositionalEncoding(hidden_dim)
        self.mean_pool = MaskedMeanPooling()
        self.attn_pool = MaskedAttentionPooling(hidden_dim, dropout=dropout)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def build_mask(seq_x: torch.Tensor) -> torch.Tensor:
        return (seq_x.abs().sum(dim=-1) > 0).long()

    def forward(self, seq_x: torch.Tensor) -> torch.Tensor:
        mask = self.build_mask(seq_x)
        x = self.input_proj(seq_x)
        x = self.input_norm(x)
        x = self.positional_encoding(x)
        mean_repr = self.mean_pool(x, mask)
        attn_repr = self.attn_pool(x, mask)
        out = self.out_proj(torch.cat([mean_repr, attn_repr], dim=-1))
        out = out + mean_repr
        return F.normalize(out, p=2, dim=-1)
