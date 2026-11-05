"""Semantic encoder f_alpha (paper Sec. IV-A): Conv2d subsampling + 16 Conformer blocks.

torchaudio's ConformerLayer is exactly Eq. 10a-d (half-step FFN, MHA, conv module, half-step FFN,
LayerNorm). It has no positional encoding of its own, so a sinusoidal one is added after subsampling.
"""
import math

import torch
from torch import nn
from torchaudio.models import Conformer


class Conv2dSubsampling(nn.Module):
    """Two ReLU-activated 3x3 stride-2 Conv2d layers with d_model channels: 4x time subsampling."""

    def __init__(self, n_mels: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, 3, stride=2), nn.ReLU(),
            nn.Conv2d(d_model, d_model, 3, stride=2), nn.ReLU(),
        )
        freq = ((n_mels - 3) // 2 + 1 - 3) // 2 + 1
        self.out = nn.Linear(d_model * freq, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def output_lengths(lengths: torch.Tensor) -> torch.Tensor:
        return ((lengths - 3) // 2 + 1 - 3) // 2 + 1

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """x: (B, T, n_mels) -> (B, T', d_model)."""
        x = self.conv(x.unsqueeze(1))  # (B, C, T', F')
        x = self.out(x.transpose(1, 2).flatten(2))
        return self.dropout(x), self.output_lengths(lengths)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)
        self.scale = math.sqrt(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x * self.scale + self.pe[: x.shape[1]])


class SemanticEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 144,
        num_heads: int = 4,
        ffn_dim: int = 576,
        num_layers: int = 16,
        conv_kernel: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.subsample = Conv2dSubsampling(n_mels, d_model, dropout)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout)
        self.conformer = Conformer(
            input_dim=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=conv_kernel,
            dropout=dropout,
        )

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor):
        """feats: (B, T, n_mels) log-mel fbank -> F: (B, T', d_model), lengths (B,)."""
        x, lengths = self.subsample(feats, lengths)
        x = self.pos_enc(x)
        return self.conformer(x, lengths)
