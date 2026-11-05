"""Channel encoder g_beta / decoder g^-1_chi (paper Sec. IV-B).

Encoder: Dense 256 -> Dense 64 -> reshape to 32 complex IQ symbols per frame -> power normalization.
Decoder: reshape -> Dense 256 -> Dense d_att (144).
"""
from typing import Optional

import torch
from torch import nn

from .channel import power_normalize


class ChannelEncoder(nn.Module):
    def __init__(self, d_model: int = 144, hidden: int = 256, n_symbols: int = 32):
        super().__init__()
        self.n_symbols = n_symbols
        self.net = nn.Sequential(nn.Linear(d_model, hidden), nn.ReLU(), nn.Linear(hidden, 2 * n_symbols))

    def forward(self, feats: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """feats: (B, T, d_model) -> unit-power symbols (B, T, n_symbols, 2)."""
        x = self.net(feats).unflatten(-1, (self.n_symbols, 2))
        return power_normalize(x, mask)


class ChannelDecoder(nn.Module):
    def __init__(self, d_model: int = 144, hidden: int = 256, n_symbols: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * n_symbols, hidden), nn.ReLU(), nn.Linear(hidden, d_model))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """y: (B, T, n_symbols, 2) -> recovered semantic features (B, T, d_model)."""
        return self.net(y.flatten(-2))
