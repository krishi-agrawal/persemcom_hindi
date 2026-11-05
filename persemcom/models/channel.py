"""Untrainable wireless channel layer (paper Eq. 2): Y = H X + N with perfect CSI at the receiver.

Symbols are real tensors of shape (B, T, S, 2) holding S complex IQ symbols per frame.
The transmitter is expected to power-normalize so that E|x|^2 = 1, which makes the noise
variance per complex symbol 10^(-SNR/10).
"""
from typing import Optional, Union

import torch
from torch import nn

SNR = Union[float, torch.Tensor]


def power_normalize(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Scale each sample to unit average power per complex symbol over its valid frames.

    x: (B, T, S, 2); mask: (B, T) bool, True on valid (non-padding) frames. Padding frames are
    zeroed: they are not transmitted.
    """
    power = x.pow(2).sum(-1)  # |x|^2 per complex symbol, (B, T, S)
    if mask is None:
        mean_power = power.mean(dim=(1, 2))
    else:
        m = mask.unsqueeze(-1).to(power.dtype)
        mean_power = (power * m).sum(dim=(1, 2)) / (m.sum(dim=(1, 2)) * x.shape[2]).clamp_min(1)
        x = x * mask[..., None, None].to(x.dtype)
    return x / mean_power.clamp_min(1e-12).sqrt()[:, None, None, None]


def sample_snr(batch_size: int, low: float, high: float, device=None) -> torch.Tensor:
    """Per-sample SNR in dB, uniform on [low, high]."""
    return torch.empty(batch_size, device=device).uniform_(low, high)


class Channel(nn.Module):
    """AWGN or Rayleigh block-fading channel with zero-forcing (or MMSE) equalization.

    kind: "awgn" | "rayleigh" | "none"
    fading_block: "frame" draws one h ~ CN(0, 1) per transmitted frame of S symbols,
                  "sample" one per utterance (DeepSC-style quasi-static fading).
    equalizer: "zf" (x_hat = y / h) or "mmse" (x_hat = h* y / (|h|^2 + sigma^2)).
    """

    MAX_AMPLITUDE = 1e3

    def __init__(self, kind: str = "awgn", fading_block: str = "frame", equalizer: str = "zf"):
        super().__init__()
        if kind not in ("awgn", "rayleigh", "none"):
            raise ValueError(f"unknown channel kind: {kind}")
        if fading_block not in ("frame", "sample"):
            raise ValueError(f"unknown fading block: {fading_block}")
        if equalizer not in ("zf", "mmse"):
            raise ValueError(f"unknown equalizer: {equalizer}")
        self.kind, self.fading_block, self.equalizer = kind, fading_block, equalizer

    def forward(self, x: torch.Tensor, snr_db: SNR) -> torch.Tensor:
        if self.kind == "none":
            return x
        in_dtype = x.dtype
        xc = torch.view_as_complex(x.float().contiguous())  # (B, T, S), complex64
        B, T, _ = xc.shape

        snr = torch.as_tensor(snr_db, dtype=torch.float32, device=x.device).expand(B)
        noise_var = 10.0 ** (-snr / 10.0)  # per complex symbol, signal power 1
        noise_var = noise_var[:, None, None]
        # complex randn is CN(0, 1): each of real/imag has variance 1/2
        noise = torch.randn_like(xc) * noise_var.sqrt()

        if self.kind == "awgn":
            y = xc + noise
            return torch.view_as_real(y).to(in_dtype)

        shape = (B, T, 1) if self.fading_block == "frame" else (B, 1, 1)
        h = torch.randn(shape, dtype=torch.complex64, device=x.device)  # CN(0, 1)
        y = h * xc + noise
        if self.equalizer == "zf":
            x_hat = y / h
        else:
            x_hat = h.conj() * y / (h.abs() ** 2 + noise_var)
        # ZF in a deep fade (|h| -> 0) blows up the noise; those frames are lost anyway, and the clamp
        # keeps them finite under fp16 autocast in the channel decoder.
        return torch.view_as_real(x_hat).clamp(-self.MAX_AMPLITUDE, self.MAX_AMPLITUDE).to(in_dtype)
