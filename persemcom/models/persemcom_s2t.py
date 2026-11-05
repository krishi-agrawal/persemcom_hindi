"""Speech-to-text half of PerSemCom: f_alpha -> g_beta -> channel -> g^-1_chi -> m^-1_delta,
plus the phased-training stage switch and losses of Algorithm 1 (Eqs. 15-16).
"""
from typing import Optional

import torch
import torch.nn.functional as Fn
from torch import nn

from .channel import Channel, SNR
from .channel_codec import ChannelDecoder, ChannelEncoder
from .ctc_head import CTCHead, ctc_greedy_decode
from .semantic_encoder import SemanticEncoder

STAGES = ("pretrain", "stage1", "stage2")


def length_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


class PerSemComS2T(nn.Module):
    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 144,
        num_heads: int = 4,
        ffn_dim: int = 576,
        num_layers: int = 16,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        codec_hidden: int = 256,
        n_symbols: int = 32,
        vocab_size: int = 5001,
        channel: str = "awgn",
        fading_block: str = "frame",
        equalizer: str = "mmse",
    ):
        super().__init__()
        self.blank_id = vocab_size - 1
        self.semantic_encoder = SemanticEncoder(n_mels, d_model, num_heads, ffn_dim, num_layers, conv_kernel, dropout)
        self.channel_encoder = ChannelEncoder(d_model, codec_hidden, n_symbols)
        self.channel = Channel(channel, fading_block, equalizer)
        self.channel_decoder = ChannelDecoder(d_model, codec_hidden, n_symbols)
        self.semantic_decoder = CTCHead(d_model, vocab_size)
        self.stage = "stage2"

    def set_stage(self, stage: str) -> None:
        """pretrain: ASR only, channel bypassed. stage1: train the channel codec only (semantic codec
        frozen, MSE loss). stage2: everything trainable end to end through the channel (CTC loss)."""
        if stage not in STAGES:
            raise ValueError(f"unknown stage: {stage}")
        self.stage = stage
        semantic = stage != "stage1"
        codec = stage != "pretrain"
        for m in (self.semantic_encoder, self.semantic_decoder):
            m.requires_grad_(semantic)
        for m in (self.channel_encoder, self.channel_decoder):
            m.requires_grad_(codec)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.stage == "stage1":
            # frozen semantic codec: also freeze dropout so F is a fixed target for Eq. 15
            self.semantic_encoder.eval()
            self.semantic_decoder.eval()
        return self

    def forward(self, feats: torch.Tensor, feat_lens: torch.Tensor, snr_db: Optional[SNR] = None) -> dict:
        F, lens = self.semantic_encoder(feats, feat_lens)
        if self.stage == "pretrain":
            F_hat, X = F, None
        else:
            if snr_db is None:
                raise ValueError("snr_db is required when transmitting over the channel")
            mask = length_mask(lens, F.shape[1])
            X = self.channel_encoder(F, mask)
            Y = self.channel(X, snr_db)
            F_hat = self.channel_decoder(Y)
        if self.stage == "stage1":
            F = F.detach()
        return {"F": F, "F_hat": F_hat, "X": X, "lengths": lens, "log_probs": self.semantic_decoder(F_hat)}

    def syntactic_loss(self, out: dict) -> torch.Tensor:
        """Eq. 15, MSE between F and F_hat over valid frames."""
        mask = length_mask(out["lengths"], out["F"].shape[1]).unsqueeze(-1)
        se = (out["F"].float() - out["F_hat"].float()).pow(2) * mask
        return se.sum() / (mask.sum() * out["F"].shape[-1]).clamp_min(1)

    def semantic_loss(self, out: dict, targets: torch.Tensor, target_lens: torch.Tensor) -> torch.Tensor:
        """Eq. 16, CTC negative log-likelihood (mean over batch, normalized by target length)."""
        return Fn.ctc_loss(
            out["log_probs"].transpose(0, 1), targets, out["lengths"], target_lens,
            blank=self.blank_id, zero_infinity=True,
        )

    def loss(self, out: dict, targets: torch.Tensor, target_lens: torch.Tensor) -> torch.Tensor:
        if self.stage == "stage1":
            return self.syntactic_loss(out)
        return self.semantic_loss(out, targets, target_lens)

    @torch.no_grad()
    def transcribe_ids(self, feats: torch.Tensor, feat_lens: torch.Tensor, snr_db: Optional[SNR] = None):
        out = self.forward(feats, feat_lens, snr_db)
        return ctc_greedy_decode(out["log_probs"], out["lengths"], self.blank_id)
