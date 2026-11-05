"""Personalized speech generator G_S (paper Sec. IV-E, Fig. 5): Tacotron 2 conditioned on the speaker
embedding I(si), which is concatenated to every encoder output frame before attention.

Built from torchaudio's Tacotron 2 parts. The decoder is constructed with memory width
encoder_dim + speaker_dim. torchaudio says only 1 frame per step is supported, but its decoder's input
and output reshaping handles r > 1 correctly: this module pads mels to a multiple of r, predicts one stop
token per decoder step, and scales the inference lengths by r.
"""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchaudio.models.tacotron2 import _Decoder, _Encoder, _Postnet


def lengths_to_mask(lengths: Tensor, max_len: Optional[int] = None) -> Tensor:
    max_len = max_len or int(lengths.max())
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


class SpeakerTacotron2(nn.Module):
    def __init__(
        self,
        n_symbol: int,
        n_mels: int = 100,
        speaker_dim: int = 256,
        frames_per_step: int = 2,
        symbol_embedding_dim: int = 512,
        encoder_dim: int = 512,
        encoder_n_convolution: int = 3,
        encoder_kernel_size: int = 5,
        decoder_rnn_dim: int = 1024,
        decoder_max_step: int = 1000,
        decoder_dropout: float = 0.1,
        attention_rnn_dim: int = 1024,
        attention_hidden_dim: int = 128,
        attention_location_n_filter: int = 32,
        attention_location_kernel_size: int = 31,
        attention_dropout: float = 0.1,
        prenet_dim: int = 256,
        postnet_n_convolution: int = 5,
        postnet_kernel_size: int = 5,
        postnet_embedding_dim: int = 512,
        gate_threshold: float = 0.5,
        input_frame_dropout: float = 0.0,
    ):
        super().__init__()
        if symbol_embedding_dim != encoder_dim:
            raise ValueError("torchaudio's encoder expects symbol_embedding_dim == encoder_dim")
        self.n_mels, self.r, self.speaker_dim = n_mels, frames_per_step, speaker_dim
        # Training only: replace this fraction of teacher-forced input frame groups with the utterance's mean
        # frame, so the decoder cannot just continue the previous frame (free-running output droned otherwise)
        self.input_frame_dropout = input_frame_dropout
        self.embedding = nn.Embedding(n_symbol, symbol_embedding_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.encoder = _Encoder(encoder_dim, encoder_n_convolution, encoder_kernel_size)
        self.decoder = _Decoder(
            n_mels, frames_per_step, encoder_dim + speaker_dim, decoder_rnn_dim, decoder_max_step,
            decoder_dropout, True, attention_rnn_dim, attention_hidden_dim, attention_location_n_filter,
            attention_location_kernel_size, attention_dropout, prenet_dim, gate_threshold,
        )
        self.postnet = _Postnet(n_mels, postnet_embedding_dim, postnet_kernel_size, postnet_n_convolution)

    def _memory(self, tokens: Tensor, token_lens: Tensor, spk: Tensor) -> Tensor:
        enc = self.encoder(self.embedding(tokens).transpose(1, 2), token_lens)  # (B, L, encoder_dim)
        return torch.cat([enc, spk[:, None, :].expand(-1, enc.shape[1], -1).to(enc.dtype)], dim=-1)

    def pad_to_r(self, mel: Tensor) -> Tensor:
        """(B, n_mels, T) -> right-padded so T is a multiple of r."""
        extra = (-mel.shape[-1]) % self.r
        return F.pad(mel, (0, extra)) if extra else mel

    def forward(self, tokens: Tensor, token_lens: Tensor, mel: Tensor, mel_lens: Tensor, spk: Tensor):
        """Teacher-forced pass. mel: (B, n_mels, T) with T a multiple of r.
        Returns mel_before (B, n_mels, T), mel_after, gate logits (B, T/r), alignments (B, T/r, L)."""
        memory = self._memory(tokens, token_lens, spk)
        # torchaudio transposes to (B, T, n_mels) and then calls .view(B, T/r, -1); hand it a tensor whose
        # transpose is contiguous so that view is valid for r > 1
        frames = mel.transpose(1, 2).contiguous()  # (B, T, n_mels): decoder *inputs*; targets stay `mel`
        if self.training and self.input_frame_dropout > 0:
            B, T, _ = frames.shape
            drop = torch.rand(B, T // self.r, device=mel.device) < self.input_frame_dropout
            drop = drop.repeat_interleave(self.r, dim=1)[:, :, None]
            valid = lengths_to_mask(mel_lens, T)[:, :, None].to(frames.dtype)
            mean = (frames * valid).sum(1, keepdim=True) / valid.sum(1, keepdim=True).clamp_min(1)
            frames = torch.where(drop, mean.expand_as(frames), frames)
        mel_before, gate, align = self.decoder(memory, frames.transpose(1, 2), memory_lengths=token_lens)
        mel_after = mel_before + self.postnet(mel_before)
        mask = ~lengths_to_mask(mel_lens, mel.shape[-1])[:, None, :]
        mel_before = mel_before.masked_fill(mask, 0.0)
        mel_after = mel_after.masked_fill(mask, 0.0)
        return mel_before, mel_after, gate, align

    @torch.no_grad()
    def infer(self, tokens: Tensor, token_lens: Tensor, spk: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Any batch order. Returns mel (B, n_mels, T), frame lengths (B,), alignments (B, steps, L)."""
        order = torch.argsort(token_lens, descending=True)  # the encoder needs decreasing lengths
        undo = torch.argsort(order)
        tokens, token_lens, spk = tokens[order], token_lens[order], spk[order]
        memory = self._memory(tokens, token_lens, spk)
        mel, step_lens, _, align = self.decoder.infer(memory, token_lens)
        # torchaudio's decoder.infer returns alignments flattened to (L, steps * B); un-flatten as its
        # Tacotron2.infer does -> (B, steps, L)
        B = tokens.shape[0]
        align = align.unfold(1, B, B).transpose(0, 2)
        mel = mel + self.postnet(mel)
        lens = step_lens.long() * self.r
        mel = mel.masked_fill(~lengths_to_mask(lens, mel.shape[-1])[:, None, :], 0.0)
        return mel[undo], lens[undo], align[undo]


def guided_attention_loss(align: Tensor, token_lens: Tensor, step_lens: Tensor, g: float = 0.2) -> Tensor:
    """Tachibana et al. (2018): penalize attention far from the diagonal. align: (B, steps, L).

    Normalized per decoder step (sum over text positions, mean over valid steps): each step's attention
    sums to 1 over the text, so the loss is the expected off-diagonal penalty per step, in [0, 1).
    Averaging over all (step, character) cells instead, as ESPnet does, divides it by the text length
    (~80 Hindi graphemes): a fully stuck attention then scored ~0.008, too weak to teach alignment on
    this corpus (run 1: attention covered 14% of the text after 5000 steps).
    """
    B, S, L = align.shape
    n = torch.arange(L, device=align.device)[None, None, :] / token_lens[:, None, None].float()
    t = torch.arange(S, device=align.device)[None, :, None] / step_lens[:, None, None].float()
    W = 1.0 - torch.exp(-((n - t) ** 2) / (2 * g * g))
    mask = lengths_to_mask(step_lens, S)[:, :, None] & lengths_to_mask(token_lens, L)[:, None, :]
    return (align.float() * W * mask).sum() / lengths_to_mask(step_lens, S).sum()


def attention_coverage(align: Tensor, token_lens: Tensor, step_lens: Tensor) -> float:
    """Fraction of text positions that are the attention argmax at some decoder step (~1 when aligned)."""
    cov = []
    for a, L, S in zip(align, token_lens.tolist(), step_lens.tolist()):
        cov.append(len(set(a[:S, :L].argmax(-1).tolist())) / L)
    return float(sum(cov) / len(cov))


def attention_focus_rate(align: Tensor, token_lens: Tensor, step_lens: Tensor) -> float:
    """Mean over valid decoder steps of the max attention weight; ~1 for sharp diagonal alignment."""
    B, S, L = align.shape
    mask_s = lengths_to_mask(step_lens, S)
    mask_l = lengths_to_mask(token_lens, L)[:, None, :]
    peak = (align.float() * mask_l).max(-1).values
    return float((peak * mask_s).sum() / mask_s.sum())
