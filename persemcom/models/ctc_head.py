"""Semantic decoder m^-1_delta (paper Sec. IV-C): 1-D conv to vocabulary logits + CTC greedy decoding."""
from typing import List

import torch
from torch import nn


class CTCHead(nn.Module):
    def __init__(self, d_model: int = 144, vocab_size: int = 5001):
        super().__init__()
        self.proj = nn.Conv1d(d_model, vocab_size, kernel_size=1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (B, T, d_model) -> log-probs (B, T, vocab_size)."""
        logits = self.proj(feats.transpose(1, 2)).transpose(1, 2)
        return logits.float().log_softmax(-1)


def ctc_greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor, blank_id: int) -> List[List[int]]:
    """Best-path decoding: argmax per frame, collapse repeats, drop blanks."""
    best = log_probs.argmax(-1).cpu()
    out = []
    for seq, n in zip(best, lengths.cpu().tolist()):
        seq = torch.unique_consecutive(seq[:n])
        out.append(seq[seq != blank_id].tolist())
    return out
