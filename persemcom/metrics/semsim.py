"""Semantic similarity (paper Eq. 8): cosine between BERT sentence representations of reference and
hypothesis, in [0, 1].

The paper uses BERT. For Hindi this uses LaBSE (a multilingual BERT trained for sentence embeddings).
Mean-pooled raw MuRIL was tried first and is unusable: it scored an unrelated sentence 0.993 against
1.000 for an identical one. LaBSE on the same probe: identical 1.000, gender inflection 1.000,
dropped word 0.997, different place 0.707, unrelated 0.466.
"""
from functools import lru_cache
from typing import List, Sequence

import torch

from persemcom.data.text_norm import normalize

MODEL = "sentence-transformers/LaBSE"


@lru_cache(maxsize=1)
def _load(device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL, device=device)


def embed(texts: List[str], device: str = None, batch_size: int = 64) -> torch.Tensor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return _load(device).encode([normalize(t) for t in texts], batch_size=batch_size, convert_to_tensor=True,
                                normalize_embeddings=True, show_progress_bar=False).float().cpu()


def similarity_scores(refs: Sequence[str], hyps: Sequence[str], device: str = None) -> torch.Tensor:
    """Per-pair cosine similarity, clamped to [0, 1]. An empty hypothesis scores 0."""
    sims = (embed(list(refs), device) * embed(list(hyps), device)).sum(-1).clamp(0, 1)
    empty = torch.tensor([not normalize(h) for h in hyps])
    return sims.masked_fill(empty, 0.0)


def similarity(refs: Sequence[str], hyps: Sequence[str], device: str = None) -> float:
    return float(similarity_scores(refs, hyps, device).mean())
