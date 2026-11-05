"""WER (paper Eq. 7) and CER on normalized Hindi text. CER is reported too because many Hindi
recognition errors are single-matra substitutions that WER counts as whole-word errors."""
from typing import Sequence

import jiwer

from persemcom.data.text_norm import normalize


def _prepare(refs: Sequence[str], hyps: Sequence[str]):
    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} references vs {len(hyps)} hypotheses")
    pairs = [(normalize(r), normalize(h)) for r, h in zip(refs, hyps)]
    pairs = [(r, h) for r, h in pairs if r]  # jiwer rejects empty references
    return [r for r, _ in pairs], [h for _, h in pairs]


def wer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    r, h = _prepare(refs, hyps)
    return jiwer.wer(r, h)


def cer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    r, h = _prepare(refs, hyps)
    return jiwer.cer(r, h)
