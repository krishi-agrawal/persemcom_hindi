"""Speaker-verification metrics: EER over (utterance, speaker-centroid) trials, and cosine similarity."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve


def eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """Equal error rate given trial scores and 1/0 same-speaker labels."""
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def centroid_eer(utt_emb: torch.Tensor, utt_spk: list, centroids: dict) -> float:
    """Score every utterance against every speaker centroid (closed-set, as in the knowledge base)."""
    spks = sorted(centroids)
    C = torch.stack([centroids[s] for s in spks])
    scores = (F.normalize(utt_emb, dim=-1) @ F.normalize(C, dim=-1).T).cpu().numpy()
    col = {s: i for i, s in enumerate(spks)}
    labels = np.zeros_like(scores)
    for r, s in enumerate(utt_spk):
        labels[r, col[s]] = 1
    return eer(scores.ravel(), labels.ravel())


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a, b, dim=-1)
