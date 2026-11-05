"""Speaker style extractor (paper Sec. IV-E.1): GE2E speaker-verification network.

3 LSTM layers of 768 cells -> 256-d linear projection -> L2 norm, trained with the GE2E softmax loss
(Wan et al., 2018). Input: log-mel fbank partials of 160 frames (1.6 s at 10 ms). Utterance embeddings
average the L2-normalized embeddings of 50%-overlapping partials, then re-normalize.
"""
import torch
import torch.nn.functional as F
from torch import nn

PARTIAL_FRAMES = 160


class SpeakerEncoder(nn.Module):
    def __init__(self, n_mels: int = 80, hidden: int = 768, layers: int = 3, emb_dim: int = 256):
        super().__init__()
        self.lstm = nn.LSTM(n_mels, hidden, num_layers=layers, batch_first=True)
        self.proj = nn.Linear(hidden, emb_dim)
        # GE2E similarity scale/bias (w > 0); the paper initializes w=10, b=-5
        self.w = nn.Parameter(torch.tensor(10.0))
        self.b = nn.Parameter(torch.tensor(-5.0))

    def forward(self, partials: torch.Tensor) -> torch.Tensor:
        """partials: (B, T, n_mels) -> (B, emb_dim), L2-normalized."""
        _, (h, _) = self.lstm(partials)
        return F.normalize(self.proj(h[-1]), dim=-1)

    def ge2e_loss(self, emb: torch.Tensor) -> torch.Tensor:
        """emb: (N speakers, M utterances, D), normalized. Softmax GE2E loss with exclusive centroids."""
        N, M, _ = emb.shape
        centroids = F.normalize(emb.mean(1), dim=-1)                       # (N, D)
        excl = F.normalize((emb.sum(1, keepdim=True) - emb) / (M - 1), dim=-1)  # (N, M, D), self excluded
        sim = torch.einsum("nmd,kd->nmk", emb, centroids)                  # (N, M, N)
        own = (emb * excl).sum(-1)                                         # (N, M)
        idx = torch.arange(N, device=emb.device)
        sim[idx, :, idx] = own
        sim = sim * self.w.clamp(min=1e-6) + self.b
        target = idx.repeat_interleave(M)
        return F.cross_entropy(sim.reshape(N * M, N), target)

    @torch.no_grad()
    def embed_utterance(self, feats: torch.Tensor, overlap: float = 0.5) -> torch.Tensor:
        """feats: (T, n_mels) -> (emb_dim,). Short utterances are zero-padded to one partial."""
        T = feats.shape[0]
        if T < PARTIAL_FRAMES:
            feats = F.pad(feats, (0, 0, 0, PARTIAL_FRAMES - T))
            T = PARTIAL_FRAMES
        step = max(1, int(PARTIAL_FRAMES * (1 - overlap)))
        starts = list(range(0, T - PARTIAL_FRAMES + 1, step))
        parts = torch.stack([feats[s:s + PARTIAL_FRAMES] for s in starts])
        return F.normalize(self.forward(parts).mean(0), dim=-1)
