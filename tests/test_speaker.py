import numpy as np
import torch
import torch.nn.functional as F

from persemcom.metrics.spk_sim import centroid_eer, eer
from persemcom.models.speaker_encoder import PARTIAL_FRAMES, SpeakerEncoder


def test_forward_is_normalized_256d():
    m = SpeakerEncoder(hidden=64)
    e = m(torch.randn(6, PARTIAL_FRAMES, 80))
    assert e.shape == (6, 256)
    assert torch.allclose(e.norm(dim=-1), torch.ones(6), atol=1e-5)


def test_ge2e_loss_prefers_separated_speakers():
    m = SpeakerEncoder(hidden=8)
    torch.manual_seed(0)
    N, M, D = 4, 5, 256
    centers = F.normalize(torch.randn(N, 1, D), dim=-1)
    tight = F.normalize(centers + 0.05 * torch.randn(N, M, D), dim=-1)
    mixed = F.normalize(torch.randn(N, M, D), dim=-1)
    assert m.ge2e_loss(tight) < m.ge2e_loss(mixed)


def test_ge2e_loss_backprops():
    m = SpeakerEncoder(hidden=16)
    e = m(torch.randn(4 * 3, PARTIAL_FRAMES, 80)).view(4, 3, -1)
    m.ge2e_loss(e).backward()
    assert m.lstm.weight_ih_l0.grad is not None and m.w.grad is not None


def test_embed_utterance_short_and_long():
    m = SpeakerEncoder(hidden=16).eval()
    for T in (50, PARTIAL_FRAMES, 777):
        e = m.embed_utterance(torch.randn(T, 80))
        assert e.shape == (256,) and abs(e.norm().item() - 1) < 1e-5


def test_eer():
    assert eer(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0])) == 0.0
    assert abs(eer(np.array([0.1, 0.2, 0.8, 0.9]), np.array([1, 1, 0, 0])) - 1.0) < 1e-9
    cent = {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([0.0, 1.0])}
    assert centroid_eer(torch.tensor([[1.0, 0.1], [0.1, 1.0]]), ["a", "b"], cent) == 0.0
