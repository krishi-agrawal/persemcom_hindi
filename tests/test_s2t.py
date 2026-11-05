import pytest
import torch

from persemcom.models.ctc_head import ctc_greedy_decode
from persemcom.models.persemcom_s2t import PerSemComS2T


def _batch(B=2, T=301, n_mels=80):
    torch.manual_seed(0)
    feats = torch.randn(B, T, n_mels)
    lens = torch.tensor([T, T - 100])
    targets = torch.randint(0, 5000, (B, 12))
    target_lens = torch.tensor([12, 7])
    return feats, lens, targets, target_lens


def test_encoder_size_matches_conformer_s():
    model = PerSemComS2T()
    n = sum(p.numel() for p in model.semantic_encoder.parameters())
    assert 8e6 < n < 13e6, n
    assert model.semantic_decoder.proj.out_channels == 5001
    assert model.blank_id == 5000


def test_shapes_and_symbol_layout():
    model = PerSemComS2T(channel="rayleigh").eval()
    feats, lens, _, _ = _batch()
    out = model(feats, lens, snr_db=5.0)
    T_sub = ((301 - 3) // 2 + 1 - 3) // 2 + 1
    assert out["F"].shape == (2, T_sub, 144)
    assert out["X"].shape == (2, T_sub, 32, 2)  # B x n_sub x 32 x 2, as in the paper
    assert out["log_probs"].shape == (2, T_sub, 5001)
    assert out["lengths"].tolist()[0] == T_sub


@pytest.mark.parametrize("stage", ["pretrain", "stage1", "stage2"])
def test_stage_losses_backprop_only_into_trainable_parts(stage):
    model = PerSemComS2T()
    model.set_stage(stage)
    model.train()
    feats, lens, targets, target_lens = _batch()
    out = model(feats, lens, snr_db=None if stage == "pretrain" else 10.0)
    loss = model.loss(out, targets, target_lens)
    assert torch.isfinite(loss)
    loss.backward()

    def has_grad(m):
        return any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())

    assert has_grad(model.semantic_encoder) == (stage != "stage1")
    assert has_grad(model.channel_encoder) == (stage != "pretrain")
    assert has_grad(model.channel_decoder) == (stage != "pretrain")
    if stage == "stage1":
        assert not model.semantic_encoder.training  # frozen codec has dropout off


def test_channel_requires_snr():
    model = PerSemComS2T()
    feats, lens, _, _ = _batch()
    with pytest.raises(ValueError):
        model(feats, lens)


def test_ctc_greedy_decode_collapses_and_drops_blank():
    blank = 4
    frames = torch.tensor([[1, 1, 4, 1, 2, 2, 4, 4, 3, 0]])
    log_probs = torch.nn.functional.one_hot(frames, 5).float().log()
    assert ctc_greedy_decode(log_probs, torch.tensor([10]), blank) == [[1, 1, 2, 3, 0]]
    assert ctc_greedy_decode(log_probs, torch.tensor([5]), blank) == [[1, 1, 2]]
