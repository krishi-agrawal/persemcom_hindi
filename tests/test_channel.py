import math

import pytest
import torch

from persemcom.models.channel import Channel, power_normalize


def _symbols(B=8, T=50, S=32):
    torch.manual_seed(0)
    return power_normalize(torch.randn(B, T, S, 2) * 3.0 + 0.5)


def test_power_normalize_unit_power():
    x = _symbols()
    assert x.pow(2).sum(-1).mean(dim=(1, 2)) == pytest.approx(torch.ones(8), abs=1e-5)


def test_power_normalize_ignores_and_zeroes_padding():
    torch.manual_seed(0)
    x = torch.randn(2, 10, 32, 2)
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[0, :10] = True
    mask[1, :4] = True
    y = power_normalize(x, mask)
    assert y[1, 4:].abs().max() == 0
    assert y[1, :4].pow(2).sum(-1).mean().item() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("snr_db", [-10.0, 0.0, 10.0, 20.0])
def test_awgn_empirical_snr(snr_db):
    x = _symbols(B=16, T=200)
    y = Channel("awgn")(x, snr_db)
    noise_power = (y - x).pow(2).sum(-1).mean().item()
    measured = 10 * math.log10(1.0 / noise_power)
    assert measured == pytest.approx(snr_db, abs=0.2)


def test_per_sample_snr_tensor():
    x = _symbols(B=2, T=400)
    y = Channel("awgn")(x, torch.tensor([0.0, 20.0]))
    p = (y - x).pow(2).sum(-1).mean(dim=(1, 2))
    assert 10 * math.log10(1 / p[0].item()) == pytest.approx(0.0, abs=0.2)
    assert 10 * math.log10(1 / p[1].item()) == pytest.approx(20.0, abs=0.2)


@pytest.mark.parametrize("block", ["frame", "sample"])
def test_rayleigh_zf_recovers_at_high_snr(block):
    x = _symbols()
    x_hat = Channel("rayleigh", fading_block=block, equalizer="zf")(x, 40.0)
    # E[1/|h|^2] is unbounded for Rayleigh, so check the median, not the mean
    err = (x_hat - x).pow(2).sum(-1).flatten()
    assert err.median().item() < 1e-3
    assert torch.isfinite(x_hat).all()


def test_rayleigh_mmse_is_bounded():
    x = _symbols()
    x_hat = Channel("rayleigh", equalizer="mmse")(x, 0.0)
    assert torch.isfinite(x_hat).all()
    assert x_hat.abs().max() < 1e3


def test_none_channel_is_identity():
    x = _symbols()
    assert torch.equal(Channel("none")(x, 0.0), x)


def test_bad_kind():
    with pytest.raises(ValueError):
        Channel("rician")
