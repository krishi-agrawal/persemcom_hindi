import numpy as np
import pytest

from persemcom.baselines.digital_link import DigitalLink


@pytest.fixture(scope="module")
def bits():
    return np.random.default_rng(0).integers(0, 2, 5000).astype(np.uint8)


def test_rate_and_symbols():
    link = DigitalLink("awgn")
    assert abs(link.rate - 2 / 3) < 0.01
    assert link.symbols_for(1022) == 765 and link.symbols_for(1023) == 1530


@pytest.mark.parametrize("channel,snr", [("awgn", 10.0), ("rayleigh", 20.0)])
def test_error_free_at_high_snr(bits, channel, snr):
    r = DigitalLink(channel).send(bits, snr, seed=1)
    assert len(r.bits) == len(bits)
    assert r.block_ok.all()
    assert np.array_equal(r.bits, bits)


def test_fails_at_very_low_snr_and_reports_it(bits):
    r = DigitalLink("awgn").send(bits, -6.0, seed=1)
    assert not r.block_ok.all()
    assert (r.bits != bits).mean() > 0.05


def test_cliff_between_low_and_high_snr(bits):
    link = DigitalLink("awgn")
    ber = {s: (link.send(bits, s, seed=2).bits != bits).mean() for s in (0.0, 4.0)}
    assert ber[4.0] < ber[0.0]
