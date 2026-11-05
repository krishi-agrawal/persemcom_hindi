import numpy as np

from persemcom.metrics.mcd import mcd


def _tone(f, sr=16000, dur=1.0, noise=0.0, seed=0):
    t = np.arange(int(sr * dur)) / sr
    rng = np.random.default_rng(seed)
    return (0.3 * np.sin(2 * np.pi * f * t) + noise * rng.standard_normal(t.shape)).astype(np.float32)


def test_identical_is_zero():
    x = _tone(220, noise=0.05)
    assert mcd(x, 16000, x, 16000) < 1e-6


def test_time_stretch_is_absorbed_by_dtw():
    x = _tone(220, noise=0.01)
    slow = np.repeat(x, 2)[: len(x) * 2]  # crude 2x stretch, same spectrum
    other = _tone(1500, noise=0.01)
    assert mcd(x, 16000, slow, 16000) < mcd(x, 16000, other, 16000)


def test_resampling_inputs():
    x16 = _tone(300, sr=16000, noise=0.02)
    x24 = _tone(300, sr=24000, noise=0.02)
    assert mcd(x16, 16000, x24, 24000) < mcd(x16, 16000, _tone(2500, noise=0.02), 16000)
