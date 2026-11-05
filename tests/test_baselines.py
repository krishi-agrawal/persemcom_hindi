import numpy as np

from persemcom.baselines import amrwb, txt2vid
from persemcom.baselines.digital_link import DigitalLink


def test_text_roundtrip_and_compression():
    t = "मैं घर जा रहा हूँ और फिर बाज़ार जाऊँगा"
    bits = txt2vid.text_to_bits(t)
    assert txt2vid.bits_to_text(bits) == t
    assert len(bits) < len(t.encode("utf-8")) * 8 + 200


def test_corrupt_text_is_dropped():
    bits = txt2vid.text_to_bits("मैं घर जा रहा हूँ")
    bits[5:40] ^= 1
    assert txt2vid.bits_to_text(bits) == ""


def test_txt2vid_link_high_and_low_snr():
    link = DigitalLink("awgn")
    t = "हम लोग बस से यात्रा करते हैं"
    assert txt2vid.transmit_text(t, link, 10.0)[0] == t
    assert txt2vid.transmit_text(t, link, -6.0)[0] == ""


def test_amrwb_parse_build_roundtrip():
    frames = [(8 << 3 | 0x04, np.random.default_rng(0).integers(0, 2, 477).astype(np.uint8)),
              (amrwb.SPEECH_LOST << 3, np.zeros(0, np.uint8)),
              (9 << 3 | 0x04, np.ones(40, np.uint8))]
    data = amrwb.build(frames)
    back = amrwb.parse(data)
    assert [t for t, _ in back] == [t for t, _ in frames]
    assert all(np.array_equal(a, b) for (_, a), (_, b) in zip(back, frames))
