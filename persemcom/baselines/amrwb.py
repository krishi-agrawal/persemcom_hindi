"""Conventional baseline (paper Sec. V-B.6): AMR-WB (ACELP, 3GPP TS 26.190) source coding at 23.85 kbps
+ the LDPC/QPSK digital link.

Only the speech payload bits travel over the channel; the storage-format header and per-frame TOC bytes
are assumed delivered intact (as signalling would be). Frames overlapping a failed LDPC block are passed
to the decoder as SPEECH_LOST (frame type 14), which makes libopencore_amrwb run its lost-frame
concealment, as a frame CRC would in a real system. (Clearing the TOC quality bit instead is ignored by
the decoder: an all-bad-flagged file decoded to the same RMS as the clean one.)
"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from persemcom.baselines.digital_link import DigitalLink

FFMPEG = Path("data/tools/ffmpeg/ffmpeg-9.0.2-full_build/bin/ffmpeg.exe")
MAGIC = b"#!AMR-WB\n"
# speech bits per frame for frame types 0-8 (6.6 ... 23.85 kbps) and 9 (SID); 15 = no data
FRAME_BITS = {0: 132, 1: 177, 2: 253, 3: 285, 4: 317, 5: 365, 6: 397, 7: 461, 8: 477, 9: 40, 14: 0, 15: 0}
SPEECH_LOST = 14


def _ffmpeg(*args):
    exe = FFMPEG if FFMPEG.exists() else Path("ffmpeg")
    subprocess.run([str(exe), "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def encode(wav: np.ndarray, sr: int, bitrate: str = "23.85k") -> bytes:
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d, "in.wav"), Path(d, "out.awb")
        sf.write(src, wav, sr)
        _ffmpeg("-i", str(src), "-ar", "16000", "-ac", "1", "-c:a", "libvo_amrwbenc", "-b:a", bitrate,
                "-f", "amr", str(dst))
        return dst.read_bytes()


def decode(awb: bytes) -> np.ndarray:
    """-> 16 kHz float waveform."""
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d, "in.awb"), Path(d, "out.wav")
        src.write_bytes(awb)
        _ffmpeg("-c:a", "libopencore_amrwb", "-i", str(src), "-ar", "16000", "-ac", "1", str(dst))
        wav, _ = sf.read(dst, dtype="float32")
        return wav


def parse(awb: bytes):
    """-> list of (toc_byte, payload_bits ndarray uint8)."""
    if not awb.startswith(MAGIC):
        raise ValueError("not an AMR-WB storage file")
    frames, pos = [], len(MAGIC)
    while pos < len(awb):
        toc = awb[pos]
        ft = (toc >> 3) & 0x0F
        n_bits = FRAME_BITS.get(ft)
        if n_bits is None:
            raise ValueError(f"unsupported frame type {ft}")
        n_bytes = (n_bits + 7) // 8
        payload = np.unpackbits(np.frombuffer(awb[pos + 1: pos + 1 + n_bytes], dtype=np.uint8))[:n_bits]
        frames.append((toc, payload))
        pos += 1 + n_bytes
    return frames


def build(frames) -> bytes:
    out = bytearray(MAGIC)
    for toc, bits in frames:
        out.append(toc)
        n_bytes = (len(bits) + 7) // 8
        padded = np.zeros(n_bytes * 8, dtype=np.uint8)
        padded[: len(bits)] = bits
        out += np.packbits(padded).tobytes()
    return bytes(out)


def transmit(wav: np.ndarray, sr: int, link: DigitalLink, snr_db: float, seed: int = 0):
    """Returns (decoded 16 kHz waveform, complex symbols used, fraction of frames flagged bad)."""
    frames = parse(encode(wav, sr))
    sizes = [len(b) for _, b in frames]
    bits = np.concatenate([b for _, b in frames]) if sum(sizes) else np.zeros(0, np.uint8)
    res = link.send(bits, snr_db, seed=seed)
    bad_block = ~res.block_ok
    out_frames, start, n_bad = [], 0, 0
    for (toc, _), n in zip(frames, sizes):
        b = res.bits[start:start + n]
        if n:
            blocks = range(start // link.k, (start + n - 1) // link.k + 1)
            if any(bad_block[i] for i in blocks):
                toc, b = SPEECH_LOST << 3, np.zeros(0, np.uint8)  # decoder conceals this frame
                n_bad += 1
        out_frames.append((toc, b))
        start += n
    return decode(build(out_frames)), res.n_symbols, n_bad / max(1, sum(1 for n in sizes if n))
