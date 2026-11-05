"""Mel-cepstral distortion (paper Eq. 9) between reference and reconstructed speech.

Cepstra are the DCT of the natural-log mel *amplitude* spectrum (16 kHz, 25 ms / 10 ms, 40 mel bands),
coefficients 1-13 (c0, i.e. loudness, excluded). The constant 10*sqrt(2)/ln 10 in Eq. 9 converts
natural-log cepstral distance to dB. Pitfalls measured on 5 test clips (AMR-WB 23.85 kbps decode vs
original): librosa's dB-scaled MFCCs gave 158 dB; log power gave 24.3 dB; log amplitude gives 12.6 dB,
vs 41.7 dB for an unrelated utterance by another speaker. Excluding silent frames changed it by < 0.2 dB.
Absolute values depend on filterbank/cepstrum conventions (SPTK mel-cepstra give smaller numbers), so
compare systems against each other only.
Frames are DTW-aligned because regenerated speech never has the reference's timing;
MCD = K * mean over aligned frames of ||c - c_hat||_2.
"""
import math

import librosa
import numpy as np
from scipy.fft import dct

K = 10.0 * math.sqrt(2.0) / math.log(10.0)


def mel_cepstrum(wav: np.ndarray, sr: int, n_ceps: int = 13, n_mels: int = 40) -> np.ndarray:
    if sr != 16000:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
    S = librosa.feature.melspectrogram(y=wav, sr=16000, n_fft=400, hop_length=160, win_length=400,
                                       n_mels=n_mels, power=1.0)
    c = dct(np.log(S + 1e-10), type=2, axis=0, norm="ortho")
    return c[1:n_ceps + 1]  # (n_ceps, frames)


def mcd(ref_wav: np.ndarray, ref_sr: int, hyp_wav: np.ndarray, hyp_sr: int) -> float:
    a, b = mel_cepstrum(ref_wav, ref_sr), mel_cepstrum(hyp_wav, hyp_sr)
    if a.shape[1] == 0 or b.shape[1] == 0:
        return float("nan")
    _, path = librosa.sequence.dtw(X=a, Y=b, metric="euclidean")
    d = np.linalg.norm(a[:, path[:, 0]] - b[:, path[:, 1]], axis=0)
    return float(K * d.mean())
