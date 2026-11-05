"""Shared evaluation helpers for scripts/09_baselines.py and scripts/10_eval_all.py, so PerSemCom and the
baselines are scored by exactly the same judges on exactly the same utterances."""
import csv
import random
from typing import List

import numpy as np
import soundfile as sf
import torch
import yaml

from persemcom.data.features import AUDIO_SR, asr_fbank
from persemcom.data.manifest import read_manifest
from persemcom.data.tokenizer import Tokenizer
from persemcom.kb.build_kb import load_speaker_encoder
from persemcom.metrics.mcd import mcd
from persemcom.metrics.wer import wer
from persemcom.models.ctc_head import ctc_greedy_decode
from persemcom.train.s2t import build_model


def s2s_subset(manifest: str, n: int = 200, max_duration: float = 10.0, seed: int = 0) -> List[dict]:
    """Fixed test subset for speech-to-speech scoring and the MOS study."""
    pool = [r for r in read_manifest(manifest) if r["duration"] <= max_duration]
    return sorted(random.Random(seed).sample(pool, min(n, len(pool))), key=lambda r: r["id"])


def load_audio(row: dict) -> np.ndarray:
    wav, sr = sf.read(row["path"], dtype="float32")
    assert sr == AUDIO_SR, (row["path"], sr)
    return wav


class SpeechJudge:
    """Objective scores for (re)generated speech against the original utterance:
    MCD, speaker similarity (GE2E cosine) and intelligibility (WER of the clean ASR model on it)."""

    def __init__(self, asr_config: str = "configs/asr.yaml", asr_ckpt: str = "checkpoints/asr/best.pt",
                 speaker_ckpt: str = "checkpoints/speaker/best.pt", device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        cfg = yaml.safe_load(open(asr_config, encoding="utf-8"))
        self.spk = load_speaker_encoder(speaker_ckpt, self.device)
        stats = np.load(cfg["data"]["cmvn"])
        self.mean, self.std = torch.from_numpy(stats["mean"]), torch.from_numpy(stats["std"])
        self.tok = Tokenizer(cfg["data"]["tokenizer"])
        self.asr = build_model(cfg["model"], self.tok.vocab_size)
        self.asr.load_state_dict(torch.load(asr_ckpt, map_location="cpu")["model"])
        self.asr.set_stage("pretrain")
        self.asr.to(self.device).eval()

    def _fbank(self, wav: np.ndarray, sr: int) -> torch.Tensor:
        if sr != AUDIO_SR:
            from persemcom.data.features import resample
            wav = resample(torch.from_numpy(wav), sr, AUDIO_SR).numpy()
        return (asr_fbank(torch.from_numpy(wav)) - self.mean) / self.std

    @torch.no_grad()
    def embed(self, wav: np.ndarray, sr: int = AUDIO_SR) -> torch.Tensor:
        return self.spk.embed_utterance(self._fbank(wav, sr).to(self.device)).cpu()

    @torch.no_grad()
    def transcribe(self, wav: np.ndarray, sr: int = AUDIO_SR) -> str:
        f = self._fbank(wav, sr)[None].to(self.device)
        out = self.asr(f, torch.tensor([f.shape[1]], device=self.device))
        return self.tok.decode(ctc_greedy_decode(out["log_probs"], out["lengths"], self.asr.blank_id)[0])

    def score(self, refs: List[dict], ref_audio: dict, hyp_wavs: List[np.ndarray], hyp_sr: int,
              ref_emb: dict) -> dict:
        mcds, sims, hyps = [], [], []
        for r, w in zip(refs, hyp_wavs):
            # < 0.1 s (e.g. the synthesizer stopping at once on an empty received transcript) is scored as a
            # failed transmission; shorter than one 25 ms window it would also crash the feature extraction
            if len(w) < hyp_sr // 10 or not np.isfinite(w).all() or np.abs(w).max() == 0:
                mcds.append(float("nan"))
                sims.append(0.0)
                hyps.append("")
                continue
            mcds.append(mcd(ref_audio[r["id"]], AUDIO_SR, w, hyp_sr))
            sims.append(float(torch.nn.functional.cosine_similarity(self.embed(w, hyp_sr), ref_emb[r["id"]], dim=0)))
            hyps.append(self.transcribe(w, hyp_sr))
        texts = [r["text"] for r in refs]
        return {"mcd": float(np.nanmean(mcds)) if np.isfinite(mcds).any() else float("nan"),
                "speaker_cos": float(np.mean(sims)), "asr_wer_of_speech": wer(texts, hyps), "n_utts": len(refs)}


def write_csv(rows: List[dict], path) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)
