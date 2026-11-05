"""End-to-end PerSemCom inference (paper Algorithm 2).

transmitter: fbank S -> f_alpha -> g_beta -> X        (32 complex symbols per encoder frame)
channel:     Y = H X + N                               (AWGN / Rayleigh, perfect CSI)
receiver:    g^-1_chi -> m^-1_delta -> T_hat -> h -> T_C -> G_S(T_C, I(si)) -> Vocos -> waveform
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import yaml

from persemcom.data.tokenizer import Tokenizer
from persemcom.data.tts_text import GraphemeVocab
from persemcom.kb.build_kb import load_kb
from persemcom.models.channel import Channel
from persemcom.models.ctc_head import ctc_greedy_decode
from persemcom.models.tacotron2_spk import SpeakerTacotron2
from persemcom.train.s2t import build_model


@dataclass
class TransmitResult:
    texts: List[str]          # T_hat, decoded at the receiver
    n_symbols: List[int]      # complex channel symbols sent per utterance


class PerSemComPipeline:
    def __init__(self, s2t_ckpt: str, channel: str, asr_config: str = "configs/asr.yaml",
                 corrector: Optional[str] = None, tts_ckpt: Optional[str] = None,
                 speaker_kb: str = "artifacts/speaker_kb.npz", equalizer: str = "mmse", device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        cfg = yaml.safe_load(open(asr_config, encoding="utf-8"))
        self.tok = Tokenizer(cfg["data"]["tokenizer"])
        stats = np.load(cfg["data"]["cmvn"])
        self.mean = torch.from_numpy(stats["mean"]).to(self.device)
        self.std = torch.from_numpy(stats["std"]).to(self.device)
        self.s2t = build_model(cfg["model"], self.tok.vocab_size)
        self.s2t.load_state_dict(torch.load(s2t_ckpt, map_location="cpu")["model"])
        self.s2t.set_stage("stage2")
        self.s2t.channel = Channel(channel, "frame", equalizer)
        self.s2t.to(self.device).eval()
        self.n_symbols_per_frame = self.s2t.channel_encoder.n_symbols

        self.corrector = None
        if corrector:
            from persemcom.corrector.model import Corrector
            self.corrector = Corrector(corrector, device=str(self.device))

        self.tts = self.vocab = self.kb = self.vocos = None
        if tts_ckpt:
            ck = torch.load(tts_ckpt, map_location="cpu")
            self.vocab = GraphemeVocab(ck["vocab"])
            self.tts = SpeakerTacotron2(n_symbol=len(self.vocab), **ck["config"])
            self.tts.load_state_dict(ck["model"])
            self.tts.to(self.device).eval()
            self.kb = load_kb(speaker_kb)
            from vocos import Vocos
            self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(self.device).eval()

    @torch.no_grad()
    def transmit(self, feats: torch.Tensor, feat_lens: torch.Tensor, snr_db: float) -> TransmitResult:
        """feats: raw (un-normalized) fbank (B, T, 80) as cached by scripts/03_features.py."""
        feats = (feats.to(self.device) - self.mean) / self.std
        feat_lens = feat_lens.to(self.device)
        with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
            out = self.s2t(feats, feat_lens, snr_db)
        ids = ctc_greedy_decode(out["log_probs"], out["lengths"], self.s2t.blank_id)
        n_sym = (out["lengths"] * self.n_symbols_per_frame).tolist()
        return TransmitResult([self.tok.decode(x) for x in ids], n_sym)

    def correct(self, texts: List[str], num_beams: int = 4) -> List[str]:
        if self.corrector is None:
            return list(texts)
        return self.corrector.correct(list(texts), num_beams=num_beams)

    @torch.no_grad()
    def synthesize(self, texts: List[str], speaker_ids: List[str], speaker_emb: Optional[torch.Tensor] = None,
                   batch_size: int = 16) -> List[np.ndarray]:
        """24 kHz waveforms in the style of each speaker (KB lookup by identity, paper Fig. 6).
        speaker_emb overrides the KB (e.g. zeros for the 'w/o speaker embedding' ablation)."""
        if self.tts is None:
            raise RuntimeError("pipeline was built without a TTS checkpoint")
        wavs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            toks = [torch.tensor(self.vocab.encode(t)) for t in chunk]
            lens = torch.tensor([len(t) for t in toks], device=self.device)
            tokens = torch.nn.utils.rnn.pad_sequence(toks, batch_first=True).to(self.device)
            spk = (speaker_emb[i:i + batch_size] if speaker_emb is not None
                   else torch.stack([self.kb[s] for s in speaker_ids[i:i + batch_size]])).to(self.device)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                mel, mel_lens, _ = self.tts.infer(tokens, lens, spk)
            for m, n in zip(mel, mel_lens):
                wavs.append(self.vocos.decode(m[None, :, :n].float())[0].cpu().numpy())
        return wavs
