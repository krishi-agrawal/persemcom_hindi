"""Storage and inference-time complexity per module (paper Table III).

    python scripts/11_complexity.py --channel awgn

Parameters and fp32 size of every module, and the mean per-utterance latency of each processing step at
batch size 1 on the speech-to-speech test subset (CUDA-synchronized wall clock, after warm-up).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from persemcom.data.features import AUDIO_SR, asr_fbank  # noqa: E402
from persemcom.evaluation import load_audio, s2s_subset, write_csv  # noqa: E402
from persemcom.kb.build_kb import load_speaker_encoder  # noqa: E402
from persemcom.models.ctc_head import ctc_greedy_decode  # noqa: E402
from persemcom.models.persemcom_s2t import length_mask  # noqa: E402
from persemcom.pipeline import PerSemComPipeline  # noqa: E402


def n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


class Timer:
    def __init__(self, device):
        self.device, self.times = device, {}

    def lap(self, name, t0):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        self.times.setdefault(name, []).append(1000 * (t1 - t0))
        return t1


@torch.no_grad()
def run_once(pipe, wav, speaker_id, snr, timer=None):
    dev, s2t = pipe.device, pipe.s2t
    t = time.perf_counter()
    lap = timer.lap if timer else (lambda name, t0: time.perf_counter())
    feats = ((asr_fbank(torch.from_numpy(wav)).to(dev) - pipe.mean) / pipe.std)[None]
    lens = torch.tensor([feats.shape[1]], device=dev)
    t = lap("tx: fbank", t)
    with torch.autocast(dev.type, enabled=dev.type == "cuda"):
        F, lens = s2t.semantic_encoder(feats, lens)
        t = lap("tx: semantic encoder", t)
        X = s2t.channel_encoder(F, length_mask(lens, F.shape[1]))
        t = lap("tx: channel encoder", t)
        Y = s2t.channel(X, snr)
        t = lap("channel", t)
        log_probs = s2t.semantic_decoder(s2t.channel_decoder(Y))
    text = pipe.tok.decode(ctc_greedy_decode(log_probs, lens, s2t.blank_id)[0])
    t = lap("rx: channel + semantic decoder", t)
    text = pipe.correct([text])[0]
    t = lap("rx: semantic corrector", t)
    tokens = torch.tensor(pipe.vocab.encode(text), device=dev)[None]
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        mel, mel_lens, _ = pipe.tts.infer(tokens, torch.tensor([tokens.shape[1]], device=dev),
                                          pipe.kb[speaker_id][None].to(dev))
    t = lap("rx: synthesizer (Tacotron 2)", t)
    pipe.vocos.decode(mel[:, :, :mel_lens[0]].float())
    lap("rx: vocoder (Vocos)", t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", choices=["awgn", "rayleigh"], default="awgn")
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--corrector", default="checkpoints/corrector/best")
    ap.add_argument("--tts-ckpt", default="checkpoints/tts_finetune/best.pt")
    ap.add_argument("--manifest", default="manifests/test.jsonl")
    ap.add_argument("--utts", type=int, default=50)
    ap.add_argument("--out", default="results/complexity.csv")
    args = ap.parse_args()

    pipe = PerSemComPipeline(f"checkpoints/{args.channel}/stage2/best.pt", args.channel,
                             corrector=args.corrector, tts_ckpt=args.tts_ckpt)
    spk_enc = load_speaker_encoder("checkpoints/speaker/best.pt", pipe.device)
    modules = {
        "tx: semantic encoder": pipe.s2t.semantic_encoder,
        "tx: channel encoder": pipe.s2t.channel_encoder,
        "rx: channel + semantic decoder": torch.nn.ModuleList([pipe.s2t.channel_decoder, pipe.s2t.semantic_decoder]),
        "rx: semantic corrector": pipe.corrector.model,
        "rx: synthesizer (Tacotron 2)": pipe.tts,
        "rx: vocoder (Vocos)": pipe.vocos,
        "offline: speaker encoder (GE2E, builds the KB)": spk_enc,
    }

    subset = s2s_subset(args.manifest, n=args.utts)
    audio = [load_audio(r) for r in subset]
    torch.manual_seed(0)
    for wav, r in list(zip(audio, subset))[:3]:  # warm-up: CUDA kernels, cuDNN autotune, generate() caches
        run_once(pipe, wav, r["speaker_id"], args.snr)
    timer = Timer(pipe.device)
    for wav, r in zip(audio, subset):
        run_once(pipe, wav, r["speaker_id"], args.snr, timer)

    rows = []
    for name in ["tx: fbank", *list(modules)[:2], "channel", *list(modules)[2:]]:
        ms = timer.times.get(name, [])
        p = n_params(modules[name]) if name in modules else 0
        rows.append({"module": name, "params_m": p / 1e6, "size_mb_fp32": p * 4 / 1e6,
                     "latency_ms_mean": float(np.mean(ms)) if ms else "",
                     "latency_ms_std": float(np.std(ms)) if ms else ""})
        print(rows[-1], flush=True)
    dur = float(np.mean([len(w) / AUDIO_SR for w in audio]))
    print(f"mean utterance duration {dur:.2f} s; kb {len(pipe.kb)} speakers x 256 floats")
    write_csv(rows, args.out)


if __name__ == "__main__":
    main()
