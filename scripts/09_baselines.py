"""Baselines over AWGN / Rayleigh (paper Sec. V-B): AMR-WB + LDPC (conventional) and Txt2Vid-style
(clean ASR -> compressed text -> LDPC -> TTS).

    python scripts/09_baselines.py --channel awgn --systems amrwb txt2vid
    python scripts/09_baselines.py --channel rayleigh --systems amrwb --judge-device cpu   # while TTS trains

Speech-to-speech scoring uses the same 200-utterance subset, judges and WAV layout as scripts/10_eval_all.py.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from persemcom.baselines import amrwb, txt2vid  # noqa: E402
from persemcom.baselines.digital_link import DigitalLink  # noqa: E402
from persemcom.data.datasets import ASRDataset, DurationBucketSampler, collate_asr  # noqa: E402
from persemcom.data.features import AUDIO_SR  # noqa: E402
from persemcom.data.tokenizer import Tokenizer  # noqa: E402
from persemcom.evaluation import SpeechJudge, load_audio, s2s_subset, write_csv  # noqa: E402
from persemcom.metrics.semsim import similarity_scores  # noqa: E402
from persemcom.metrics.wer import cer, wer  # noqa: E402
from persemcom.models.ctc_head import ctc_greedy_decode  # noqa: E402
from persemcom.train.s2t import build_model  # noqa: E402


@torch.no_grad()
def clean_transcripts(cfg, manifest, device):
    """Transmitter-side ASR for Txt2Vid: the clean PerSemCom ASR model, no channel."""
    tok = Tokenizer(cfg["data"]["tokenizer"])
    m = build_model(cfg["model"], tok.vocab_size)
    m.load_state_dict(torch.load("checkpoints/asr/best.pt", map_location="cpu")["model"])
    m.set_stage("pretrain")
    m.to(device).eval()
    ds = ASRDataset(manifest, cfg["data"]["feat_dir"], tok, cfg["data"]["cmvn"])
    dl = DataLoader(ds, collate_fn=collate_asr,
                    batch_sampler=DurationBucketSampler(ds.durations, cfg["data"]["max_batch_seconds"], shuffle=False))
    out = {}
    for b in dl:
        lp = m(b["feats"].to(device), b["feat_lens"].to(device))
        for i, ids in zip(b["ids"], ctc_greedy_decode(lp["log_probs"], lp["lengths"], m.blank_id)):
            out[i] = tok.decode(ids)
    return {r["id"]: (r, out[r["id"]]) for r in ds.rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", choices=["awgn", "rayleigh"], required=True)
    ap.add_argument("--systems", nargs="+", choices=["amrwb", "txt2vid"], default=["amrwb", "txt2vid"])
    ap.add_argument("--asr-config", default="configs/asr.yaml")
    ap.add_argument("--manifest", default="manifests/test.jsonl")
    ap.add_argument("--snrs", type=float, nargs="+", default=[-10, -6, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ap.add_argument("--s2s-snrs", type=float, nargs="+", default=[-10, -6, -2, 2, 10, 20])
    ap.add_argument("--s2s-utts", type=int, default=200)
    ap.add_argument("--tts-ckpt", default="checkpoints/tts/best.pt")
    ap.add_argument("--judge-device", default=None, help="cpu to stay off the GPU while TTS trains")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.asr_config, encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    link = DigitalLink(args.channel)
    subset = s2s_subset(args.manifest, args.s2s_utts, seed=args.seed)
    ref_audio = {r["id"]: load_audio(r) for r in subset}
    judge = SpeechJudge(args.asr_config, device=args.judge_device)
    ref_emb = {i: judge.embed(w) for i, w in ref_audio.items()}
    wav_root = out_dir / "wavs" / args.channel

    if "amrwb" in args.systems:
        rows = []
        for snr in args.s2s_snrs:
            wavs, n_sym, bad = [], [], []
            for k, r in enumerate(subset):
                w, ns, fb = amrwb.transmit(ref_audio[r["id"]], AUDIO_SR, link, snr, seed=args.seed + k)
                wavs.append(w)
                n_sym.append(ns)
                bad.append(fb)
            d = wav_root / f"amrwb_{snr:g}dB"
            d.mkdir(parents=True, exist_ok=True)
            for r, w in zip(subset, wavs):
                sf.write(d / f"{r['id']}.wav", w, 16000)
            row = {"system": "amrwb_ldpc", "channel": args.channel, "snr_db": snr,
                   **judge.score(subset, ref_audio, wavs, 16000, ref_emb),
                   "lost_frame_rate": float(np.mean(bad)), "symbols_per_utt": float(np.mean(n_sym))}
            rows.append(row)
            print(row, flush=True)
        write_csv(rows, out_dir / f"baseline_amrwb_{args.channel}.csv")

    if "txt2vid" in args.systems:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tx = clean_transcripts(cfg, args.manifest, device)
        rows, received = [], {}
        for snr in args.snrs:
            refs, hyps, n_sym = [], [], []
            for k, (uid, (r, text)) in enumerate(sorted(tx.items())):
                rx, ns = txt2vid.transmit_text(text, link, snr, seed=args.seed + k)
                refs.append(r["text"])
                hyps.append(rx)
                n_sym.append(ns)
                received[(snr, uid)] = rx
            row = {"system": "txt2vid", "channel": args.channel, "snr_db": snr, "wer": wer(refs, hyps),
                   "cer": cer(refs, hyps), "sim": float(similarity_scores(refs, hyps).mean()),
                   "lost_utt_rate": float(np.mean([h == "" for h in hyps])),
                   "symbols_per_utt": float(np.mean(n_sym)), "n_utts": len(refs)}
            rows.append(row)
            print(row, flush=True)
        write_csv(rows, out_dir / f"baseline_txt2vid_s2t_{args.channel}.csv")
        with open(out_dir / f"baseline_txt2vid_received_{args.channel}.jsonl", "w", encoding="utf-8") as f:
            for (snr, uid), t in received.items():
                f.write(json.dumps({"snr_db": snr, "id": uid, "text": t}, ensure_ascii=False) + "\n")

        if Path(args.tts_ckpt).exists():
            from persemcom.pipeline import PerSemComPipeline
            pipe = PerSemComPipeline(f"checkpoints/{args.channel}/stage2/best.pt", args.channel,
                                     args.asr_config, tts_ckpt=args.tts_ckpt)
            rows = []
            for snr in args.s2s_snrs:
                texts = [received[(snr, r["id"])] for r in subset]
                keep = [i for i, t in enumerate(texts) if t]
                wavs = [np.zeros(0, np.float32)] * len(subset)
                if keep:
                    synth = pipe.synthesize([texts[i] for i in keep], [subset[i]["speaker_id"] for i in keep])
                    for i, w in zip(keep, synth):
                        wavs[i] = w
                d = wav_root / f"txt2vid_{snr:g}dB"
                d.mkdir(parents=True, exist_ok=True)
                for r, w in zip(subset, wavs):
                    sf.write(d / f"{r['id']}.wav", w if len(w) else np.zeros(AUDIO_SR // 10, np.float32), AUDIO_SR)
                row = {"system": "txt2vid", "channel": args.channel, "snr_db": snr,
                       **judge.score(subset, ref_audio, wavs, AUDIO_SR, ref_emb),
                       "lost_utt_rate": 1 - len(keep) / len(subset)}
                rows.append(row)
                print(row, flush=True)
            write_csv(rows, out_dir / f"baseline_txt2vid_s2s_{args.channel}.csv")
        else:
            print(f"no TTS checkpoint at {args.tts_ckpt}; skipped Txt2Vid speech-to-speech")


if __name__ == "__main__":
    main()
