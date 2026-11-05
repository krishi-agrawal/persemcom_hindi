"""Full evaluation of PerSemCom for one channel type (paper Sec. V-C .. V-I).

    python scripts/10_eval_all.py --channel awgn
    python scripts/10_eval_all.py --channel rayleigh

Speech-to-text (whole test split, every SNR): WER, CER, similarity score, without and with the corrector,
plus transmitted symbols per utterance.
Speech-to-speech (fixed test subset, fewer SNRs): MCD vs the original audio, speaker similarity
(GE2E cosine, generated vs original), and the clean-ASR WER of the generated speech (objective
intelligibility). WAVs go to results/wavs/ for the MOS test. Ablations: synthesis from the ground-truth
transcript (upper bound) and with a speaker-agnostic embedding (mean of the knowledge base, Fig. 15c).
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
from persemcom.data.datasets import ASRDataset, DurationBucketSampler, collate_asr  # noqa: E402
from persemcom.data.features import AUDIO_SR  # noqa: E402
from persemcom.evaluation import SpeechJudge, load_audio, s2s_subset, write_csv  # noqa: E402
from persemcom.metrics.semsim import similarity_scores  # noqa: E402
from persemcom.metrics.wer import cer, wer  # noqa: E402
from persemcom.pipeline import PerSemComPipeline  # noqa: E402


def s2t_sweep(pipe, ds, snrs, max_batch_seconds, out_dir, channel, seed):
    dl = DataLoader(ds, collate_fn=collate_asr,
                    batch_sampler=DurationBucketSampler(ds.durations, max_batch_seconds, shuffle=False))
    rows, per_utt = [], {}
    for snr in snrs:
        torch.manual_seed(seed)
        ids, refs, hyps, n_sym = [], [], [], []
        for b in dl:
            res = pipe.transmit(b["feats"], b["feat_lens"], snr)
            ids += b["ids"]
            refs += b["texts"]
            hyps += res.texts
            n_sym += res.n_symbols
        corr = pipe.correct(hyps)
        sim_raw, sim_cor = similarity_scores(refs, hyps), similarity_scores(refs, corr)
        row = {"channel": channel, "snr_db": snr,
               "wer": wer(refs, hyps), "cer": cer(refs, hyps), "sim": float(sim_raw.mean()),
               "wer_corrected": wer(refs, corr), "cer_corrected": cer(refs, corr), "sim_corrected": float(sim_cor.mean()),
               "symbols_per_utt": float(np.mean(n_sym)), "n_utts": len(refs)}
        rows.append(row)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        per_utt[snr] = {i: {"ref": r, "hyp": h, "corrected": c} for i, r, h, c in zip(ids, refs, hyps, corr)}
        with open(out_dir / f"s2t_hyps_{channel}_{snr:g}dB.jsonl", "w", encoding="utf-8") as f:
            for i, r, h, c in zip(ids, refs, hyps, corr):
                f.write(json.dumps({"id": i, "ref": r, "hyp": h, "corrected": c}, ensure_ascii=False) + "\n")
    return rows, per_utt


def s2s_eval(pipe, judge, subset, per_utt, snrs, channel, wav_dir, kb_mean):
    rows = []
    refs_audio = {r["id"]: load_audio(r) for r in subset}
    orig = wav_dir / channel / "original"
    orig.mkdir(parents=True, exist_ok=True)
    for r in subset:
        sf.write(orig / f"{r['id']}.wav", refs_audio[r["id"]], AUDIO_SR)
    orig_emb = {i: judge.embed(w) for i, w in refs_audio.items()}

    conditions = [("gt_text", None)] + [(f"{s:g}dB", s) for s in snrs] + [("gt_text_no_speaker", None)]
    for name, snr in conditions:
        if snr is None:
            texts = [r["text"] for r in subset]
        else:
            texts = [per_utt[snr][r["id"]]["corrected"] for r in subset]
        spk_override = kb_mean.expand(len(subset), -1) if name == "gt_text_no_speaker" else None
        wavs = pipe.synthesize(texts, [r["speaker_id"] for r in subset], speaker_emb=spk_override)
        out = wav_dir / channel / f"persemcom_{name}"
        out.mkdir(parents=True, exist_ok=True)
        for r, w in zip(subset, wavs):
            sf.write(out / f"{r['id']}.wav", w, AUDIO_SR)
        row = {"system": "persemcom", "channel": channel, "condition": name, "snr_db": "" if snr is None else snr,
               **judge.score(subset, refs_audio, wavs, AUDIO_SR, orig_emb),
               "input_text_wer": wer([r["text"] for r in subset], texts)}
        rows.append(row)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", choices=["awgn", "rayleigh"], required=True)
    ap.add_argument("--asr-config", default="configs/asr.yaml")
    ap.add_argument("--s2t-ckpt", default=None, help="default: checkpoints/<channel>/stage2/best.pt")
    ap.add_argument("--corrector", default="checkpoints/corrector/best")
    ap.add_argument("--tts-ckpt", default="checkpoints/tts/best.pt")
    ap.add_argument("--manifest", default="manifests/test.jsonl")
    ap.add_argument("--snrs", type=float, nargs="+", default=[-10, -6, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ap.add_argument("--s2s-snrs", type=float, nargs="+", default=[-10, -6, -2, 2, 10, 20])
    ap.add_argument("--s2s-utts", type=int, default=200)
    ap.add_argument("--s2s-max-duration", type=float, default=10.0)
    ap.add_argument("--skip-s2s", action="store_true")
    ap.add_argument("--reuse-s2t", action="store_true",
                    help="load the receiver transcripts saved by an earlier run (s2t_hyps_*.jsonl) instead of "
                         "repeating the speech-to-text sweep; s2t_full_<channel>.csv is left as it is")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.asr_config, encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    s2t_ckpt = args.s2t_ckpt or f"checkpoints/{args.channel}/stage2/best.pt"
    corrector = args.corrector if Path(args.corrector).exists() else None
    tts = None if args.skip_s2s else args.tts_ckpt
    if corrector is None:
        print("WARNING: no corrector checkpoint; corrected columns equal the raw transcripts")
    pipe = PerSemComPipeline(s2t_ckpt, args.channel, args.asr_config, corrector=corrector, tts_ckpt=tts)

    if args.reuse_s2t:
        per_utt = {}
        for snr in args.s2s_snrs:
            path = out_dir / f"s2t_hyps_{args.channel}_{snr:g}dB.jsonl"
            per_utt[snr] = {r["id"]: r for r in map(json.loads, open(path, encoding="utf-8"))}
        print(f"reused receiver transcripts for {sorted(per_utt)} dB", flush=True)
    else:
        ds = ASRDataset(args.manifest, cfg["data"]["feat_dir"], pipe.tok, cmvn=None)  # pipeline normalizes
        rows, per_utt = s2t_sweep(pipe, ds, args.snrs, cfg["data"]["max_batch_seconds"], out_dir, args.channel,
                                  args.seed)
        write_csv(rows, out_dir / f"s2t_full_{args.channel}.csv")

    if args.skip_s2s:
        return
    missing = [s for s in args.s2s_snrs if s not in per_utt]
    if missing:
        sys.exit(f"--s2s-snrs {missing} must also be in --snrs")
    subset = s2s_subset(args.manifest, args.s2s_utts, args.s2s_max_duration, seed=args.seed)
    judge = SpeechJudge(args.asr_config, device=str(pipe.device))
    kb_mean = torch.nn.functional.normalize(torch.stack(list(pipe.kb.values())).mean(0), dim=0)[None]
    rows = s2s_eval(pipe, judge, subset, per_utt, args.s2s_snrs, args.channel, out_dir / "wavs", kb_mean)
    write_csv(rows, out_dir / f"s2s_{args.channel}.csv")


if __name__ == "__main__":
    main()
