"""Evaluate a speech-to-text checkpoint (WER/CER) on a manifest, clean or over a channel at several SNRs.

    python scripts/eval_s2t.py --ckpt checkpoints/asr/best.pt --channel none
    python scripts/eval_s2t.py --ckpt checkpoints/awgn/stage2/best.pt --channel awgn --snrs 0 5 10 15 20
    python scripts/eval_s2t.py --ckpt checkpoints/awgn/stage2/best.pt --channel rayleigh   # mismatch test

Writes one CSV row per SNR and, with --hyps, the decoded transcripts (corrector training/eval input).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from persemcom.data.datasets import ASRDataset, DurationBucketSampler, collate_asr  # noqa: E402
from persemcom.data.tokenizer import Tokenizer  # noqa: E402
from persemcom.metrics.wer import cer, wer  # noqa: E402
from persemcom.models.channel import Channel  # noqa: E402
from persemcom.models.ctc_head import ctc_greedy_decode  # noqa: E402
from persemcom.train.s2t import build_model  # noqa: E402


@torch.no_grad()
def decode_split(model, dl, tok, device, snr_db, seed):
    torch.manual_seed(seed)  # reproducible channel noise / fading
    out_rows = []
    for b in dl:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out = model(b["feats"].to(device), b["feat_lens"].to(device), snr_db)
        for uid, ref, ids in zip(b["ids"], b["texts"], ctc_greedy_decode(out["log_probs"], out["lengths"], model.blank_id)):
            out_rows.append({"id": uid, "ref": ref, "hyp": tok.decode(ids)})
    return out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr-config", default="configs/asr.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="manifests/test.jsonl")
    ap.add_argument("--channel", choices=["none", "awgn", "rayleigh"], default="none")
    ap.add_argument("--fading-block", default="frame")
    ap.add_argument("--equalizer", default="mmse")
    ap.add_argument("--snrs", type=float, nargs="+", default=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ap.add_argument("--out", default=None, help="CSV path (default: results/<ckpt-stem>_<channel>.csv)")
    ap.add_argument("--hyps", default=None, help="optional JSONL of per-utterance ref/hyp/snr")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.asr_config, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer(cfg["data"]["tokenizer"])
    model = build_model(cfg["model"], tok.vocab_size)
    ck = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    if args.channel == "none":
        model.set_stage("pretrain")  # bypass channel codec entirely
        snrs = [None]
    else:
        model.set_stage("stage2")
        model.channel = Channel(args.channel, args.fading_block, args.equalizer)  # allows mismatched testing
        snrs = args.snrs

    ds = ASRDataset(args.manifest, cfg["data"]["feat_dir"], tok, cfg["data"].get("cmvn"))
    dl = DataLoader(ds, collate_fn=collate_asr,
                    batch_sampler=DurationBucketSampler(ds.durations, cfg["data"]["max_batch_seconds"], shuffle=False))

    out_csv = Path(args.out or f"results/{Path(args.ckpt).parent.as_posix().replace('/', '_')}_{args.channel}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    hyp_f = open(args.hyps, "w", encoding="utf-8") if args.hyps else None
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ckpt", "trained_stage", "channel", "snr_db", "wer", "cer", "n_utts"])
        for snr in snrs:
            rows = decode_split(model, dl, tok, device, snr, args.seed)
            refs, hyps = [r["ref"] for r in rows], [r["hyp"] for r in rows]
            res = (wer(refs, hyps), cer(refs, hyps))
            w.writerow([args.ckpt, ck.get("stage"), args.channel, "" if snr is None else snr,
                        f"{res[0]:.4f}", f"{res[1]:.4f}", len(rows)])
            print(f"{args.channel:8s} snr={snr!s:>5}  WER {res[0]:.4f}  CER {res[1]:.4f}  ({len(rows)} utts)", flush=True)
            if hyp_f:
                for r in rows:
                    hyp_f.write(json.dumps({**r, "channel": args.channel, "snr_db": snr}, ensure_ascii=False) + "\n")
    if hyp_f:
        hyp_f.close()
    print("wrote", out_csv)


if __name__ == "__main__":
    main()
