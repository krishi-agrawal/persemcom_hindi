"""Real (noisy transcript -> reference) pairs for the semantic corrector (paper Sec. IV-D).

Decodes a manifest through trained Stage-2 models at a per-batch random SNR, several passes with
different channel seeds. Only the *train* split should be used, so the corrector never sees val/test text.

    python -m persemcom.corrector.make_pairs --ckpt checkpoints/awgn/stage2/best.pt --channel awgn \
        --passes 3 --out data/corrector/pairs_awgn.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from persemcom.data.datasets import ASRDataset, DurationBucketSampler, collate_asr  # noqa: E402
from persemcom.data.tokenizer import Tokenizer  # noqa: E402
from persemcom.models.channel import Channel  # noqa: E402
from persemcom.models.ctc_head import ctc_greedy_decode  # noqa: E402
from persemcom.train.s2t import build_model, spec_augment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr-config", default="configs/asr.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--channel", choices=["awgn", "rayleigh"], required=True)
    ap.add_argument("--equalizer", default="mmse")
    ap.add_argument("--manifest", default="manifests/train.jsonl")
    ap.add_argument("--snr-low", type=float, default=-10.0)
    ap.add_argument("--snr-high", type=float, default=20.0)
    ap.add_argument("--passes", type=int, default=3)
    # The ASR model has seen the train audio, so it makes ~37% fewer errors there than on test (12.8% vs
    # 20.3% WER at 20 dB). Input time masking raises the train error rate back to test-like levels;
    # 7 masks x <=4% calibrated on 1326 train utts: masks 6 -> 19.2%, 8 -> 21.7% (test 20.3%) at 20 dB.
    ap.add_argument("--time-masks", type=int, default=7)
    ap.add_argument("--time-width-ratio", type=float, default=0.04)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if Path(args.manifest).stem in ("val", "test"):
        sys.exit("corrector pairs must not be built from val/test transcripts")

    cfg = yaml.safe_load(open(args.asr_config, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer(cfg["data"]["tokenizer"])
    model = build_model(cfg["model"], tok.vocab_size)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.set_stage("stage2")
    model.channel = Channel(args.channel, "frame", args.equalizer)
    model.to(device).eval()

    ds = ASRDataset(args.manifest, cfg["data"]["feat_dir"], tok, cfg["data"].get("cmvn"))
    dl = DataLoader(ds, collate_fn=collate_asr, num_workers=2,
                    batch_sampler=DurationBucketSampler(ds.durations, cfg["data"]["max_batch_seconds"], shuffle=False))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n = same = 0
    with open(args.out, "w", encoding="utf-8") as f, torch.no_grad():
        for p in range(args.passes):
            gen = torch.Generator().manual_seed(1000 + p)
            torch.manual_seed(1000 + p)
            for b in dl:
                snr = torch.empty(()).uniform_(args.snr_low, args.snr_high, generator=gen).item()
                feats, lens = b["feats"].to(device), b["feat_lens"].to(device)
                if args.time_masks:
                    feats = spec_augment(feats, lens, freq_masks=0, time_masks=args.time_masks,
                                         time_width_ratio=args.time_width_ratio)
                with torch.autocast(device.type, enabled=device.type == "cuda"):
                    out = model(feats, lens, snr)
                ids = ctc_greedy_decode(out["log_probs"], out["lengths"], model.blank_id)
                for uid, ref, hyp_ids in zip(b["ids"], b["texts"], ids):
                    hyp = tok.decode(hyp_ids)
                    f.write(json.dumps({"id": uid, "src": hyp, "tgt": ref, "channel": args.channel,
                                        "snr_db": round(snr, 2), "pass": p}, ensure_ascii=False) + "\n")
                    n += 1
                    same += hyp == ref
            print(f"pass {p + 1}/{args.passes}: {n} pairs, {same / n:.1%} already correct", flush=True)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
