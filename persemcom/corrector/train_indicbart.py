"""Fine-tune IndicBART as the semantic corrector on (noisy transcript -> reference) pairs.

    python -m persemcom.corrector.train_indicbart --pairs data/corrector/pairs_awgn.jsonl \
        data/corrector/pairs_rayleigh.jsonl --out checkpoints/corrector

Dev pairs are held out *by utterance id*, so a sentence never appears on both sides.
Checkpoint selection: dev WER after correction (Eq. 7).
"""
import argparse
import json
import random
import sys
import time
import zlib
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from persemcom.corrector.model import BASE_MODEL, Corrector  # noqa: E402
from persemcom.metrics.wer import wer  # noqa: E402


def load_pairs(paths, dev_frac, max_identity_frac, seed):
    rows = [json.loads(line) for p in paths for line in open(p, encoding="utf-8") if line.strip()]
    rows = [r for r in rows if r["tgt"].strip()]
    is_dev = lambda r: zlib.crc32(r["id"].encode()) % 1000 < dev_frac * 1000  # noqa: E731
    train = [r for r in rows if not is_dev(r)]
    dev = [r for r in rows if is_dev(r)]
    # keep some already-correct pairs so the model learns to copy, but don't let them dominate
    rng = random.Random(seed)
    same = [r for r in train if r["src"] == r["tgt"]]
    diff = [r for r in train if r["src"] != r["tgt"]]
    keep = int(max_identity_frac / (1 - max_identity_frac) * len(diff))
    rng.shuffle(same)
    train = diff + same[:keep]
    rng.shuffle(train)
    return train, dev, len(same)


def batches(rows, batch_size):
    # sort by length within large shuffled pools -> little padding, still random
    pool = 100 * batch_size
    for i in range(0, len(rows), pool):
        chunk = sorted(rows[i:i + pool], key=lambda r: len(r["src"]))
        bs = [chunk[j:j + batch_size] for j in range(0, len(chunk), batch_size)]
        random.shuffle(bs)
        yield from bs


def evaluate(corr: Corrector, dev, n, num_beams=1):
    sub = dev[:n]
    hyps = corr.correct([r["src"] for r in sub], num_beams=num_beams)
    refs = [r["tgt"] for r in sub]
    return wer(refs, [r["src"] for r in sub]), wer(refs, hyps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--out", default="checkpoints/corrector")
    ap.add_argument("--init", default=BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--dev-frac", type=float, default=0.02)
    ap.add_argument("--max-identity-frac", type=float, default=0.3)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-n", type=int, default=1500)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train, dev, n_same = load_pairs(args.pairs, args.dev_frac, args.max_identity_frac, args.seed)
    print(f"train pairs {len(train)} (identity pairs available {n_same}) | dev pairs {len(dev)}", flush=True)

    corr = Corrector(args.init)
    model = corr.model
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(out_dir / "tb"))

    steps_per_epoch = (len(train) + args.batch_size - 1) // args.batch_size
    total = args.max_steps or steps_per_epoch * args.epochs
    no_decay = ("bias", "layer_norm.weight", "layernorm_embedding.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=args.lr)
    sched = get_linear_schedule_with_warmup(opt, args.warmup, total)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=args.label_smoothing)

    before, after = evaluate(corr, dev, args.eval_n)
    print(f"[step 0] dev WER: input {before:.4f} -> uncorrected-model output {after:.4f}", flush=True)
    best, step = float("inf"), 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for b in batches(train, args.batch_size):
            model.train()
            enc = corr.encode_src([r["src"] for r in b])
            tgt = corr.encode_tgt([r["tgt"] for r in b])
            dec_in, labels = tgt[:, :-1], tgt[:, 1:].clone()
            labels[labels == corr.pad] = -100
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**enc, decoder_input_ids=dec_in,
                               decoder_attention_mask=(dec_in != corr.pad).long()).logits
            loss = loss_fn(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                writer.add_scalar("corrector/loss", loss.item(), step)
                print(f"ep {epoch} step {step}/{total} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            if step % args.eval_every == 0 or step == total:
                model.eval()
                before, after = evaluate(corr, dev, args.eval_n)
                writer.add_scalar("corrector/dev_wer_after", after, step)
                print(f"[step {step}] dev WER: input {before:.4f} -> corrected {after:.4f}", flush=True)
                if after < best:
                    best = after
                    model.save_pretrained(out_dir / "best")
                    corr.tok.save_pretrained(out_dir / "best")
            if step >= total:
                break
        if step >= total:
            break
    print(f"best dev WER after correction: {best:.4f} (saved {out_dir / 'best'})")
    writer.close()


if __name__ == "__main__":
    main()
