"""Corrected train transcripts T_C for the synthesizer fine-tune (paper Algorithm 1, lines 12-15).

For every train utterance usable by the TTS (<= max duration), take one noisy receiver transcript from
the corrector pair files (alternating AWGN / Rayleigh, first pass = a random SNR per batch), run the
semantic corrector on it and write {"id", "text"} lines for scripts/08_train_tts.py --finetune.
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from persemcom.corrector.model import Corrector  # noqa: E402
from persemcom.data.manifest import read_manifest  # noqa: E402
from persemcom.metrics.wer import wer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=["data/corrector/pairs_awgn.jsonl", "data/corrector/pairs_rayleigh.jsonl"])
    ap.add_argument("--manifest", default="manifests/train.jsonl")
    ap.add_argument("--corrector", default="checkpoints/corrector/best")
    ap.add_argument("--max-duration", type=float, default=10.0)
    ap.add_argument("--out", default="data/corrector/train_corrected.jsonl")
    ap.add_argument("--num-beams", type=int, default=4)
    args = ap.parse_args()

    wanted = {r["id"]: r for r in read_manifest(args.manifest) if r["duration"] <= args.max_duration}
    noisy = {}
    for k, path in enumerate(args.pairs):
        for line in open(path, encoding="utf-8"):
            p = json.loads(line)
            # pass 0 only; alternate channels by utterance so both error types are represented
            if p["pass"] == 0 and p["id"] in wanted and zlib.crc32(p["id"].encode()) % len(args.pairs) == k:
                noisy[p["id"]] = p
    ids = sorted(noisy)
    print(f"{len(ids)} of {len(wanted)} TTS train utterances have a noisy transcript", flush=True)

    corr = Corrector(args.corrector)
    srcs = [noisy[i]["src"] for i in ids]
    outs = []
    for s in range(0, len(srcs), 512):
        outs += corr.correct(srcs[s:s + 512], num_beams=args.num_beams)
        print(f"corrected {len(outs)}/{len(srcs)}", flush=True)
    refs = [wanted[i]["text"] for i in ids]
    print(f"WER vs reference: noisy {wer(refs, srcs):.4f} -> corrected {wer(refs, outs):.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, t in zip(ids, outs):
            f.write(json.dumps({"id": i, "text": t}, ensure_ascii=False) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
