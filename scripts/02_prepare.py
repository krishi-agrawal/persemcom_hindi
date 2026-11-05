"""Decode IndicVoices-R Hindi shards to mono 24 kHz FLAC, filter, make the speaker-shared split,
and train the SentencePiece tokenizer on the train transcripts.

The official train/test shards are pooled: the paper's personalization needs every test speaker
to also appear in train (and in the speaker knowledge base), so the split is made per speaker.
"""
import argparse
import io
import random
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from persemcom.data.features import AUDIO_SR, resample, to_mono  # noqa: E402
from persemcom.data.manifest import read_manifest, write_manifest  # noqa: E402
from persemcom.data.text_norm import is_clean_hindi, normalize  # noqa: E402
from persemcom.data.tokenizer import train_tokenizer  # noqa: E402

META_COLS = ["text", "speaker_id", "gender", "scenario", "duration", "snr", "cer"]


def _parse_cer(v) -> float:
    # stored as the string "tensor(0.1234)"
    try:
        return float(str(v).replace("tensor(", "").rstrip(")"))
    except ValueError:
        return float("inf")


def process_shard(job):
    shard, audio_dir, min_dur, max_dur, max_cer = job
    pf = pq.ParquetFile(shard)
    tag = Path(shard).stem  # e.g. train-00012-of-00099
    rows, dropped = [], defaultdict(int)
    for batch in pf.iter_batches(batch_size=64, columns=META_COLS + ["audio"]):
        for i, r in enumerate(batch.to_pylist()):
            if not is_clean_hindi(r["text"]):
                dropped["text"] += 1
                continue
            if _parse_cer(r["cer"]) > max_cer:
                dropped["cer"] += 1
                continue
            if not (min_dur <= r["duration"] <= max_dur):
                dropped["duration"] += 1
                continue
            wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            wav = resample(to_mono(torch.from_numpy(wav)), sr, AUDIO_SR)
            utt_id = f"{tag}_{len(rows) + sum(dropped.values()):06d}_{Path(r['audio']['path']).stem}"
            out = Path(audio_dir) / r["speaker_id"] / f"{utt_id}.flac"
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, wav.numpy(), AUDIO_SR)
            rows.append({
                "id": utt_id,
                "path": out.as_posix(),
                "duration": round(len(wav) / AUDIO_SR, 3),
                "text": normalize(r["text"]),
                "speaker_id": r["speaker_id"],
                "gender": r["gender"],
                "scenario": r["scenario"],
                "snr": r["snr"],
            })
    return rows, dict(dropped)


def speaker_split(rows, min_utts, val_frac, test_frac, seed):
    by_spk = defaultdict(list)
    for r in rows:
        by_spk[r["speaker_id"]].append(r)
    rng = random.Random(seed)
    split = {"train": [], "val": [], "test": []}
    kept = 0
    for spk in sorted(by_spk):
        utts = sorted(by_spk[spk], key=lambda r: r["id"])
        if len(utts) < min_utts:
            continue
        kept += 1
        rng.shuffle(utts)
        n_val = max(1, round(val_frac * len(utts)))
        n_test = max(1, round(test_frac * len(utts)))
        split["val"] += utts[:n_val]
        split["test"] += utts[n_val:n_val + n_test]
        split["train"] += utts[n_val + n_test:]
    return split, kept, len(by_spk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/indicvoices_r/Hindi")
    ap.add_argument("--audio-dir", default="data/wav24")
    ap.add_argument("--manifests", default="manifests")
    ap.add_argument("--tokenizer", default="artifacts/spm_hi_5000")
    ap.add_argument("--vocab-size", type=int, default=5000)
    ap.add_argument("--min-dur", type=float, default=1.0)
    # 15-25 s clips are 28% of IndicVoices-R Hindi; TTS applies its own <= 10 s filter at load time
    ap.add_argument("--max-dur", type=float, default=25.0)
    ap.add_argument("--max-cer", type=float, default=0.1, help="dataset-provided transcript CER filter")
    ap.add_argument("--min-utts", type=int, default=20, help="drop speakers with fewer utterances")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tokenizer-only", action="store_true", help="retrain the tokenizer from existing manifests")
    args = ap.parse_args()

    if args.tokenizer_only:
        rows = read_manifest(f"{args.manifests}/train.jsonl")
        train_tokenizer((r["text"] for r in rows), args.tokenizer, args.vocab_size)
        print("tokenizer:", args.tokenizer + ".model")
        return

    shards = sorted(str(p) for p in Path(args.raw).glob("*.parquet"))
    if not shards:
        sys.exit(f"no parquet shards under {args.raw}; run scripts/01_download.py first")
    jobs = [(s, args.audio_dir, args.min_dur, args.max_dur, args.max_cer) for s in shards]
    rows, dropped = [], defaultdict(int)
    with Pool(args.workers) as pool:
        for i, (r, d) in enumerate(pool.imap_unordered(process_shard, jobs), 1):
            rows += r
            for k, v in d.items():
                dropped[k] += v
            print(f"[{i}/{len(shards)}] kept {len(rows)} utts", flush=True)

    split, kept, total = speaker_split(rows, args.min_utts, args.val_frac, args.test_frac, args.seed)
    for name, part in split.items():
        write_manifest(part, f"{args.manifests}/{name}.jsonl")
        print(f"{name}: {len(part)} utts, {sum(r['duration'] for r in part) / 3600:.1f} h")
    print(f"speakers kept {kept}/{total}; dropped utterances: {dict(dropped)}")

    train_tokenizer((r["text"] for r in split["train"]), args.tokenizer, args.vocab_size)
    print("tokenizer:", args.tokenizer + ".model")


if __name__ == "__main__":
    main()
