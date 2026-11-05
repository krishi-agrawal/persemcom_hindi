"""Cache features for every manifest utterance and compute fbank CMVN on train.

features/fbank/{id}.npy : (frames, 80) float16, ASR input at 16 kHz / 10 ms
features/mel24/{id}.npy : (frames, 100) float16, Vocos-matched TTS target at 24 kHz / hop 256
artifacts/fbank_cmvn.npz: global mean/std of train fbank
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from persemcom.data.features import AUDIO_SR, asr_fbank, vocos_mel  # noqa: E402
from persemcom.data.manifest import read_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", default="manifests")
    ap.add_argument("--out", default="features")
    ap.add_argument("--cmvn", default="artifacts/fbank_cmvn.npz")
    args = ap.parse_args()

    fb_dir, mel_dir = Path(args.out, "fbank"), Path(args.out, "mel24")
    fb_dir.mkdir(parents=True, exist_ok=True)
    mel_dir.mkdir(parents=True, exist_ok=True)

    s1 = np.zeros(80, np.float64)
    s2 = np.zeros(80, np.float64)
    n = 0
    for split in ("train", "val", "test"):
        rows = read_manifest(f"{args.manifests}/{split}.jsonl")
        for i, r in enumerate(rows, 1):
            fb_path, mel_path = fb_dir / f"{r['id']}.npy", mel_dir / f"{r['id']}.npy"
            if fb_path.exists() and mel_path.exists():
                fb = np.load(fb_path).astype(np.float32) if split == "train" else None
            else:
                wav, sr = sf.read(r["path"], dtype="float32")
                assert sr == AUDIO_SR, (r["path"], sr)
                wav = torch.from_numpy(wav)
                fb = asr_fbank(wav).numpy()
                np.save(fb_path, fb.astype(np.float16))
                np.save(mel_path, vocos_mel(wav).numpy().astype(np.float16))
            if split == "train":
                s1 += fb.sum(0)
                s2 += (fb.astype(np.float64) ** 2).sum(0)
                n += len(fb)
            if i % 2000 == 0:
                print(f"{split}: {i}/{len(rows)}", flush=True)
        print(f"{split}: done ({len(rows)} utts)", flush=True)

    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-8))
    Path(args.cmvn).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.cmvn, mean=mean.astype(np.float32), std=std.astype(np.float32))
    print("cmvn:", args.cmvn, "frames:", n)


if __name__ == "__main__":
    main()
