"""Build the clip set for the MOS listening test (paper Sec. V-G, Fig. 14; ITU-T P.800 ACR).

    python scripts/13_mos_set.py

Picks 12 test sentences (one per speaker, 3-8 s, every condition actually delivered speech) plus one
calibration sentence from the speech-to-speech subset, and exports each condition's WAV from results/wavs/
as level-equalized MP3 (-23 dBFS RMS, as P.800 asks for equal listening levels) under an anonymous file name,
so listeners cannot tell the condition from the URL. Writes results/mos/clips.json + results/mos/audio/.
Conditions where a digital link delivered no speech (AMR-WB / Txt2Vid at <= 2 dB) are not played: they
score 1 by definition.
"""
import argparse
import hashlib
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from persemcom.baselines.amrwb import FFMPEG  # noqa: E402
from persemcom.evaluation import s2s_subset  # noqa: E402

# (id, folder under results/wavs, system, channel, snr_db)
CONDITIONS = [
    ("orig", "awgn/original", "Original recording", "", None),
    ("ps_gt", "awgn/persemcom_gt_text", "PerSemCom, true transcript", "", None),
    ("ps_a-10", "awgn/persemcom_-10dB", "PerSemCom", "AWGN", -10),
    ("ps_a-2", "awgn/persemcom_-2dB", "PerSemCom", "AWGN", -2),
    ("ps_a2", "awgn/persemcom_2dB", "PerSemCom", "AWGN", 2),
    ("ps_a10", "awgn/persemcom_10dB", "PerSemCom", "AWGN", 10),
    ("ps_r-10", "rayleigh/persemcom_-10dB", "PerSemCom", "Rayleigh", -10),
    ("ps_r-2", "rayleigh/persemcom_-2dB", "PerSemCom", "Rayleigh", -2),
    ("ps_r2", "rayleigh/persemcom_2dB", "PerSemCom", "Rayleigh", 2),
    ("ps_r10", "rayleigh/persemcom_10dB", "PerSemCom", "Rayleigh", 10),
    ("amr_a20", "awgn/amrwb_20dB", "AMR-WB + LDPC", "AWGN", 20),  # no frame lost: same audio as 10 dB
    ("amr_r10", "rayleigh/amrwb_10dB", "AMR-WB + LDPC", "Rayleigh", 10),
    ("t2v_a10", "awgn/txt2vid_10dB", "Txt2Vid-style", "AWGN", 10),
    ("t2v_r10", "rayleigh/txt2vid_10dB", "Txt2Vid-style", "Rayleigh", 10),
]
CALIBRATION = ["orig", "amr_r10", "ps_r-10"]  # spans the range; played before rating, never scored


def level(wav: np.ndarray, rms_db: float = -23.0) -> np.ndarray:
    gain = 10 ** (rms_db / 20) / max(float(np.sqrt(np.mean(wav ** 2))), 1e-6)
    return wav * min(gain, 0.99 / max(float(np.abs(wav).max()), 1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifests/test.jsonl")
    ap.add_argument("--wavs", default="results/wavs")
    ap.add_argument("--out", default="results/mos")
    ap.add_argument("--sentences", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    wav_dir, out = Path(args.wavs), Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)

    def path(cond, utt):
        return wav_dir / dict((c[0], c[1]) for c in CONDITIONS)[cond] / f"{utt}.wav"

    def usable(r):
        return 3.0 <= r["duration"] <= 8.0 and all(
            path(c[0], r["id"]).exists() and sf.info(path(c[0], r["id"])).duration >= 0.5 for c in CONDITIONS)

    pool = [r for r in s2s_subset(args.manifest) if usable(r)]
    random.Random(args.seed).shuffle(pool)
    chosen, speakers = [], set()
    by_gender = {g: [r for r in pool if r.get("gender") == g] for g in {r.get("gender") for r in pool}}
    while len(chosen) < args.sentences + 1 and any(by_gender.values()):  # alternate genders, one per speaker
        for g in sorted(by_gender, key=str):
            while by_gender[g]:
                r = by_gender[g].pop()
                if r["speaker_id"] not in speakers:
                    speakers.add(r["speaker_id"])
                    chosen.append(r)
                    break
    calib, sentences = chosen[0], chosen[1:args.sentences + 1]
    print(f"{len(pool)} usable sentences; picked {len(sentences)} + 1 calibration")

    clips = []
    with tempfile.TemporaryDirectory() as tmp:
        for r in [calib] + sentences:
            for c in CONDITIONS:
                if r is calib and c[0] not in CALIBRATION:
                    continue
                wav, sr = sf.read(path(c[0], r["id"]), dtype="float32")
                name = hashlib.sha1(f"{args.seed}:{c[0]}:{r['id']}".encode()).hexdigest()[:12] + ".mp3"
                src = Path(tmp) / "x.wav"
                sf.write(src, level(wav), sr)
                subprocess.run([str(FFMPEG), "-v", "error", "-y", "-i", str(src), "-ac", "1", "-c:a", "libmp3lame",
                                "-b:a", "96k", str(out / "audio" / name)], check=True)
                clips.append({"file": f"audio/{name}", "cond": c[0], "sentence": r["id"],
                              "calibration": r is calib, "duration": round(len(wav) / sr, 2)})
    conds = [{"id": c[0], "system": c[2], "channel": c[3], "snr_db": c[4]} for c in CONDITIONS]
    json.dump({"conditions": conds, "sentences": [r["id"] for r in sentences], "clips": clips},
              open(out / "clips.json", "w", encoding="utf-8"), indent=1)
    print(f"wrote {len(clips)} clips to {out}")


if __name__ == "__main__":
    main()
