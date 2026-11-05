"""Speaker-embedding visualization (paper Fig. 15 a, c).

    python scripts/12_speaker_tsne.py --channel awgn --snr 10

For the 10 test speakers with the most clips: GE2E embeddings of the transmitted speech (Tr) and of PerSemCom's
reconstruction (Re) from the corrected received transcript, once with the speaker's knowledge-base entry and
once with the speaker-agnostic KB mean. Writes 2-D t-SNE coordinates (one t-SNE per panel) and, as a number to
go with the picture, how often a reconstruction is closest to its own speaker's Tr centroid (chance = 10%).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from persemcom.data.manifest import read_manifest  # noqa: E402
from persemcom.evaluation import SpeechJudge, load_audio, write_csv  # noqa: E402
from persemcom.pipeline import PerSemComPipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", choices=["awgn", "rayleigh"], default="awgn")
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--tts-ckpt", default="checkpoints/tts_finetune/best.pt")
    ap.add_argument("--manifest", default="manifests/test.jsonl")
    ap.add_argument("--speakers", type=int, default=10)
    ap.add_argument("--max-duration", type=float, default=10.0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    by_spk = defaultdict(list)
    for r in read_manifest(args.manifest):
        if r["duration"] <= args.max_duration:
            by_spk[r["speaker_id"]].append(r)
    speakers = sorted(by_spk, key=lambda s: (-len(by_spk[s]), s))[:args.speakers]
    rows = [r for s in speakers for r in by_spk[s]]
    hyps = {d["id"]: d["corrected"] for d in map(
        json.loads, open(Path(args.out) / f"s2t_hyps_{args.channel}_{args.snr:g}dB.jsonl", encoding="utf-8"))}

    pipe = PerSemComPipeline(f"checkpoints/{args.channel}/stage2/best.pt", args.channel, tts_ckpt=args.tts_ckpt)
    judge = SpeechJudge(device=str(pipe.device))
    kb_mean = torch.nn.functional.normalize(torch.stack(list(pipe.kb.values())).mean(0), dim=0)[None]
    texts, spk_ids = [hyps[r["id"]] for r in rows], [r["speaker_id"] for r in rows]
    emb = {
        "Tr": torch.stack([judge.embed(load_audio(r)) for r in rows]).numpy(),
        "Re": torch.stack([judge.embed(w) for w in pipe.synthesize(texts, spk_ids)]).numpy(),
        "Re_no_speaker": torch.stack([judge.embed(w) for w in pipe.synthesize(
            texts, spk_ids, speaker_emb=kb_mean.expand(len(rows), -1))]).numpy(),
    }

    labels = np.array(spk_ids)
    centroids = np.stack([emb["Tr"][labels == s].mean(0) for s in speakers])
    out, summary = [], []
    for panel, kind in [("a_with_speaker_embedding", "Re"), ("c_without_speaker_embedding", "Re_no_speaker")]:
        pred = np.array(speakers)[(emb[kind] @ centroids.T).argmax(1)]
        acc = float((pred == labels).mean())
        summary.append({"panel": panel, "n_speakers": len(speakers), "n_clips": len(rows),
                        "re_identified_as_own_speaker": acc})
        print(summary[-1], flush=True)
        xy = TSNE(n_components=2, perplexity=min(15, len(rows) - 1), init="pca", random_state=0).fit_transform(
            np.concatenate([emb["Tr"], emb[kind]]))
        for i, (x, y) in enumerate(xy):
            r = rows[i % len(rows)]
            out.append({"panel": panel, "speaker": f"S{speakers.index(r['speaker_id']) + 1}",
                        "kind": "Tr" if i < len(rows) else "Re", "id": r["id"], "x": float(x), "y": float(y)})
    write_csv(out, Path(args.out) / "tsne_speakers.csv")
    write_csv(summary, Path(args.out) / "tsne_speaker_id.csv")


if __name__ == "__main__":
    main()
