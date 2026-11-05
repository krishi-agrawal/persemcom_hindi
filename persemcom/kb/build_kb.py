"""Speaker knowledge base (paper Sec. IV-E.2): speaker id -> mean of L2-normalized utterance embeddings
over that speaker's *train* utterances, re-normalized. The receiver indexes it by speaker identity (Fig. 6).
"""
from typing import Dict

import numpy as np
import torch

from persemcom.data.manifest import read_manifest
from persemcom.metrics.spk_sim import centroid_eer
from persemcom.models.speaker_encoder import SpeakerEncoder
from persemcom.train.ge2e import embed_manifest, speaker_centroids


def load_speaker_encoder(ckpt: str, device=None) -> SpeakerEncoder:
    ck = torch.load(ckpt, map_location="cpu")
    model = SpeakerEncoder(**ck["config"])
    model.load_state_dict(ck["model"])
    return model.to(device or "cpu").eval()


def build_kb(cfg: dict, ckpt: str) -> Dict[str, np.ndarray]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = cfg["data"]
    model = load_speaker_encoder(ckpt, device)
    stats = np.load(d["cmvn"])
    mean, std = torch.from_numpy(stats["mean"]), torch.from_numpy(stats["std"])
    train = read_manifest(d["train_manifest"])
    embs = embed_manifest(model, train, d["feat_dir"], mean, std, device)
    cent = speaker_centroids(embs, [r["speaker_id"] for r in train])
    kb = {s: c.numpy().astype(np.float32) for s, c in cent.items()}
    np.savez(cfg["kb"]["out"], **kb)
    val = read_manifest(d["val_manifest"])
    val_emb = embed_manifest(model, val, d["feat_dir"], mean, std, device)
    print(f"knowledge base: {len(kb)} speakers -> {cfg['kb']['out']} | "
          f"val EER vs full-train centroids {centroid_eer(val_emb, [r['speaker_id'] for r in val], cent):.4f}")
    return kb


def load_kb(path: str) -> Dict[str, torch.Tensor]:
    z = np.load(path)
    return {k: torch.from_numpy(z[k]) for k in z.files}
