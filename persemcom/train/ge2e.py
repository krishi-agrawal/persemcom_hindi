"""GE2E training of the speaker encoder on train-split speakers, with closed-set EER on val utterances."""
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter

from persemcom.data.manifest import read_manifest
from persemcom.metrics.spk_sim import centroid_eer
from persemcom.models.speaker_encoder import PARTIAL_FRAMES, SpeakerEncoder


def load_feats(feat_dir: Path, uid: str, mean=None, std=None) -> torch.Tensor:
    x = torch.from_numpy(np.load(feat_dir / f"{uid}.npy").astype(np.float32))
    return (x - mean) / std if mean is not None else x


class GE2EBatches(IterableDataset):
    """Endless N x M batches of random 1.6 s partials: N speakers, M utterances each."""

    def __init__(self, manifest, feat_dir, n_speakers, n_utts, cmvn=None, seed=0):
        self.by_spk = defaultdict(list)
        for r in read_manifest(manifest):
            self.by_spk[r["speaker_id"]].append(r["id"])
        self.spks = sorted(s for s, u in self.by_spk.items() if len(u) >= n_utts)
        self.feat_dir, self.N, self.M, self.seed = Path(feat_dir), n_speakers, n_utts, seed
        stats = np.load(cmvn) if cmvn else None
        self.mean = torch.from_numpy(stats["mean"]) if stats is not None else None
        self.std = torch.from_numpy(stats["std"]) if stats is not None else None

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (info.id if info else 0) + int(time.time()))
        while True:
            batch = []
            for spk in rng.sample(self.spks, self.N):
                for uid in rng.sample(self.by_spk[spk], self.M):
                    x = load_feats(self.feat_dir, uid, self.mean, self.std)
                    if len(x) < PARTIAL_FRAMES:
                        x = torch.nn.functional.pad(x, (0, 0, 0, PARTIAL_FRAMES - len(x)))
                    s = rng.randint(0, len(x) - PARTIAL_FRAMES)
                    batch.append(x[s:s + PARTIAL_FRAMES])
            yield torch.stack(batch)


@torch.no_grad()
def embed_manifest(model, manifest_rows, feat_dir, mean, std, device):
    embs = []
    for r in manifest_rows:
        x = load_feats(Path(feat_dir), r["id"], mean, std).to(device)
        embs.append(model.embed_utterance(x).cpu())
    return torch.stack(embs)


def speaker_centroids(embs: torch.Tensor, spk_ids: list) -> dict:
    by = defaultdict(list)
    for e, s in zip(embs, spk_ids):
        by[s].append(e)
    return {s: torch.nn.functional.normalize(torch.stack(v).mean(0), dim=-1) for s, v in by.items()}


def validate(model, train_rows, val_rows, feat_dir, mean, std, device, max_train_per_spk=10):
    """Closed-set EER: val utterances scored against centroids built from (a few) train utterances."""
    model.eval()
    per = defaultdict(list)
    for r in train_rows:
        if len(per[r["speaker_id"]]) < max_train_per_spk:
            per[r["speaker_id"]].append(r)
    enroll = [r for v in per.values() for r in v]
    cent = speaker_centroids(embed_manifest(model, enroll, feat_dir, mean, std, device), [r["speaker_id"] for r in enroll])
    val_emb = embed_manifest(model, val_rows, feat_dir, mean, std, device)
    model.train()
    return centroid_eer(val_emb, [r["speaker_id"] for r in val_rows], cent)


def train_ge2e(cfg: dict, max_steps=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d, t = cfg["data"], cfg["train"]
    ds = GE2EBatches(d["train_manifest"], d["feat_dir"], t["n_speakers"], t["n_utts"], d.get("cmvn"))
    dl = DataLoader(ds, batch_size=None, num_workers=t.get("workers", 4), persistent_workers=True, prefetch_factor=4)
    model = SpeakerEncoder(**cfg["model"]).to(device)
    # The GE2E paper scales the gradient of w and b by 0.01 (for SGD). Adam is invariant to gradient scale,
    # and a 0.01x learning rate instead left w, b frozen at their init in the smoke test, so use the base LR.
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])
    ckpt_dir = Path(t["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(ckpt_dir / "tb"))
    train_rows, val_rows = read_manifest(d["train_manifest"]), read_manifest(d["val_manifest"])
    mean, std = ds.mean, ds.std
    step, best = 0, float("inf")
    if (ckpt_dir / "last.pt").exists():
        ck = torch.load(ckpt_dir / "last.pt", map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, best = ck["step"], ck["best"]
        print(f"resumed at step {step}")
    total = max_steps or t["steps"]
    t0 = time.time()
    for batch in dl:
        x = batch.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            emb = model(x)
        loss = model.ge2e_loss(emb.float().view(t["n_speakers"], t["n_utts"], -1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step()
        step += 1
        if step % 100 == 0:
            writer.add_scalar("ge2e/loss", loss.item(), step)
            print(f"step {step}/{total} loss {loss.item():.4f} w {model.w.item():.2f} b {model.b.item():.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if step % t["eval_every"] == 0 or step >= total:
            e = validate(model, train_rows, val_rows, d["feat_dir"], mean, std, device)
            writer.add_scalar("ge2e/val_eer", e, step)
            print(f"[step {step}] val closed-set EER {e:.4f}", flush=True)
            state = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": min(best, e),
                     "config": cfg["model"]}
            if e < best:
                best = e
                torch.save(state, ckpt_dir / "best.pt")
            torch.save(state, ckpt_dir / "last.pt")
        if step >= total:
            break
    writer.close()
    return best
