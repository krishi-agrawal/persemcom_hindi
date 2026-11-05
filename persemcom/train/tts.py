"""Training of the speaker-conditioned Tacotron 2 on Vocos-matched mels (paper Sec. IV-E, Alg. 1 l.12-15)."""
import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from persemcom.data.datasets import DurationBucketSampler
from persemcom.data.manifest import read_manifest
from persemcom.data.tts_text import GraphemeVocab
from persemcom.kb.build_kb import load_kb
from persemcom.models.tacotron2_spk import (SpeakerTacotron2, attention_coverage, attention_focus_rate,
                                            guided_attention_loss, lengths_to_mask)


class TTSDataset(Dataset):
    """(graphemes, Vocos mel, speaker KB embedding). text_override maps utterance id -> input text,
    used for the Algorithm-1 fine-tune on corrected transcripts T_C (targets stay the true mels)."""

    def __init__(self, manifest, mel_dir, vocab: GraphemeVocab, kb: dict, min_duration=0.5, max_duration=10.0,
                 text_override: Optional[dict] = None):
        rows = [r for r in read_manifest(manifest) if min_duration <= r["duration"] <= max_duration]
        if text_override is not None:
            rows = [dict(r, text=text_override[r["id"]]) for r in rows if r["id"] in text_override]
        self.rows, self.mel_dir, self.vocab, self.kb = rows, Path(mel_dir), vocab, kb

    def __len__(self):
        return len(self.rows)

    @property
    def durations(self):
        return [r["duration"] for r in self.rows]

    def __getitem__(self, i):
        r = self.rows[i]
        mel = torch.from_numpy(np.load(self.mel_dir / f"{r['id']}.npy").astype(np.float32)).T  # (n_mels, T)
        return {"id": r["id"], "tokens": torch.tensor(self.vocab.encode(r["text"])), "mel": mel,
                "spk": self.kb[r["speaker_id"]], "speaker_id": r["speaker_id"], "text": r["text"]}


class TTSCollate:
    """A class, not a closure: Windows DataLoader workers receive the collate function by pickling."""

    def __init__(self, r: int):
        self.r = r

    def __call__(self, items):
        r = self.r
        # torchaudio's encoder packs sequences and requires decreasing token lengths
        items = sorted(items, key=lambda x: len(x["tokens"]), reverse=True)
        tok_lens = torch.tensor([len(x["tokens"]) for x in items])
        mel_lens = torch.tensor([x["mel"].shape[1] for x in items])
        T = int(math.ceil(mel_lens.max().item() / r) * r)
        mel = torch.zeros(len(items), items[0]["mel"].shape[0], T)
        for i, x in enumerate(items):
            mel[i, :, : x["mel"].shape[1]] = x["mel"]
        steps = torch.ceil(mel_lens / r).long()
        # stop target: 1 from the last valid decoder step on (padding included, as in NVIDIA's recipe)
        gate = (torch.arange(T // r)[None, :] >= (steps - 1)[:, None]).float()
        return {"tokens": torch.nn.utils.rnn.pad_sequence([x["tokens"] for x in items], batch_first=True),
                "token_lens": tok_lens, "mel": mel, "mel_lens": mel_lens, "steps": steps, "gate": gate,
                "spk": torch.stack([x["spk"] for x in items]), "ids": [x["id"] for x in items],
                "speaker_ids": [x["speaker_id"] for x in items], "texts": [x["text"] for x in items]}


def make_collate(r: int) -> TTSCollate:
    return TTSCollate(r)


def frame_dynamics(mel: torch.Tensor, lens: torch.Tensor) -> float:
    """Mean absolute frame-to-frame change of valid mel frames. Free-running output whose dynamics fall far
    below the real speech's (run 2, r=2: 0.118 vs 0.459) is a stationary drone that the decoder produces by
    continuing its own previous frame instead of following the attended text."""
    vals = [m[:, :n].float().diff(dim=1).abs().mean() for m, n in zip(mel, lens) if n > 1]
    return float(torch.stack(vals).mean())


def tts_loss(out, b, ga_weight: float, gate_pos_weight: float):
    mel_before, mel_after, gate_logits, align = out
    mask = lengths_to_mask(b["mel_lens"], b["mel"].shape[-1])[:, None, :].float()
    denom = mask.sum() * b["mel"].shape[1]
    l_before = ((mel_before.float() - b["mel"]) ** 2 * mask).sum() / denom
    l_after = ((mel_after.float() - b["mel"]) ** 2 * mask).sum() / denom
    l_gate = F.binary_cross_entropy_with_logits(gate_logits.float(), b["gate"],
                                                pos_weight=torch.tensor(gate_pos_weight, device=gate_logits.device))
    l_ga = guided_attention_loss(align, b["token_lens"], b["steps"]) if ga_weight > 0 else torch.zeros(())
    total = l_before + l_after + l_gate + ga_weight * l_ga
    return total, {"mel_before": l_before.item(), "mel_after": l_after.item(), "gate": l_gate.item(),
                   "guided_attn": l_ga.item()}


def build_tts(cfg_model: dict, n_symbol: int) -> SpeakerTacotron2:
    return SpeakerTacotron2(n_symbol=n_symbol, **cfg_model)


def load_partial(model: torch.nn.Module, state: dict) -> list:
    """Copy every tensor whose name and shape match (e.g. when only the reduction factor r changed, which
    resizes the prenet input and the output projection); returns the names that were left at init."""
    own = model.state_dict()
    keep = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(keep, strict=False)
    return sorted(k for k in own if k not in keep)


class TTSTrainer:
    def __init__(self, cfg: dict, text_override: Optional[dict] = None, init_from: Optional[str] = None):
        self.cfg, d, t = cfg, cfg["data"], cfg["train"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not Path(d["vocab"]).exists():
            GraphemeVocab.build(r["text"] for r in read_manifest(d["train_manifest"])).save(d["vocab"])
        self.vocab = GraphemeVocab.load(d["vocab"])
        kb = load_kb(d["speaker_kb"])
        self.model = build_tts(cfg["model"], len(self.vocab)).to(self.device)
        if init_from:
            fresh = load_partial(self.model, torch.load(init_from, map_location="cpu")["model"])
            print(f"initialized from {init_from}; newly initialized: {fresh or 'none'}", flush=True)
        r = self.model.r
        common = dict(mel_dir=d["mel_dir"], vocab=self.vocab, kb=kb, max_duration=d["max_duration"])
        self.train_ds = TTSDataset(d["train_manifest"], text_override=text_override, **common)
        self.val_ds = TTSDataset(d["val_manifest"], **common)
        self.sampler = DurationBucketSampler(self.train_ds.durations, d["max_batch_seconds"])
        self.train_dl = DataLoader(self.train_ds, batch_sampler=self.sampler, collate_fn=make_collate(r),
                                   num_workers=t.get("workers", 4), persistent_workers=True)
        val_rows = list(range(0, len(self.val_ds), max(1, len(self.val_ds) // t.get("val_utts", 200))))
        self.val_dl = DataLoader(torch.utils.data.Subset(self.val_ds, val_rows), batch_size=16,
                                 collate_fn=make_collate(r))
        self.opt = torch.optim.Adam(self.model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 1e-6))
        self.ckpt_dir = Path(t["ckpt_dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.ckpt_dir / "tb"))
        self.step, self.epoch, self.best = 0, 0, float("inf")
        if (self.ckpt_dir / "last.pt").exists():
            ck = torch.load(self.ckpt_dir / "last.pt", map_location="cpu")
            self.model.load_state_dict(ck["model"])
            self.opt.load_state_dict(ck["opt"])
            self.step, self.epoch, self.best = ck["step"], ck["epoch"], ck["best"]
            print(f"resumed at epoch {self.epoch}, step {self.step}")
        self._vocos = None

    def _to(self, b):
        return {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}

    def _lr(self):
        t = self.cfg["train"]
        warm = t.get("warmup_steps", 1000)
        decay_start, decay_rate = t.get("lr_decay_start", 40000), t.get("lr_decay_rate", 0.5)
        lr = t["lr"] * min(1.0, (self.step + 1) / warm)
        if self.step > decay_start:  # halve every decay_start steps after decay_start
            lr *= decay_rate ** ((self.step - decay_start) / decay_start)
        return max(lr, t.get("lr_min", 1e-5))

    def _ga_weight(self):
        t = self.cfg["train"]
        w0, stop = t.get("guided_attn_weight", 1.0), t.get("guided_attn_until", 20000)
        return w0 if self.step < stop else 0.0

    @torch.no_grad()
    def validate(self, n_samples: int = 4):
        self.model.eval()
        t = self.cfg["train"]
        tot, n, tf_cov = 0.0, 0, []
        for b in self.val_dl:
            b = self._to(b)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                out = self.model(b["tokens"], b["token_lens"], b["mel"], b["mel_lens"], b["spk"])
            loss, parts = tts_loss(out, b, 0.0, t.get("gate_pos_weight", 5.0))
            tot += parts["mel_after"] * len(b["ids"])
            n += len(b["ids"])
            tf_cov.append(attention_coverage(out[3], b["token_lens"], b["steps"]))
        # free-running inference on a few utterances: alignment sharpness, length ratio, audio samples
        b = self._to(next(iter(self.val_dl)))
        k = min(n_samples, len(b["ids"]))
        mel, lens, align = self.model.infer(b["tokens"][:k], b["token_lens"][:k], b["spk"][:k])
        steps = torch.ceil(lens / self.model.r).long()
        focus = attention_focus_rate(align, b["token_lens"][:k], steps)
        len_ratio = float((lens.float() / b["mel_lens"][:k].float()).mean())
        self._write_samples(mel, lens, b["ids"][:k])
        self.model.train()
        return {"mel_after": tot / max(n, 1), "focus_rate": focus, "len_ratio": len_ratio,
                "tf_coverage": sum(tf_cov) / len(tf_cov),
                "fr_coverage": attention_coverage(align, b["token_lens"][:k], steps),
                "fr_dynamics": frame_dynamics(mel, lens) / frame_dynamics(b["mel"][:k], b["mel_lens"][:k])}

    def _write_samples(self, mel, lens, ids):
        if self._vocos is None:
            from vocos import Vocos
            self._vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(self.device).eval()
        out = self.ckpt_dir / "samples" / f"step{self.step:07d}"
        out.mkdir(parents=True, exist_ok=True)
        for m, n, uid in zip(mel, lens, ids):
            wav = self._vocos.decode(m[None, :, :n].float())[0].cpu().numpy()
            sf.write(out / f"{uid}.wav", wav, 24000)

    def _save(self, path):
        torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict(), "step": self.step,
                    "epoch": self.epoch, "best": self.best, "config": self.cfg["model"],
                    "vocab": self.vocab.symbols[2:]}, path)

    def fit(self, max_steps: Optional[int] = None):
        t = self.cfg["train"]
        total = max_steps or t["steps"]
        self.model.train()
        t0 = time.time()
        while self.step < total:
            self.sampler.set_epoch(self.epoch)
            for b in self.train_dl:
                for g in self.opt.param_groups:
                    g["lr"] = self._lr()
                b = self._to(b)
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    out = self.model(b["tokens"], b["token_lens"], b["mel"], b["mel_lens"], b["spk"])
                loss, parts = tts_loss(out, b, self._ga_weight(), t.get("gate_pos_weight", 5.0))
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), t.get("grad_clip", 1.0))
                if torch.isfinite(gnorm):
                    self.opt.step()
                self.step += 1
                if self.step % 50 == 0:
                    for k, v in parts.items():
                        self.writer.add_scalar(f"tts/{k}", v, self.step)
                    print(f"ep {self.epoch} step {self.step}/{total} loss {loss.item():.4f} "
                          + " ".join(f"{k} {v:.4f}" for k, v in parts.items())
                          + f" lr {self._lr():.2e} ({time.time() - t0:.0f}s)", flush=True)
                if self.step % t["eval_every"] == 0 or self.step >= total:
                    res = self.validate()
                    for k, v in res.items():
                        self.writer.add_scalar(f"tts/val_{k}", v, self.step)
                    print(f"[step {self.step}] val mel_after {res['mel_after']:.4f} focus {res['focus_rate']:.3f} "
                          f"len_ratio {res['len_ratio']:.2f} coverage teacher-forced {res['tf_coverage']:.3f} "
                          f"free-running {res['fr_coverage']:.3f} dynamics {res['fr_dynamics']:.2f}", flush=True)
                    if res["mel_after"] < self.best:
                        self.best = res["mel_after"]
                        self._save(self.ckpt_dir / "best.pt")
                    self._save(self.ckpt_dir / "last.pt")
                if self.step >= total:
                    break
            self.epoch += 1
        self.writer.close()


def load_text_override(path: str) -> dict:
    """JSONL with {"id", "text"} per utterance (e.g. corrector outputs on train transcripts)."""
    return {r["id"]: r["text"] for r in map(json.loads, open(path, encoding="utf-8")) if r.get("text")}
