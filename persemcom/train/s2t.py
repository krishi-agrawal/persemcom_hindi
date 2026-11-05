"""Trainer for the speech-to-text half (Algorithm 1): ASR pretraining, Stage 1 and Stage 2."""
import contextlib
import math
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from persemcom.data.datasets import ASRDataset, DurationBucketSampler, collate_asr
from persemcom.data.tokenizer import Tokenizer
from persemcom.metrics.wer import cer, wer
from persemcom.models.channel import sample_snr
from persemcom.models.ctc_head import ctc_greedy_decode
from persemcom.models.persemcom_s2t import PerSemComS2T


def spec_augment(feats, lens, freq_masks=2, freq_width=27, time_masks=10, time_width_ratio=0.05):
    """In-place SpecAugment (no time warp) on (B, T, F), respecting each utterance's length."""
    B, _, F = feats.shape
    for b in range(B):
        T = int(lens[b])
        for _ in range(freq_masks):
            w = torch.randint(0, freq_width + 1, ()).item()
            f0 = torch.randint(0, max(1, F - w), ()).item()
            feats[b, :, f0:f0 + w] = 0
        max_w = max(1, int(time_width_ratio * T))
        for _ in range(time_masks):
            w = torch.randint(0, max_w + 1, ()).item()
            t0 = torch.randint(0, max(1, T - w), ()).item()
            feats[b, t0:t0 + w, :] = 0
    return feats


@contextlib.contextmanager
def variational_noise(params, std: float):
    """Add Gaussian noise to weight matrices for one forward/backward, then remove it."""
    if std <= 0:
        yield
        return
    noises = []
    with torch.no_grad():
        for p in params:
            n = torch.randn_like(p) * std
            p.add_(n)
            noises.append((p, n))
    try:
        yield
    finally:
        with torch.no_grad():
            for p, n in noises:
                p.sub_(n)


def noam_lr(step: int, peak: float, warmup: int) -> float:
    step = max(step, 1)
    return peak * min(step / warmup, math.sqrt(warmup / step))


def build_model(model_cfg: dict, vocab_size: int) -> PerSemComS2T:
    cfg = dict(model_cfg)
    if cfg.pop("vocab_size", vocab_size) != vocab_size:
        raise ValueError(f"config vocab_size != tokenizer vocab ({vocab_size})")
    return PerSemComS2T(vocab_size=vocab_size, **cfg)


class S2TTrainer:
    def __init__(self, data_cfg: dict, model_cfg: dict, train_cfg: dict, stage: str,
                 snr_range: Optional[tuple] = None, select_snr: Optional[float] = None,
                 init_from: Optional[str] = None, num_workers: int = 2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = Tokenizer(data_cfg["tokenizer"])
        self.model = build_model(model_cfg, self.tok.vocab_size).to(self.device)
        if init_from:
            state = torch.load(init_from, map_location="cpu")["model"]
            self.model.load_state_dict(state)
        self.model.set_stage(stage)
        self.stage, self.cfg = stage, train_cfg
        self.snr_range, self.select_snr = snr_range, select_snr

        common = dict(tokenizer=self.tok, cmvn=data_cfg.get("cmvn"), feat_dir=data_cfg["feat_dir"])
        self.train_ds = ASRDataset(data_cfg["train_manifest"], min_duration=data_cfg["min_duration"],
                                   max_duration=data_cfg["max_duration"], **common)
        self.val_ds = ASRDataset(data_cfg["val_manifest"], **common)
        self.sampler = DurationBucketSampler(self.train_ds.durations, data_cfg["max_batch_seconds"])
        self.train_dl = DataLoader(self.train_ds, batch_sampler=self.sampler, collate_fn=collate_asr,
                                   num_workers=num_workers, persistent_workers=num_workers > 0)
        self.val_dl = DataLoader(self.val_ds, collate_fn=collate_asr, num_workers=num_workers,
                                 batch_sampler=DurationBucketSampler(self.val_ds.durations,
                                                                     data_cfg["max_batch_seconds"], shuffle=False))

        opt = train_cfg["optimizer"]
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.Adam(params, lr=opt["lr_peak"], betas=tuple(opt["betas"]), eps=opt["eps"],
                                    weight_decay=opt.get("weight_decay", 0.0))
        self.noise_params = [p for p in params if p.dim() > 1]
        self.scaler = torch.amp.GradScaler(enabled=train_cfg.get("amp", True) and self.device.type == "cuda")
        self.ckpt_dir = Path(train_cfg["ckpt_dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.ckpt_dir / "tb"))
        self.step, self.epoch, self.best = 0, 0, float("inf")
        if (self.ckpt_dir / "last.pt").exists():
            self._load(self.ckpt_dir / "last.pt")

    def _snr(self, batch_size: int):
        if self.stage == "pretrain":
            return None
        return sample_snr(batch_size, *self.snr_range, device=self.device)

    def _save(self, path):
        torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict(),
                    "scaler": self.scaler.state_dict(), "step": self.step, "epoch": self.epoch,
                    "best": self.best, "stage": self.stage}, path)

    def _load(self, path):
        ck = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.scaler.load_state_dict(ck["scaler"])
        self.step, self.epoch, self.best = ck["step"], ck["epoch"], ck["best"]
        print(f"resumed from {path} at epoch {self.epoch}, step {self.step}")

    @torch.no_grad()
    def validate(self, snr_db: Optional[float] = None, max_batches: Optional[int] = None) -> dict:
        self.model.eval()
        refs, hyps, mse, n = [], [], 0.0, 0
        for i, b in enumerate(self.val_dl):
            if max_batches is not None and i >= max_batches:
                break
            feats, lens = b["feats"].to(self.device), b["feat_lens"].to(self.device)
            snr = None if self.stage == "pretrain" else snr_db
            with torch.autocast(self.device.type, enabled=self.scaler.is_enabled()):
                out = self.model(feats, lens, snr)
            if self.stage != "pretrain":
                mse += self.model.syntactic_loss(out).item()
                n += 1
            ids = ctc_greedy_decode(out["log_probs"], out["lengths"], self.model.blank_id)
            hyps += [self.tok.decode(x) for x in ids]
            refs += b["texts"]
        self.model.train()
        return {"wer": wer(refs, hyps), "cer": cer(refs, hyps), "mse": mse / max(n, 1),
                "example": (refs[0], hyps[0]) if refs else None}

    def fit(self, max_epochs: int, log_every: int = 50, max_steps: Optional[int] = None):
        opt_cfg, aug = self.cfg["optimizer"], dict(self.cfg.get("spec_augment") or {})
        aug_start = aug.pop("start_step", 0)
        vn = self.cfg.get("variational_noise", {"std": 0.0, "start_step": 0})
        constant_lr = self.cfg.get("lr")  # channel stages use a constant LR, pretraining uses Noam
        self.model.train()
        while self.epoch < max_epochs:
            self.sampler.set_epoch(self.epoch)
            t0 = time.time()
            for b in self.train_dl:
                lr = constant_lr or noam_lr(self.step + 1, opt_cfg["lr_peak"], self.cfg["warmup_steps"])
                for g in self.opt.param_groups:
                    g["lr"] = lr
                feats, lens = b["feats"].to(self.device), b["feat_lens"].to(self.device)
                if aug and self.stage != "stage1" and self.step >= aug_start:
                    feats = spec_augment(feats, lens, **aug)
                tokens, tok_lens = b["tokens"].to(self.device), b["token_lens"].to(self.device)
                std = vn["std"] if self.step >= vn["start_step"] and self.stage != "stage1" else 0.0
                with variational_noise(self.noise_params, std):
                    with torch.autocast(self.device.type, enabled=self.scaler.is_enabled()):
                        out = self.model(feats, lens, self._snr(len(feats)))
                        loss = self.model.loss(out, tokens, tok_lens)
                    self.opt.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.get("grad_clip", 5.0))
                self.scaler.step(self.opt)
                self.scaler.update()
                self.step += 1
                if self.step % log_every == 0:
                    self.writer.add_scalar(f"{self.stage}/loss", loss.item(), self.step)
                    self.writer.add_scalar(f"{self.stage}/lr", lr, self.step)
                    self.writer.add_scalar(f"{self.stage}/grad_norm", gnorm.item(), self.step)
                    print(f"ep {self.epoch} step {self.step} loss {loss.item():.4f} lr {lr:.2e} "
                          f"gnorm {gnorm.item():.2f}", flush=True)
                if max_steps is not None and self.step >= max_steps:
                    break
            self.epoch += 1
            res = self.validate(self.select_snr)
            for k in ("wer", "cer", "mse"):
                self.writer.add_scalar(f"{self.stage}/val_{k}", res[k], self.step)
            print(f"== epoch {self.epoch} ({time.time() - t0:.0f}s) val WER {res['wer']:.4f} "
                  f"CER {res['cer']:.4f} MSE {res['mse']:.4f}", flush=True)
            if res["example"]:
                print(f"   ref: {res['example'][0]}\n   hyp: {res['example'][1]}", flush=True)
            if res["wer"] < self.best:
                self.best = res["wer"]
                self._save(self.ckpt_dir / "best.pt")
            self._save(self.ckpt_dir / "last.pt")
            if max_steps is not None and self.step >= max_steps:
                break
        self.writer.close()
