"""Conventional digital link for the baselines: LDPC (rate ~2/3) + QPSK over the same AWGN / Rayleigh
channel as PerSemCom (unit-power symbols, per-complex-symbol noise variance 10^(-SNR/10), perfect CSI).

Code: regular (3, 9) LDPC, n=1530, k=1022 (rate 0.668), from pyldpc; systematic with the message in the
first k codeword bits. Decoder: normalized min-sum (alpha 0.75) on exact QPSK LLRs, vectorized over blocks.
A block whose final hard decision fails the parity check is reported as failed (stands in for a CRC).
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

N, DV, DC = 1530, 3, 9


@lru_cache(maxsize=1)
def _code(seed: int = 0):
    import pyldpc

    H, G = pyldpc.make_ldpc(N, DV, DC, systematic=True, sparse=True, seed=seed)
    H = np.asarray(H.todense() if hasattr(H, "todense") else H).astype(np.int64) % 2
    G = np.asarray(G.todense() if hasattr(G, "todense") else G).astype(np.int64) % 2
    k = G.shape[1]
    assert np.array_equal(G[:k], np.eye(k, dtype=np.int64)), "expected message in the first k bits"
    assert not ((H @ G) % 2).any()
    return H, G


@dataclass
class LinkResult:
    bits: np.ndarray          # decoded message bits (same length as the input)
    block_ok: np.ndarray      # per-LDPC-block parity check passed
    n_symbols: int            # complex channel symbols used


class DigitalLink:
    def __init__(self, channel: str, iterations: int = 50, alpha: float = 0.75, device: str = "cpu", seed: int = 0):
        if channel not in ("awgn", "rayleigh"):
            raise ValueError(channel)
        H, G = _code(seed)
        self.channel, self.iterations, self.alpha, self.device = channel, iterations, alpha, torch.device(device)
        self.n, self.k = G.shape
        self.G = torch.from_numpy(G.T.copy()).float().to(self.device)  # (k, n)
        rows, cols = np.nonzero(H)
        self.m = H.shape[0]
        self.edge_row = torch.from_numpy(rows).to(self.device)
        self.edge_col = torch.from_numpy(cols).to(self.device)
        # per-check edge lists (each check has exactly DC edges)
        order = np.argsort(rows, kind="stable")
        self.check_edges = torch.from_numpy(order.reshape(self.m, DC)).to(self.device)
        self.H = torch.from_numpy(H).float().to(self.device)

    @property
    def rate(self) -> float:
        return self.k / self.n

    def symbols_for(self, n_bits: int) -> int:
        blocks = -(-n_bits // self.k)
        return blocks * self.n // 2

    # --- encoder / modem / channel -------------------------------------------------------------
    def _encode(self, msg: torch.Tensor) -> torch.Tensor:
        return (msg.float() @ self.G) % 2  # (blocks, n)

    def _transmit(self, cw: torch.Tensor, snr_db: float, gen: torch.Generator) -> torch.Tensor:
        """QPSK (Gray), unit energy; returns per-bit LLRs (positive favours bit 0)."""
        b = cw.view(cw.shape[0], -1, 2)
        s = torch.complex(1 - 2 * b[..., 0], 1 - 2 * b[..., 1]) / np.sqrt(2)
        var = 10 ** (-snr_db / 10)
        n = torch.complex(torch.randn(s.shape, generator=gen), torch.randn(s.shape, generator=gen)) * np.sqrt(var / 2)
        if self.channel == "awgn":
            h = torch.ones_like(s)
        else:  # Rayleigh block fading per 32-symbol frame, as in the semantic system
            nf = -(-s.shape[1] // 32)
            hf = torch.complex(torch.randn(s.shape[0], nf, generator=gen), torch.randn(s.shape[0], nf, generator=gen))
            h = (hf / np.sqrt(2)).repeat_interleave(32, dim=1)[:, : s.shape[1]]
        y = h * s + n.to(s.device)
        z = h.conj() * y  # matched filter with perfect CSI
        llr = torch.stack([z.real, z.imag], -1) * (2 * np.sqrt(2) / var)
        return llr.reshape(cw.shape[0], -1)

    # --- decoder ---------------------------------------------------------------------------------
    def _decode(self, llr: torch.Tensor):
        B = llr.shape[0]
        E = self.edge_col.numel()
        v2c = llr[:, self.edge_col]                                   # (B, E)
        c2v = torch.zeros(B, E, device=llr.device)
        for _ in range(self.iterations):
            msgs = v2c[:, self.check_edges]                           # (B, m, DC)
            sign = torch.where(msgs < 0, -1.0, 1.0)
            prod_sign = sign.prod(-1, keepdim=True)
            mag = msgs.abs()
            min1, idx1 = mag.min(-1, keepdim=True)
            mag2 = mag.scatter(-1, idx1, float("inf"))
            min2 = mag2.min(-1, keepdim=True).values
            out_mag = torch.where(torch.arange(DC, device=llr.device)[None, None, :] == idx1, min2, min1)
            out = self.alpha * prod_sign * sign * out_mag              # extrinsic (excludes own edge)
            c2v = torch.zeros(B, E, device=llr.device).scatter(1, self.check_edges.view(1, -1).expand(B, -1),
                                                               out.view(B, -1))
            total = llr.clone().index_add_(1, self.edge_col, c2v)      # (B, n) posterior
            hard = (total < 0).float()
            if not ((hard @ self.H.T) % 2).any():
                break
            v2c = total[:, self.edge_col] - c2v
        ok = ~((hard @ self.H.T) % 2).bool().any(-1)
        return hard, ok

    def send(self, bits: np.ndarray, snr_db: float, seed: int = 0) -> LinkResult:
        """bits: 1-D array of 0/1. Zero-padded to whole blocks; padding is stripped on return."""
        n_bits = len(bits)
        blocks = max(1, -(-n_bits // self.k))
        msg = np.zeros(blocks * self.k, dtype=np.float32)
        msg[:n_bits] = bits
        msg = torch.from_numpy(msg).view(blocks, self.k).to(self.device)
        gen = torch.Generator().manual_seed(seed)
        llr = self._transmit(self._encode(msg), snr_db, gen)
        hard, ok = self._decode(llr.to(self.device))
        out = hard[:, : self.k].reshape(-1)[:n_bits].cpu().numpy().astype(np.uint8)
        return LinkResult(out, ok.cpu().numpy(), blocks * self.n // 2)
