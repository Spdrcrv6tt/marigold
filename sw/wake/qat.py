"""Quantization-aware fine-tuning of the BN-folded wake model, and export to the integer golden format.

Fake quantization mirrors the integer rules of docs/golden-wake.md: weights int4 in [-7, 7] per output
channel (scale max|w|/7), biases int32 in units of s_x*s_w, hidden activations uint8 with a fixed per-layer
scale, input x = u - 128 at scale 1/32, output logits at 1/16 nat per integer step.
"""
import json, random, sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fe
from model import Net, fold, K_DW

S_IN, S_OUT = 1.0 / 32, 1.0 / 16


def ste_round(x):
    return x + (torch.round(x) - x).detach()


class QNet(nn.Module):
    def __init__(self, init):
        super().__init__()
        sd = torch.load(init, map_location="cpu")
        if any(k.startswith("Ws.") for k in sd):            # a QNet checkpoint
            self.kinds, self.dils = sd.pop("_kinds"), sd.pop("_dils")
            self.Ws = nn.ParameterList([nn.Parameter(sd[f"Ws.{i}"]) for i in range(len(self.kinds))])
            self.bs = nn.ParameterList([nn.Parameter(sd[f"bs.{i}"]) for i in range(len(self.kinds))])
            self.register_buffer("sa", sd["sa"])
            self.calibrated = True
            return
        m = Net(c=sd["l0.weight"].shape[0]); m.load_state_dict(sd); m.eval()
        L = fold(m)
        self.kinds = [k for k, _, _, _ in L]
        self.dils = [d for _, _, _, d in L]
        self.Ws = nn.ParameterList([nn.Parameter(W) for _, W, _, _ in L])
        self.bs = nn.ParameterList([nn.Parameter(b) for _, _, b, _ in L])
        self.register_buffer("sa", torch.ones(len(L)))        # output scale of each hidden layer
        self.calibrated = False

    def state_dict(self, *a, **k):
        sd = super().state_dict(*a, **k)
        sd["_kinds"], sd["_dils"] = self.kinds, self.dils
        return sd

    def load_state_dict(self, sd, strict=True):
        sd = {k: v for k, v in sd.items() if not k.startswith("_")}
        return super().load_state_dict(sd, strict)

    def scales_w(self, W):
        return (W.detach().abs().amax(dim=(1, 2)) / 7).clamp(min=1e-8)

    def forward(self, u, quant=True, record=None):
        x = (u.float() - 128.0) * S_IN
        x = x.transpose(1, 2)
        s_x = S_IN
        n = len(self.kinds)
        for i, (kind, W, b, d) in enumerate(zip(self.kinds, self.Ws, self.bs, self.dils)):
            if quant:
                sw = self.scales_w(W)
                Wq = ste_round(W / sw[:, None, None]).clamp(-7, 7) * sw[:, None, None]
                sb = s_x * sw
                bq = ste_round(b / sb) * sb
            else:
                Wq, bq = W, b
            if kind == "dw":
                acc = F.conv1d(F.pad(x, ((K_DW - 1) * d, 0)), Wq, bq, dilation=d, groups=Wq.shape[0])
            else:
                acc = F.conv1d(x, Wq, bq)
            if i == n - 1:
                return acc.transpose(1, 2)
            if record is not None:
                record.append(acc.detach())
            if quant:
                s_y = self.sa[i]
                x = ste_round(acc / s_y).clamp(0, 255) * s_y
                s_x = float(s_y)
            else:
                x = F.relu(acc)

    @torch.no_grad()
    def calibrate(self, rows, dev, n=48):
        rs = random.Random(5).sample(rows, n)
        recs = [[] for _ in range(len(self.kinds) - 1)]
        for r in rs:
            x, _ = sf.read(r["path"], dtype="int16")
            g = 10 ** (random.uniform(-25, 0) / 20)
            xt = torch.from_numpy(np.clip(np.round(x * g), -32768, 32767).astype(np.int64)).to(dev)
            rec = []
            self.forward(fe.features(xt)[None], quant=False, record=rec)
            for i, a in enumerate(rec):
                recs[i].append(F.relu(a).flatten())
        for i, rr in enumerate(recs):
            a = torch.cat(rr)
            a = a[a > 0]
            hi = torch.quantile(a[torch.randperm(len(a), device=a.device)[:2_000_000]].float(), 0.9999)
            self.sa[i] = (hi / 255).clamp(min=1e-6)
        self.calibrated = True
        print("activation scales", [f"{float(s):.4f}" for s in self.sa], flush=True)


def export(qnet):
    """Integer parameters: list of dict(kind, dil, W int8 (Co, Ci|1, k), b int64, M int64, S int)."""
    out = []
    s_x = S_IN
    n = len(qnet.kinds)
    for i, (kind, W, b, d) in enumerate(zip(qnet.kinds, qnet.Ws, qnet.bs, qnet.dils)):
        W = W.detach().double().cpu(); b = b.detach().double().cpu()
        sw = (W.abs().amax(dim=(1, 2)) / 7).clamp(min=1e-8)
        Wi = torch.round(W / sw[:, None, None]).clamp(-7, 7).long()
        bi = torch.round(b / (s_x * sw)).long()
        s_y = S_OUT if i == n - 1 else float(qnet.sa[i])
        r = (s_x * sw / s_y).numpy()
        S = int(np.floor(np.log2((2 ** 15 - 1) / r.max())))
        M = np.round(r * 2.0 ** S).astype(np.int64)
        assert 0 < S <= 47 and M.max() < 2 ** 15 and M.min() >= 0, (i, S, M.max())
        out.append(dict(kind=kind, dil=int(d), W=Wi.numpy().astype(np.int8), b=bi.numpy().astype(np.int64), M=M, S=S))
        s_x = s_y
    return out
