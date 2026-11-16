"""Marigold wake spotter: integer golden model (docs/golden-wake.md). Pure numpy, bit-exact reference for
rtl/wake/. Front end: sw/wake/fe.py. Parameters: data/wake/golden/wake_int.npz (export.py).

  run(x, tpl, sens, P) -> dict(fires=[frame t], wake_k=[sample index], u, z, best)
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import fe  # noqa: E402

NEG = -32768
K_DW = 5
LAT = 320            # wake for frame t rides on sample 320 t + 559 + LAT


def load_params(path):
    d = np.load(path, allow_pickle=False)
    n = int(d["n_layers"])
    L = []
    for i in range(n):
        L.append(dict(kind=str(d[f"kind{i}"]), dil=int(d[f"dil{i}"]), W=d[f"W{i}"].astype(np.int64),
                      b=d[f"b{i}"].astype(np.int64), M=d[f"M{i}"].astype(np.int64), S=int(d[f"S{i}"])))
    return L


def requant(acc, M, S, lo, hi):
    """(acc * M + 2^(S-1)) >> S (arithmetic), clamped. acc (T, C) int64, M (C,), S scalar."""
    y = (acc * M[None, :] + (1 << (S - 1))) >> S
    return np.clip(y, lo, hi)


def net(P, U):
    """U (T, 64) uint8 codes -> z (T, 40) int64 logits in 1/16 nat. Causal; history before frame 0 is 0."""
    x = U.astype(np.int64) - 128
    T = len(x)
    for i, l in enumerate(P):
        W, b = l["W"], l["b"]
        if l["kind"] == "dw":
            d = l["dil"]
            xp = np.concatenate([np.zeros(((K_DW - 1) * d, x.shape[1]), np.int64), x])
            acc = np.zeros_like(x) + b[None, :]
            for k in range(K_DW):          # tap k reads frame t - (4 - k) d
                acc += W[None, :, 0, k] * xp[k * d: k * d + T]
        else:
            acc = x @ W[:, :, 0].T + b[None, :]
        if l["kind"] == "out":
            return requant(acc, l["M"], l["S"], -32768, 32767)
        x = requant(acc, l["M"], l["S"], 0, 255)


def parse_tpl(tpl):
    tpl = bytes(tpl)
    assert tpl[0] == 1 and 1 <= tpl[1] <= 4, "wake template: bad format or n_var"
    nv, clip, refr = tpl[1], tpl[2], tpl[3]
    o, var = 4, []
    for _ in range(nv):
        k = tpl[o]
        assert 1 <= k <= 30
        var.append([int(p) for p in tpl[o + 1: o + 1 + k]]); o += 1 + k
    assert sum(len(v) for v in var) <= 60 and o == len(tpl)
    return var, clip, refr


def kws(z, tpl, sens):
    """Frame-by-frame CTC keyword DP with free start, per-frame deficit clip, reset and refractory on a fire.
    Returns (fires, best) with best[t] = max over variants of (S_v + K_v * sens), int64."""
    var, clip, refr_n = parse_tpl(tpl)
    A = [np.full(2 * len(v) + 1, NEG, np.int64) for v in var]
    labs = [np.array([0 if s % 2 == 0 else v[s // 2] for s in range(2 * len(v) + 1)]) for v in var]
    skip = [np.array([s % 2 == 1 and s >= 3 and v[s // 2] != v[s // 2 - 1] for s in range(2 * len(v) + 1)]) for v in var]
    fires, best = [], np.zeros(len(z), np.int64)
    refr = 0
    for t in range(len(z)):
        zt = z[t].astype(np.int64)
        d = np.maximum(zt - zt.max(), -clip)
        bt = None
        for vi, v in enumerate(var):
            a = A[vi]
            c = a.copy()
            c[1:] = np.maximum(c[1:], a[:-1])
            c[2:] = np.where(skip[vi][2:], np.maximum(c[2:], a[:-2]), c[2:])
            c[0] = max(c[0], 0); c[1] = max(c[1], 0)
            a = np.maximum(c + d[labs[vi]], NEG)
            A[vi] = a
            K = len(v)
            sc = int(max(a[2 * K - 1], a[2 * K])) + K * int(sens)
            bt = sc if bt is None else max(bt, sc)
        best[t] = bt
        if refr > 0:
            refr -= 1
        elif bt >= 0:
            fires.append(t)
            for a in A:
                a[:] = NEG
            refr = refr_n
    return fires, best


PREROLL = 10        # format 2: frames between a frame's features and its network/DP run (the VAD look-ahead)


def vad_at(vad, n):
    """in_vad as seen with sample n: the flags of VAD frame floor(n / 256) - 1 (held), 0 before the first frame."""
    f = n // 256 - 1
    return int(vad[f]) if 0 <= f < len(vad) else 0


def active_frames(n_frames, vad, D=PREROLL):
    """Format 2 gating. v(t') = in_vad at sample 320 t' + 559 (the trigger of real-time frame t'). Frame t is run at
    real-time frame t + D, and only if v(t), ..., v(t + D) has a 1. Frames whose t + D is past the stream are not run."""
    v = np.array([vad_at(vad, 320 * tt + 559) for tt in range(n_frames)], np.int64)
    act = np.zeros(n_frames, bool)
    for t in range(n_frames - D):
        act[t] = v[t:t + D + 1].any()
    return act


def runs_of(act):
    out, t = [], 0
    while t < len(act):
        if act[t]:
            a = t
            while t < len(act) and act[t]:
                t += 1
            out.append((a, t))
        else:
            t += 1
    return out


def run2(x, vad, tpl, sens, P, D=PREROLL):
    """Format 2 (VAD-gated). x: samples from the first sample with en = 1; vad: vad_o per 256-sample VAD frame from
    the same start. Each run of active frames starts with zero network history, fresh DP states and refractory 0.
    A fire for frame t raises wake at sample k = 320 (t + D) + 879."""
    x = np.asarray(x, np.int64)
    U = fe.features(x)
    act = active_frames(len(U), vad, D)
    z = np.zeros((len(U), 40), np.int64)
    best = np.zeros(len(U), np.int64)
    fires = []
    for a, b in runs_of(act):
        z[a:b] = net(P, U[a:b])
        f, bt = kws(z[a:b], tpl, sens)
        best[a:b] = bt
        fires += [a + t for t in f]
    k = [320 * (t + D) + 879 for t in fires]
    return dict(fires=fires, wake_k=[kk for kk in k if kk < len(x)], u=U, z=z, best=best, act=act)


def run(x, tpl, sens, P):
    """x: int16 samples from the first sample with en = 1 (index 0). Returns fires and wake sample indices
    (only those inside the stream: k < len(x))."""
    x = np.asarray(x, np.int64)
    U = fe.features(x)
    z = net(P, U)
    fires, best = kws(z, tpl, sens)
    k = [320 * t + 559 + LAT for t in fires]
    return dict(fires=fires, wake_k=[kk for kk in k if kk < len(x)], u=U, z=z, best=best)
