"""Marigold integer golden model of the intent encoder, value head and matcher. Contract: docs/golden-format.md.

IntIntent is the hardware reference: int16 PCM of one segment in, int8 q [D] and value tokens out, integer
arithmetic only. The weight products may run through float64 BLAS (FAST = True), exact because every partial sum
is an integer below 2^53; FAST = False uses int64 only. FloatIntent is the same graph in float64 with the same
dequantised weights (calibration of the fixed-point formats, and the float-simulated reference).

Primitives (rshift_round, isqrt64, bitlen, mult_shift, norm, lut, softmax via exp2 ROM, causal patches) are
Poinsetta's format 3.6 (D:\\poinsetta\\sw\\golden\\stream_np.py and moonshine_np.py), copied here unchanged.
"""
import math
import numpy as np

FORMAT_VERSION = "1.0"
FRAME, CHUNK = 80, 1280
W, NH, HD, NL, FF = 320, 8, 40, 2, 1280          # v1: Moonshine Streaming Tiny frontend + first 2 encoder layers
WIN = [(16, 4), (16, 0)]
V = 45
F_AUDIO, F_Q = 15, 5
I16, I8, RES_SAT = 32767, 127, (1 << 24) - 1
SCALE = HD ** -0.5


# ------------------------------------------------------------------------------------------------ primitives
def rshift_round(x, s):
    """Arithmetic right shift with round-half-up; s may be an int or an int array; s <= 0 means left shift."""
    if np.isscalar(s):
        if s <= 0:
            return x << (-s)
        return (x + (1 << (s - 1))) >> s
    return (x + (np.int64(1) << (s - 1))) >> s


def isqrt64(n):
    n = n.astype(np.int64).copy(); res = np.zeros_like(n); bit = np.int64(1) << 60
    for _ in range(31):
        t = res + bit; m = n >= t
        n = np.where(m, n - t, n); res = (res >> 1) + np.where(m, bit, 0); bit >>= 2
    return res


def bitlen(n):
    n = np.asarray(n, np.int64).copy(); r = np.zeros_like(n)
    for s in (32, 16, 8, 4, 2, 1):
        m = n >= (np.int64(1) << s); n = np.where(m, n >> s, n); r += np.where(m, s, 0)
    return r + (n > 0)


def fmt16(maxabs, margin=1.0):
    return 15 - int(math.ceil(math.log2(max(maxabs * margin, 1e-9))))


def mult_shift(m):
    m = np.asarray(m, dtype=np.float64)
    e = np.floor(np.log2(m)).astype(np.int64)
    sh = 14 - e
    M = np.rint(m * np.exp2(sh.astype(np.float64))).astype(np.int64)
    over = M >= (1 << 15)
    M = np.where(over, M >> 1, M); sh = np.where(over, sh - 1, sh)
    assert (sh >= 1).all() and (sh < 62).all()
    return M, sh


def gelu_f(x):
    from scipy.special import erf
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def silu_f(x):
    return x / (1.0 + np.exp(-x))


def causal_patches(x, k=5, s=2):
    N, C = x.shape
    xp = np.concatenate([np.zeros((k - 1, C), x.dtype), x])
    n_out = (N + s - 1) // s
    need = (n_out - 1) * s + k
    if xp.shape[0] < need:
        xp = np.concatenate([xp, np.zeros((need - xp.shape[0], C), x.dtype)])
    idx = np.arange(n_out)[:, None] * s + np.arange(k)[None]
    return xp[idx].transpose(0, 2, 1).reshape(n_out, C * k)


def band(T, past, fut):
    idx = np.arange(T)[:, None] + np.arange(-past, fut + 1)[None]
    return np.clip(idx, 0, max(T - 1, 0)), (idx >= 0) & (idx < T)


# site table: (weight, input format site, output format site); "res" = enc.res
def mac_sites():
    s = [("fe.lin", "fe.asinh", "fe.lin"), ("fe.c1", "fe.lin", "fe.c1"), ("fe.c2", "fe.c1", "enc.res")]
    for i in range(NL):
        n = f"enc{i}"
        s += [(n + ".q", n + ".ln1", n + ".q"), (n + ".k", n + ".ln1", n + ".k"), (n + ".v", n + ".ln1", n + ".v"),
              (n + ".o", n + ".v", "enc.res"), (n + ".fc1", n + ".ln2", n + ".fc1"), (n + ".fc2", n + ".fc1", "enc.res")]
    s += [("proj", "enc.ln", "proj"), ("ctc", "enc.ln", "ctc")]
    return s


def norm_sites():
    s = [("fe.cmvn", "AUDIO")]
    for i in range(NL):
        s += [(f"enc{i}.ln1", "enc.res"), (f"enc{i}.ln2", "enc.res")]
    return s + [("enc.ln", "enc.res"), ("out", "proj")]


# ------------------------------------------------------------------------------------------------ float reference
class FloatIntent:
    """P: dequantised weights '<w>' [out, in] (+ '<w>.b'), norm gains '<n>.g', 'fe.k'. rec(site, x): calibration."""

    def __init__(self, P, rec=None):
        self.P = P; self.rec = rec or (lambda s, x: None)

    def lin(self, x, w, site=None):
        y = x @ self.P[w].T
        if w + ".b" in self.P:
            y = y + self.P[w + ".b"]
        if site:
            self.rec(site, y)
        return y

    def ln(self, x, n, eps=1e-5):
        m = x.mean(-1, keepdims=True); c = x - m
        y = c / np.sqrt((c * c).mean(-1, keepdims=True) + eps)
        if n + ".g" in self.P:
            y = y * self.P[n + ".g"]
        self.rec(n, y); return y

    def encode(self, pcm):
        x = np.asarray(pcm, np.float64) / 32768.0
        x = np.concatenate([x, np.zeros((-len(x)) % FRAME)]).reshape(-1, FRAME)
        m = x.mean(-1, keepdims=True); c = x - m
        y = c / np.sqrt((c * c).mean(-1, keepdims=True) + 1e-6); self.rec("fe.cmvn", y)
        y = np.arcsinh(self.P["fe.k"] * y); self.rec("fe.asinh", y)
        h = silu_f(self.lin(y, "fe.lin", "fe.lin"))
        h = silu_f(self.lin(causal_patches(h), "fe.c1", "fe.c1"))
        x = self.lin(causal_patches(h), "fe.c2"); self.rec("enc.res", x)
        T = x.shape[0]
        for i, (p, f) in enumerate(WIN):
            n = f"enc{i}"; y = self.ln(x, n + ".ln1")
            q, k, v = (self.lin(y, n + c, n + c).reshape(T, NH, HD) for c in (".q", ".k", ".v"))
            idx, valid = band(T, p, f)
            s = np.einsum("qhd,qwhd->hqw", q, k[idx]) * SCALE
            s = np.where(valid[None], s, -np.inf)
            s = s - s.max(-1, keepdims=True); e = np.exp(s); e = e / e.sum(-1, keepdims=True)
            a = np.einsum("hqw,qwhd->qhd", e, v[idx]).reshape(T, W)
            x = x + self.lin(a, n + ".o"); self.rec("enc.res", x)
            y = self.ln(x, n + ".ln2")
            x = x + self.lin(gelu_f(self.lin(y, n + ".fc1", n + ".fc1")), n + ".fc2"); self.rec("enc.res", x)
        return self.ln(x, "enc.ln")

    def head(self, e):
        T = e.shape[0]
        m = e.sum(0) / 2.0 ** int(bitlen(np.int64(max(T - 1, 0))))
        p = self.lin(m[None], "proj", "proj")
        return self.ln(p, "out")[0]

    def logits(self, e):
        return self.lin(e, "ctc", "ctc")


# ------------------------------------------------------------------------------------------------ integer model
class IntIntent:
    FAST = True

    def __init__(self, G, trace=None):
        self.G = G; self.trace = trace
        self.F = {k[2:]: int(v) for k, v in G.items() if k.startswith("F:")}
        self._wf = {}
        assert int(G["version"]) == 1, "not a Marigold format-1 file"

    def t(self, name, x):
        if self.trace is not None:
            self.trace[name] = np.array(x)

    def matmul(self, x, w):
        if self.FAST:
            wf = self._wf.get(w)
            if wf is None:
                wf = self._wf[w] = np.ascontiguousarray(self.G[w + ".q"].astype(np.float64).T)
            return np.rint(x.astype(np.float64) @ wf).astype(np.int64)
        return x.astype(np.int64) @ self.G[w + ".q"].astype(np.int64).T

    def mac(self, x, w, sat=I16):
        G = self.G; acc = self.matmul(x, w)
        if w + ".bi" in G:
            acc = acc + G[w + ".bi"]
        return np.clip(rshift_round(acc * G[w + ".M"], G[w + ".sh"]), -sat, sat)

    def lut(self, x, f, table, R=3, tails=False, f_out=None):
        T = self.G[table]; f_out = f if f_out is None else f_out
        fw = max(f, 8 - R); x = x << (fw - f)
        lim = np.int64(1) << (fw + R)
        u = np.clip(x, -lim, lim - 1) + lim; sh = fw + R - 8
        idx = u >> sh
        y14 = T[idx] + (((T[idx + 1] - T[idx]) * (u & ((np.int64(1) << sh) - 1))) >> sh)
        y = rshift_round(y14, 14 - f_out)
        if tails:
            y = np.where(x >= lim, rshift_round(x, fw - f_out), y); y = np.where(x < -lim, 0, y)
        return y

    def norm(self, x, n, f_in):
        G = self.G; f_out = self.F[n]; N = x.shape[-1]
        S = x.sum(-1, keepdims=True); mean = (2 * S + N) // (2 * N); d = x - mean
        var = (d * d).sum(-1, keepdims=True) // N + G[n + ".eps"]; var = np.maximum(var, 1)
        s = (61 - bitlen(var)) // 2 * 2
        sd = isqrt64(np.where(s >= 0, var << np.maximum(s, 0), var >> np.maximum(-s, 0)))
        rec = (np.int64(1) << 61) // sd
        y = rshift_round(d * rec, 61 - s // 2 - 12)
        return np.clip(rshift_round(y * G[n + ".G"], 12 + int(G[n + ".fg"]) - f_out), -I16, I16)

    def softmax_mix(self, sc, v, valid, fq, fk):
        G = self.G
        mx = np.where(valid, sc, np.iinfo(np.int64).min).max(-1, keepdims=True); dlt = np.where(valid, mx - sc, 0)
        tt = rshift_round(dlt * G["sm.C"], fq + fk + 4)
        e = G["sm.exp2"][(tt & 4095) >> 4] >> np.minimum(tt >> 12, 40)
        e = np.where(valid, e, 0)
        S = e.sum(-1, keepdims=True)
        if self.FAST:
            num = np.rint(np.einsum("...k,...kd->...d", e.astype(np.float64), v.astype(np.float64))).astype(np.int64)
        else:
            num = np.einsum("...k,...kd->...d", e, v)
        return np.clip(rshift_round(num * ((np.int64(1) << 44) // S), 44), -I16, I16)

    def band_attn(self, q, k, v, fq, fk, past, fut):
        T = q.shape[0]
        idx = np.arange(T)[:, None] + np.arange(-past, fut + 1)[None]; valid = (idx >= 0) & (idx < T)
        idx = np.clip(idx, 0, T - 1)
        K = k[idx].transpose(0, 2, 1, 3); Vv = v[idx].transpose(0, 2, 1, 3)      # [T, NH, Wn, HD]
        sc = np.einsum("nhd,nhwd->nhw", q, K)
        o = self.softmax_mix(sc, Vv, np.broadcast_to(valid[:, None, :], sc.shape), fq, fk)
        return o.reshape(T, W)

    def encode(self, pcm):
        """Segment PCM (int16) -> e [T, W] (int16 at F[enc.ln])."""
        F = self.F
        pcm = np.asarray(pcm, np.int64)
        pcm = np.concatenate([pcm, np.zeros((-len(pcm)) % FRAME, np.int64)])
        fr = pcm.reshape(-1, FRAME)
        y = self.norm(fr, "fe.cmvn", F_AUDIO); self.t("fe.cmvn", y)
        y = self.lut(y, F["fe.cmvn"], "lut.asinh", R=4, f_out=F["fe.asinh"]); self.t("fe.asinh", y)
        h = self.mac(y, "fe.lin"); self.t("fe.lin", h)
        h = self.lut(h, F["fe.lin"], "lut.silu", 3, True)
        c = self.mac(causal_patches(h), "fe.c1"); self.t("fe.c1", c)
        c = self.lut(c, F["fe.c1"], "lut.silu", 3, True)
        x = self.mac(causal_patches(c), "fe.c2", RES_SAT); self.t("fe.feat", x)
        fr_ = F["enc.res"]
        for i, (p, f) in enumerate(WIN):
            n = f"enc{i}"
            y = self.norm(x, n + ".ln1", fr_)
            q = self.mac(y, n + ".q").reshape(-1, NH, HD)
            k = self.mac(y, n + ".k").reshape(-1, NH, HD)
            v = self.mac(y, n + ".v").reshape(-1, NH, HD)
            a = self.band_attn(q, k, v, F[n + ".q"], F[n + ".k"], p, f)
            h = np.clip(x + self.mac(a, n + ".o", RES_SAT), -RES_SAT, RES_SAT)
            y = self.norm(h, n + ".ln2", fr_)
            y = self.lut(self.mac(y, n + ".fc1"), F[n + ".fc1"], "lut.gelu", 3, True)
            x = np.clip(h + self.mac(y, n + ".fc2", RES_SAT), -RES_SAT, RES_SAT); self.t(f"enc{i}.out", x)
        e = self.norm(x, "enc.ln", fr_); self.t("enc.ln", e)
        return e

    def head(self, e):
        """e [T, W] -> q int8 [D]."""
        T = e.shape[0]
        s = e.sum(0)
        b = int(bitlen(np.int64(max(T - 1, 0))))
        m = np.clip(rshift_round(s, b), -I16, I16); self.t("pool.m", m)
        p = self.mac(m[None], "proj"); self.t("proj", p[0])
        q = np.clip(self.norm(p, "out", self.F["proj"])[0], -I8, I8); self.t("q", q)
        return q.astype(np.int64)

    def ctc_ids(self, e):
        lg = self.mac(e, "ctc")
        k = lg.argmax(1)                      # first maximum: lowest index wins ties
        self.t("ctc.k", k)
        return k

    def run(self, pcm):
        e = self.encode(pcm)
        return self.head(e), self.ctc_ids(e)


# ------------------------------------------------------------------------------------------------ export
def build_int(Q, maxabs, margin=1.25):
    """Q: {'<w>.q' int, '<w>.s' scale [out], '<w>.b' float bias, '<n>.g' gains, 'fe.k'}; maxabs: {site: float}."""
    G = {"version": np.int64(1)}; F = {"AUDIO": F_AUDIO}
    for site, m in maxabs.items():
        F[site] = 22 - int(math.ceil(math.log2(m))) if site == "enc.res" else fmt16(m, margin)
    F["out"] = F_Q
    for w, fin, fout in mac_sites():
        s = Q[w + ".s"]; G[w + ".q"] = Q[w + ".q"]
        G[w + ".M"], G[w + ".sh"] = mult_shift(s * 2.0 ** (F[fout] - F[fin]))
        if w + ".b" in Q:
            G[w + ".bi"] = np.rint(Q[w + ".b"] / (s * 2.0 ** -F[fin])).astype(np.int64)
    for n, src in norm_sites():
        g = Q.get(n + ".g", np.ones(FRAME if n == "fe.cmvn" else Q["proj.q"].shape[0] if n == "out" else W))
        fg = 14 if n in ("fe.cmvn", "out") else fmt16(np.abs(g).max())
        G[n + ".G"] = np.clip(np.rint(g * 2.0 ** fg), -32767, 32767).astype(np.int64); G[n + ".fg"] = np.int64(fg)  # int16 (core)
        eps = 1e-6 if n == "fe.cmvn" else 1e-5
        G[n + ".eps"] = np.int64(max(1, round(eps * 2.0 ** (2 * F[src]))))
    xs = (np.arange(513) - 256) / 32.0
    G["lut.silu"] = np.rint(silu_f(xs) * 16384).astype(np.int64)
    G["lut.gelu"] = np.rint(gelu_f(xs) * 16384).astype(np.int64)
    G["lut.asinh"] = np.rint(np.arcsinh(Q["fe.k"] * (np.arange(513) - 256) / 16.0) * 16384).astype(np.int64)
    G["sm.exp2"] = np.rint(65536 * 2.0 ** (-np.arange(256) / 256.0)).astype(np.int64)
    G["sm.C"] = np.int64(round(65536 * SCALE / math.log(2.0)))
    for k, v in F.items():
        G["F:" + k] = np.int64(v)
    return G


def Q_to_P(Q):
    P = {"fe.k": float(Q["fe.k"])}
    for w, _, _ in mac_sites():
        P[w] = Q[w + ".q"].astype(np.float64) * Q[w + ".s"][:, None]
        if w + ".b" in Q:
            P[w + ".b"] = Q[w + ".b"]
    for n, _ in norm_sites():
        if n + ".g" in Q:
            P[n + ".g"] = Q[n + ".g"]
    return P
