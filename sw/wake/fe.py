"""Marigold wake spotter front end: exact integer log-mel features (docs/golden-wake.md, section FE).

One network frame t (20 ms) covers samples [320t, 320t+560): FE frame a = x[320t : 320t+400],
FE frame b = x[320t+160 : 320t+560]. Both are Hann-windowed (Q15), packed as real and imaginary
parts of one 512-point complex radix-2 DIT FFT (int64, rounding only at twiddle products), split
into the two real spectra with adds only, magnitude by max + 3/8 min, 32 triangular mel bands
(60 to 7600 Hz, 8-bit weights), log2 with a 16-entry table, offset and clamp to uint8.
Works on numpy arrays or torch tensors (CPU or CUDA) with identical integer results.
"""
import math
import numpy as np

SR, HOP, WIN, NFFT, NB = 16000, 160, 400, 512, 32
F_LO, F_HI, FOFF = 60.0, 7600.0, 96
PAIR = 2 * HOP  # samples per network frame

# ---- tables (all integers; the RTL ROMs hold exactly these)
WIN_Q15 = np.array([int(round(32767 * math.sin(math.pi * (i + 0.5) / WIN) ** 2)) for i in range(WIN)], np.int64)
SIN_Q = np.array([int(round(32767 * math.sin(2 * math.pi * i / NFFT))) for i in range(NFFT // 4 + 1)], np.int64)


def twiddle(k):
    """cos and sin of 2*pi*k/512 for k = 0..255, Q15, from the quarter-wave table SIN_Q."""
    k = np.asarray(k)
    c = np.where(k <= 128, SIN_Q[np.clip(128 - k, 0, 128)], -SIN_Q[np.clip(k - 128, 0, 128)])
    s = np.where(k <= 128, SIN_Q[np.clip(k, 0, 128)], SIN_Q[np.clip(256 - k, 0, 128)])
    return c.astype(np.int64), s.astype(np.int64)


def _mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)


def _imel(m):
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def mel_table():
    """Per FFT bin k: (i_k, wu_k). Bin k adds mag*wu to band i (if i < NB) and mag*(256-wu) to band
    i-1 (if i >= 1). i_k = 255: bin unused. Returns (band index array, weight array, k_min, k_max)."""
    m0, m1 = _mel(F_LO), _mel(F_HI)
    c = [_imel(m0 + j * (m1 - m0) / (NB + 1)) for j in range(NB + 2)]
    bi = np.full(NFFT // 2 + 1, 255, np.int64)
    wu = np.zeros(NFFT // 2 + 1, np.int64)
    for k in range(NFFT // 2 + 1):
        f = k * SR / NFFT
        if f <= c[0] or f >= c[NB + 1]:
            continue
        i = max(j for j in range(NB + 1) if c[j] <= f)
        bi[k] = i
        wu[k] = int(round(256 * (f - c[i]) / (c[i + 1] - c[i])))
    used = np.nonzero(bi != 255)[0]
    return bi, wu, int(used[0]), int(used[-1])


MEL_I, MEL_WU, K_MIN, K_MAX = mel_table()
LOG_LUT = np.array([int(round(16 * math.log2(1 + f / 16))) for f in range(16)], np.int64)


def _bitrev(n, bits):
    return int(format(n, f"0{bits}b")[::-1], 2)


BITREV = np.array([_bitrev(i, 9) for i in range(NFFT)], np.int64)


class _NP:
    def __init__(self):
        self.t = np

    def asarr(self, a, like=None):
        return np.asarray(a, dtype=np.int64)

    zeros = staticmethod(lambda shape, like: np.zeros(shape, np.int64))
    cat = staticmethod(lambda xs, axis: np.concatenate(xs, axis))
    maximum, minimum, where, abs = map(staticmethod, (np.maximum, np.minimum, np.where, np.abs))

    @staticmethod
    def frexp_e(e):
        return np.frexp(e.astype(np.float64))[1].astype(np.int64)

    @staticmethod
    def clip(a, lo, hi):
        return np.clip(a, lo, hi)


class _TORCH:
    def __init__(self, torch):
        self.t = torch

    def asarr(self, a, like):
        return self.t.as_tensor(np.asarray(a), dtype=self.t.int64, device=like.device)

    def zeros(self, shape, like):
        return self.t.zeros(shape, dtype=self.t.int64, device=like.device)

    def cat(self, xs, axis):
        return self.t.cat(xs, axis)

    def maximum(self, a, b):
        return self.t.maximum(a, b)

    def minimum(self, a, b):
        return self.t.minimum(a, b)

    def where(self, c, a, b):
        return self.t.where(c, a, b)

    def abs(self, a):
        return a.abs()

    def frexp_e(self, e):
        return self.t.frexp(e.double())[1].long()

    def clip(self, a, lo, hi):
        return a.clamp(lo, hi)


def _ns(x):
    if isinstance(x, np.ndarray):
        return _NP()
    import torch
    return _TORCH(torch)


def frames_pairs(x, n_pairs=None):
    """x: int16-valued samples (..., n). Returns (..., P, 2, 400) int64 windows of frames a, b."""
    ns = _ns(x)
    n = x.shape[-1]
    P = (n - (PAIR + 240)) // PAIR + 1 if n_pairs is None else n_pairs
    if P <= 0:
        raise ValueError("too short")
    idx = (np.arange(P)[:, None, None] * PAIR + np.array([0, HOP])[None, :, None] + np.arange(WIN)[None, None, :])
    return x[..., ns.asarr(idx, x)] if ns.t is not np else x[..., idx].astype(np.int64)


def fft512(re, im, ns):
    """In-place style radix-2 DIT on (..., 512) int64 arrays already in natural order (bit reversal
    applied here). Rounding: (prod_sum + 2^14) >> 15 per twiddle product pair."""
    br = ns.asarr(BITREV, re)
    re, im = re[..., br], im[..., br]
    shp = re.shape[:-1]
    for s in range(1, 10):
        m, h = 1 << s, 1 << (s - 1)
        c, sn = twiddle(np.arange(h) * (NFFT // m))
        c, sn = ns.asarr(c, re), ns.asarr(sn, re)
        r4, i4 = re.reshape(*shp, NFFT // m, 2, h), im.reshape(*shp, NFFT // m, 2, h)
        ar, ai, b_r, b_i = r4[..., 0, :], i4[..., 0, :], r4[..., 1, :], i4[..., 1, :]
        tr = (b_r * c + b_i * sn + 16384) >> 15
        ti = (b_i * c - b_r * sn + 16384) >> 15
        re = ns.cat([(ar + tr)[..., None, :], (ar - tr)[..., None, :]], -2).reshape(*shp, NFFT)
        im = ns.cat([(ai + ti)[..., None, :], (ai - ti)[..., None, :]], -2).reshape(*shp, NFFT)
    return re, im


def log_code(E, ns):
    """E >= 0 int64 -> L = 16*p + LUT[f] (0 for E == 0)."""
    Ep = ns.maximum(E, E * 0 + 1)
    p = ns.frexp_e(Ep) - 1
    sh_r = ns.maximum(p - 4, p * 0)
    sh_l = ns.maximum(4 - p, p * 0)
    f = ((Ep >> sh_r) << sh_l) & 15
    lut = ns.asarr(LOG_LUT, E)
    L = 16 * p + lut[f]
    return ns.where(E > 0, L, E * 0)


def features(x, n_pairs=None):
    """x: (..., n) samples (int16 values, numpy int or torch long). Returns (..., P, 64) uint8 codes:
    [frame a bands 0..31, frame b bands 0..31] per network frame."""
    ns = _ns(x)
    fr = frames_pairs(x, n_pairs)                      # (..., P, 2, 400)
    w = ns.asarr(WIN_Q15, fr)
    y = (fr * w + 16384) >> 15
    z = ns.zeros(fr.shape[:-2] + (NFFT - WIN,), fr)
    re = ns.cat([y[..., 0, :], z], -1)
    im = ns.cat([y[..., 1, :], z], -1)
    Zr, Zi = fft512(re, im, ns)
    k = np.arange(K_MIN, K_MAX + 1)
    kk, nk = ns.asarr(k, Zr), ns.asarr((NFFT - k) % NFFT, Zr)
    zr, zi, zrn, zin = Zr[..., kk], Zi[..., kk], Zr[..., nk], Zi[..., nk]
    spec = []
    for (r, i) in ((zr + zrn, zi - zin), (zi + zin, zrn - zr)):
        a, b = ns.abs(r), ns.abs(i)
        spec.append(ns.maximum(a, b) + ((3 * ns.minimum(a, b)) >> 3))
    out = []
    bi, wu = MEL_I[K_MIN:K_MAX + 1], MEL_WU[K_MIN:K_MAX + 1]
    # dense (NB, nk) weight matrix; integer matmul by accumulation keeps it exact
    Wm = np.zeros((NB, len(k)), np.int64)
    for j in range(len(k)):
        if bi[j] < NB:
            Wm[bi[j], j] += wu[j]
        if 1 <= bi[j] <= NB:
            Wm[bi[j] - 1, j] += 256 - wu[j]
    for mag in spec:
        if ns.t is np:
            E = mag @ Wm.T
        else:
            E = (mag.unsqueeze(-2) * ns.asarr(Wm, mag)).sum(-1)   # exact int64
        L = log_code(E, ns)
        out.append(ns.clip((L >> 1) - FOFF, 0, 255))
    return ns.cat(out, -1)


if __name__ == "__main__":
    import sys, time
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(16000 * 2) * 3000).clip(-32768, 32767).astype(np.int64)
    t0 = time.time(); F = features(x); t1 = time.time()
    print("np", F.shape, F.min(), F.max(), f"{t1 - t0:.3f}s", "kmin/kmax", K_MIN, K_MAX, "lut", LOG_LUT.tolist())
    import torch
    xt = torch.tensor(x)
    Ft = features(xt).numpy()
    print("torch cpu equal:", np.array_equal(F, Ft))
    if torch.cuda.is_available():
        print("torch cuda equal:", np.array_equal(F, features(xt.cuda()).cpu().numpy()))
    # float sanity: compare frame a spectrum with numpy rfft (x2 from the pair split)
    fr = frames_pairs(x)[0, 0]
    yv = (fr * WIN_Q15 + 16384) >> 15
    ref = np.abs(np.fft.rfft(yv, 512)) * 2
    re = np.concatenate([yv, np.zeros(112, np.int64)]); im = np.concatenate([frames_pairs(x)[0, 1] * WIN_Q15 + 16384 >> 15, np.zeros(112, np.int64)])
    Zr, Zi = fft512(re[None], im[None], _NP())
    k = np.arange(1, 256)
    Ar = Zr[0, k] + Zr[0, 512 - k]; Ai = Zi[0, k] - Zi[0, 512 - k]
    print("max rel err |A| vs rfft:", np.max(np.abs(np.hypot(Ar, Ai) - ref[k]) / (ref[k] + 1e-9)))
