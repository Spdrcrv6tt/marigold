"""Bit-exact integer reference of the PDM front end (rtl/src/pdm_frontend.sv).

pdm bits (0/1) --CIC5 /R--> DW bits --half-band /2--> DW bits --comp FIR /2--> 16 bit
--DC block--> 16-bit PCM at f_pdm / (4R).

All arithmetic is integer; the RTL must match sample for sample.
Also holds the sigma-delta modulator used to make test stimulus (float, not part of the chip).
"""
import os
import numpy as np
from fixed import sat, wrap, rshift_round, read_mem

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")

DEFAULTS = dict(R=48, N=5, CIC_W=29, CIC_SHIFT=9, DW=20, OW=16, HB_SHIFT=15, FIR_SHIFT=19, DC_K=7)


def load_coefs():
    return read_mem(os.path.join(SRC, "pdm_hb_coef.mem")), read_mem(os.path.join(SRC, "pdm_fir_coef.mem"))


def cic(bits, R=48, N=5, CIC_W=29):
    """N pipelined integrators at the PDM rate, decimate by R, N combs. Returns Python ints.

    Integrator k adds the previous-cycle value of integrator k-1 (registered chain):
      i1[n] = i1[n-1] + x[n],  ik[n] = ik[n-1] + i(k-1)[n-1]
    The decimator takes iN as it stands when the R-th bit of a group is accepted (before
    that bit's own update), then the combs run on that value with zero initial state.
    Arithmetic wraps at CIC_W bits; int64 wrap-around here is consistent with that.
    """
    x = np.where(np.asarray(bits, dtype=np.int64) > 0, 1, -1).astype(np.int64)
    s = x
    with np.errstate(over="ignore"):
        for _ in range(N):
            s = np.cumsum(s, dtype=np.int64)           # value after update n
            s = np.concatenate(([0], s[:-1]))           # registered: what is visible at bit n
        v = s[R - 1::R].copy()
        for _ in range(N):
            v = np.diff(v, prepend=np.int64(0))
    return [wrap(int(t), CIC_W) for t in v]


def fir_decim2(x, coef, shift, ow, gain=0):
    """y[m] = sat_ow(round((sum_i coef[i] * x[2m+1-i]) >> (shift-gain))), x[<0] = 0."""
    nt = len(coef)
    xa = np.concatenate((np.zeros(nt - 1, dtype=np.int64), np.asarray(x, dtype=np.int64)))
    c = np.asarray(coef[::-1], dtype=np.int64)
    out = []
    for n in range(1, len(x), 2):
        acc = int(np.dot(xa[n:n + nt], c))
        out.append(sat(rshift_round(acc, shift - gain), ow))
    return out


def dc_block(x, K=7, ow=16):
    """y = x - x[-1] + (1 - 2^-K) y[-1], state kept with K fraction bits."""
    s, xp, out = 0, 0, []
    for v in x:
        s = s - (s >> K) + ((v - xp) << K)
        xp = v
        out.append(sat(rshift_round(s, K), ow))
    return out


def pdm_decimate(bits, gain=0, dc_en=True, stages=False, **kw):
    p = dict(DEFAULTS)
    p.update(kw)
    hb, fir = load_coefs()
    c = cic(bits, p["R"], p["N"], p["CIC_W"])
    a = [sat(rshift_round(v, p["CIC_SHIFT"]), p["DW"]) for v in c]
    b = fir_decim2(a, hb, p["HB_SHIFT"], p["DW"])
    d = fir_decim2(b, fir, p["FIR_SHIFT"], p["OW"], gain)
    y = dc_block(d, p["DC_K"], p["OW"]) if dc_en else d
    if stages:
        return dict(cic=c, a=a, hb=b, fir=d, out=y)
    return y


# ---------------------------------------------------------------- stimulus (not hardware)
def sdm_ntf(order=4, h_inf=1.5):
    """Butterworth high-pass NTF with out-of-band gain h_inf (Lee criterion)."""
    from scipy import signal
    lo, hi = 0.01, 0.45
    for _ in range(60):
        fc = 0.5 * (lo + hi)
        b, a = signal.butter(order, 2 * fc, "high")
        b = b / b[0]
        g = abs(np.polyval(b, -1.0) / np.polyval(a, -1.0))
        if g > h_inf:
            hi = fc
        else:
            lo = fc
    return b, a


def sdm(u, order=4, h_inf=1.5, dither=0.0, seed=1):
    """1-bit sigma-delta modulator (error-feedback form). u in [-1, 1] at the PDM rate.
    Returns a uint8 array of 0/1. A stand-in for the 4th-order modulator in a MEMS mic."""
    b, a = sdm_ntf(order, h_inf)
    g = [float(b[k] - a[k]) for k in range(1, order + 1)]
    aa = [float(a[k]) for k in range(1, order + 1)]
    e = [0.0] * order
    w = [0.0] * order
    out = np.empty(len(u), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    dz = rng.uniform(-dither, dither, len(u)).tolist() if dither else None
    ul = np.asarray(u, dtype=float).tolist()
    for n in range(len(ul)):
        wn = 0.0
        for k in range(order):
            wn += g[k] * e[k] - aa[k] * w[k]
        y = ul[n] + wn
        if y > 4.0:
            y = 4.0
        elif y < -4.0:
            y = -4.0
        v = 1.0 if (y + (dz[n] if dz is not None else 0.0)) >= 0 else -1.0
        e = [v - y] + e[:-1]
        w = [wn] + w[:-1]
        out[n] = 1 if v > 0 else 0
    return out


def upsample_to_pdm(x, r_total=192):
    """Band-limited upsample of 16 kHz float audio to the PDM rate (polyphase)."""
    from scipy import signal
    return signal.resample_poly(np.asarray(x, dtype=float), r_total, 1)
