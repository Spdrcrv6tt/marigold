"""Try VAD decision rules on the cached frame features (level E, zero crossings Z) of the
vad_eval test streams. Float-free and frame-level only, so anything that works here ports
straight into vad_model.py / vad.sv.

Rule under test (superset of the current VAD):
  nf  noise floor, follows E down fast and creeps up (as now)
  nm  mean level of the frames judged non-speech (new): nm += (E - nm) >> NM_SH while idle
  loud = E > max(nf * K_ON/4, nm * K_M/4)          (K_OFF, K_MOFF while active)
  odd  = E > max(nf * K_Z/4,  nm * K_MZ/4) and |Z - zn| > Z_DIFF
  nd  mean absolute deviation of those frames from nm (new); with K_D the thresholds are
      also at least nm + nd * K_D/4 (K_DOFF while active, K_DZ for the zero-crossing path)
  attack ATTACK frames; hangover = min(HANG, HANG_MULT * raw frames so far in this burst)
The nm tests only apply while nm * 4 > nf * FL (noise level fluctuating); FL = 0: always.
K_M = 0 switches the mean test off, HANG_MULT = 0 gives the fixed hangover: that is the first VAD.

  python rtl/model/vad_explore.py                       built-in list of variants
  python rtl/model/vad_explore.py "dict(K_M=8)" ...     your own
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import vad_eval as ve

BASE = dict(K_ON=8, K_OFF=6, K_Z=5, Z_DIFF=24, E_MIN=1024, ATTACK=2, HANG=20, UP_IDLE=5, UP_ACT=7, DOWN=2,
            K_M=0, K_MOFF=0, K_MZ=0, NM_SH=4, HANG_MULT=0, FL=0,
            K_D=0, K_DOFF=0, K_DZ=0, ND_SH=4)
EMAX = (1 << 24) - 1


def decide(e_arr, z_arr, **kw):
    p = dict(BASE)
    p.update(kw)
    k_moff = p["K_MOFF"] or p["K_M"]
    k_mz = p["K_MZ"] or p["K_M"]
    out = np.zeros(len(e_arr), dtype=np.int8)
    nf = nm = zn = nd = 0
    active = att = hang = run = 0
    first = True
    es, zs = e_arr.tolist(), z_arr.tolist()
    for i in range(len(es)):
        e, z = es[i], zs[i]
        if first:
            first = False
            nf = nm = e
            zn = z << 4
            continue
        # the mean test only applies while the noise is fluctuating: mean well above the floor
        fluct = (not p["FL"]) or ((nm << 2) > nf * p["FL"])
        thr = (nf * (p["K_OFF"] if active else p["K_ON"])) >> 2
        thr_z = (nf * p["K_Z"]) >> 2
        if fluct:
            thr = max(thr, (nm * (k_moff if active else p["K_M"])) >> 2)
            thr_z = max(thr_z, (nm * k_mz) >> 2)
        if p["K_D"]:
            # mean plus a multiple of the mean absolute deviation of the non-speech frames
            thr = max(thr, nm + ((nd * ((p["K_DOFF"] or p["K_D"]) if active else p["K_D"])) >> 2))
            thr_z = max(thr_z, nm + ((nd * (p["K_DZ"] or p["K_D"])) >> 2))
        zdiff = abs(z - (zn >> 4))
        big = e > p["E_MIN"]
        raw = (e > thr and big) or (e > thr_z and big and zdiff > p["Z_DIFF"])
        if e < nf:
            nf = nf - ((nf - e) >> p["DOWN"])
        else:
            nf = min(nf + (nf >> (p["UP_ACT"] if (active or raw) else p["UP_IDLE"])) + 1, EMAX)
        if not raw and not active:
            zn = zn + (((z << 4) - zn) >> 3)
            nd = nd + ((abs(e - nm) - nd) >> p["ND_SH"])
            nm = nm + ((e - nm) >> p["NM_SH"])
        att = min(att + 1, p["ATTACK"]) if raw else 0
        if raw and att >= p["ATTACK"]:
            if not active:
                run = att
            else:
                run = min(run + 1, 255)
            active = 1
            hang = min(p["HANG"], p["HANG_MULT"] * run) if p["HANG_MULT"] else p["HANG"]
        elif active:
            if raw:
                run = min(run + 1, 255)
                hang = min(p["HANG"], p["HANG_MULT"] * run) if p["HANG_MULT"] else p["HANG"]
            elif hang == 0:
                active = 0
                run = 0
            else:
                hang -= 1
        out[i] = active
    return out


def evaluate(**kw):
    c = ve.prepare()
    sp, guard = c["sp"], c["guard"]
    res = {}
    for name, (ft, ft5) in c["cond"].items():
        v = decide(*ft, **kw).astype(bool)
        n = min(len(v), len(sp))
        v, s2, g2 = v[:n], sp[:n], guard[:n]
        non = ~s2 & ~g2
        r = dict(miss=float((s2 & ~v).sum() / s2.sum()), fa=float((non & v).sum() / non.sum()))
        if ft5 is not None:
            r["noise_only"] = float(decide(*ft5, **kw).mean())
        res[name] = r
    return res


def show(kw, res):
    print(kw)
    print("  %-7s %s" % ("", "   ".join("%2d dB: miss/ FA /only" % s for s in (20, 10, 5, 0))))
    for kind in ("white", "pink", "hum", "babble"):
        print("  %-7s" % kind, "   ".join("%5.1f/%5.1f/%5.1f" % (
            100 * res["%s_%d" % (kind, s)]["miss"], 100 * res["%s_%d" % (kind, s)]["fa"],
            100 * res["%s_%d" % (kind, s)]["noise_only"]) for s in (20, 10, 5, 0)))
    print("  clean   %5.1f/%5.1f" % (100 * res["clean"]["miss"], 100 * res["clean"]["fa"]), flush=True)


if __name__ == "__main__":
    sets = [dict(), dict(K_M=8), dict(K_M=8, FL=5), dict(K_M=8, FL=6), dict(K_M=8, FL=7), dict(K_M=8, FL=8),
            dict(K_M=8, FL=6, K_MOFF=7), dict(K_M=9, FL=6), dict(K_M=8, FL=6, HANG_MULT=2)]
    if len(sys.argv) > 1:
        sets = [eval(a) for a in sys.argv[1:]]
    for kw in sets:
        show(kw, evaluate(**kw))
