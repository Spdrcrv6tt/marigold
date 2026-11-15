"""Try a syllable-rate modulation feature on top of the VAD, on the cached frame features of
vad_eval (no new simulation). Frame level E as in vad_model.py.

  m  += (E - m) >> MS                       running mean level
  dm += (|E - E_prev| - dm) >> MS           running mean of the frame-to-frame level change
  modulated = dm * K_MI > m * 4             (one person talking changes level every syllable;
                                             eight people talking at once change it much less)
MOD = 1: raw speech also needs `modulated`. MOD = 2: a frame the mean + deviation test rejected
is still speech if it clears the plain floor test and is `modulated` (rescues speech in babble). Hardware cost: two 24-bit running averages, an
absolute difference and one compare (about 75 flops), no multiplier.

  python rtl/model/vad_mod.py "dict(K_D=0, MOD=1, K_MI=8)" ...
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import vad_explore as vx

DEF = dict(MOD=0, K_MI=8, MS=3)


def decide(e_arr, z_arr, **kw):
    q = dict(DEF)
    for k in list(kw):
        if k in DEF:
            q[k] = kw.pop(k)
    p = dict(vx.BASE)
    p.update(kw)
    out = np.zeros(len(e_arr), dtype=np.int8)
    nf = nm = zn = nd = 0
    m = dm = eprev = 0
    active = att = hang = 0
    first = True
    es, zs = e_arr.tolist(), z_arr.tolist()
    for i in range(len(es)):
        e, z = es[i], zs[i]
        if first:
            first = False
            nf = nm = m = eprev = e
            zn = z << 4
            continue
        m = m + ((e - m) >> q["MS"])
        dm = dm + ((abs(e - eprev) - dm) >> q["MS"])
        eprev = e
        thr = (nf * (p["K_OFF"] if active else p["K_ON"])) >> 2
        thr_z = (nf * p["K_Z"]) >> 2
        if p["K_D"]:
            thr = max(thr, nm + ((nd * ((p["K_DOFF"] or p["K_D"]) if active else p["K_D"])) >> 2))
            thr_z = max(thr_z, nm + ((nd * (p["K_DZ"] or p["K_D"])) >> 2))
        zdiff = abs(z - (zn >> 4))
        big = e > p["E_MIN"]
        raw = (e > thr and big) or (e > thr_z and big and zdiff > p["Z_DIFF"])
        modulated = dm * q["K_MI"] > (m << 2)
        if q["MOD"] == 1 and raw:
            raw = modulated
        elif q["MOD"] == 2 and not raw and modulated and big:
            # rescue: a frame the deviation test rejected still counts if it clears the plain
            # floor test and the level is moving like one talker
            raw = e > ((nf * (p["K_OFF"] if active else p["K_ON"])) >> 2)
        if e < nf:
            nf = nf - ((nf - e) >> p["DOWN"])
        else:
            nf = min(nf + (nf >> (p["UP_ACT"] if (active or raw) else p["UP_IDLE"])) + 1, vx.EMAX)
        if not raw and not active:
            zn = zn + (((z << 4) - zn) >> 3)
            nd = nd + ((abs(e - nm) - nd) >> p["ND_SH"])
            nm = nm + ((e - nm) >> p["NM_SH"])
        att = min(att + 1, p["ATTACK"]) if raw else 0
        if raw and att >= p["ATTACK"]:
            active = 1
            hang = p["HANG"]
        elif active:
            if raw:
                hang = p["HANG"]
            elif hang == 0:
                active = 0
            else:
                hang -= 1
        out[i] = active
    return out


if __name__ == "__main__":
    vx.decide = decide
    for a in sys.argv[1:]:
        kw = eval(a)
        vx.show(kw, vx.evaluate(**dict(kw)))
