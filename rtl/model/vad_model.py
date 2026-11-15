"""Bit-exact integer reference of the voice activity detector (rtl/src/vad.sv).

Per frame of FRAME = 2^FRAME_LOG2 samples (16 ms at 16 kHz):
  E  = sum |b|                      level of the band-passed signal b (two one-pole filters,
                                    roughly 320 Hz to 1.8 kHz; no multiplier)
  Z  = number of sign changes       of the high-passed signal (saturates at 255)
  NF = noise floor: follows E down fast, creeps up slowly (slower still while speech is active)
  ZN = zero-crossing count of the noise, averaged over non-speech frames (4 fraction bits)
  NM = mean of E over the frames judged non-speech, ND = their mean absolute deviation from NM
A frame is "raw speech" when E clears the floor by a factor (K_ON/4 when idle, K_OFF/4 when
active), or clears it by the smaller K_Z/4 while its zero-crossing count is unlike the noise
(voiced speech in hiss has far fewer crossings). In both cases E must also exceed
NM + ND * K_D/4 (K_DOFF/4 while active): in steady noise ND is small and this changes
little, in fluctuating noise such as a crowd it lifts the threshold above the noise's own
swings, which is what stops the detector sitting on in babble. ATTACK raw frames in a row switch the output
on; it stays on for HANG frames after the last raw frame.
The noise statistics never learn speech (Poinsetta, Nov 14, 2026): NM, ND and ZN learn only from
frames that are neither raw nor active and whose E is below NF * K_G/4 (4 times the floor; noise
alone stays under about 2.6 times it, speech goes far above). And a frame above NF * K_CAP/4
(12 times the floor) and E_MIN is raw speech whatever NM and ND hold (the same as capping
NM + ND * K_D/4 there), so the detector cannot lock itself out. Without these (K_G = K_CAP = 0,
the VAD before) a loud first frame or speech frames under the threshold could raise NM + 4 ND
above the speech, and every further speech frame then raised it more.
"""
import numpy as np

PARAMS = dict(FRAME_LOG2=8, K_ON=8, K_OFF=6, K_Z=5, Z_DIFF=24, E_MIN=1024, ATTACK=2, HANG=20,
              UP_IDLE=5, UP_ACT=7, DOWN=2, PREEMPH=1, HP_SH=3, LP_SH=1,
              K_D=16, K_DOFF=12, NM_SH=4, ND_SH=4, K_G=16, K_CAP=48)
EW = 24


class Vad:
    def __init__(self, **kw):
        self.p = dict(PARAMS)
        self.p.update(kw)
        self.reset()

    def reset(self):
        self.nf = 0
        self.nm = 0
        self.nd = 0
        self.zn = 0
        self.init = True
        self.active = 0
        self.att = 0
        self.hang = 0
        self.prev_neg = False
        self.s1 = 0
        self.s2 = 0

    def frame(self, e, z):
        """Advance one frame. Returns (vad, raw, nf, zn, nm, nd) after the frame."""
        p = self.p
        emax = (1 << EW) - 1
        if self.init:
            self.init = False
            self.nf = e
            self.nm = e
            self.nd = 0
            self.zn = z << 4
            return self.active, 0, self.nf, self.zn, self.nm, self.nd
        nf, zn, nm, nd = self.nf, self.zn, self.nm, self.nd
        k = p["K_OFF"] if self.active else p["K_ON"]
        thr = (nf * k) >> 2
        thr_z = (nf * p["K_Z"]) >> 2
        if p["K_D"]:
            thr = max(thr, nm + ((nd * (p["K_DOFF"] if self.active else p["K_D"])) >> 2))
            thr_z = max(thr_z, nm + ((nd * p["K_D"]) >> 2))
        zdiff = abs(z - (zn >> 4))
        loud = e > thr and e > p["E_MIN"]
        odd = e > thr_z and e > p["E_MIN"] and zdiff > p["Z_DIFF"]
        # far above the floor is speech whatever NM and ND say (the same as capping NM + ND * K_D/4 at the floor
        # times K_CAP/4, since K_CAP is above K_ON, K_OFF and K_Z)
        huge = p["K_CAP"] and e > ((nf * p["K_CAP"]) >> 2) and e > p["E_MIN"]
        raw = 1 if (loud or odd or huge) else 0
        # noise floor
        if e < nf:
            nf = nf - ((nf - e) >> p["DOWN"])
        else:
            nf = min(nf + (nf >> (p["UP_ACT"] if (self.active or raw) else p["UP_IDLE"])) + 1, emax)
        # noise statistics: only frames that are neither speech nor far above the floor (before its update)
        if not raw and not self.active and (not p["K_G"] or e < ((self.nf * p["K_G"]) >> 2)):
            zn = zn + (((z << 4) - zn) >> 3)
            nd = nd + ((abs(e - nm) - nd) >> p["ND_SH"])
            nm = nm + ((e - nm) >> p["NM_SH"])
        # attack / hangover
        if raw:
            self.att = min(self.att + 1, p["ATTACK"])
        else:
            self.att = 0
        if raw and self.att >= p["ATTACK"]:
            self.active = 1
            self.hang = p["HANG"]
        elif self.active:
            if raw:
                self.hang = p["HANG"]
            elif self.hang == 0:
                self.active = 0
            else:
                self.hang -= 1
        self.nf, self.zn, self.nm, self.nd = nf, zn, nm, nd
        return self.active, raw, nf, zn, nm, nd

    def features(self, x):
        """Frame features for a whole int16 array (drops a partial last frame).

        Per sample (PREEMPH=1, the default; 0 feeds x straight through):
          s1 += x - (s1 >> HP_SH);  hp = x - (s1 >> HP_SH)      one-pole high-pass, ~320 Hz
          s2 += hp - (s2 >> LP_SH); b  = s2 >> LP_SH            one-pole low-pass, ~1.8 kHz
          E += |b| (saturating at 24 bits),  Z += sign change of hp
        """
        n = 1 << self.p["FRAME_LOG2"]
        nfr = len(x) // n
        xs = np.asarray(x[:nfr * n], dtype=np.int64).tolist()
        hs, ls, pre = self.p["HP_SH"], self.p["LP_SH"], self.p["PREEMPH"]
        s1, s2, pn = self.s1, self.s2, self.prev_neg
        e_out, z_out = [], []
        emax = (1 << EW) - 1
        i = 0
        for _ in range(nfr):
            e = 0
            z = 0
            for v in xs[i:i + n]:
                if pre:
                    s1 += v - (s1 >> hs)
                    hp = v - (s1 >> hs)
                    s2 += hp - (s2 >> ls)
                    b = s2 >> ls
                else:
                    hp = b = v
                ng = hp < 0
                if ng != pn:
                    z += 1
                    pn = ng
                e += -b if b < 0 else b
            i += n
            e_out.append(e if e < emax else emax)
            z_out.append(z if z < 255 else 255)
        self.s1, self.s2, self.prev_neg = s1, s2, pn
        return np.array(e_out, dtype=np.int64), np.array(z_out, dtype=np.int64)

    def decide(self, e, z):
        out = [self.frame(int(a), int(b)) for a, b in zip(e, z)]
        o = np.array(out, dtype=np.int64).reshape(-1, 6)
        return dict(vad=o[:, 0], raw=o[:, 1], e=e, z=z, nf=o[:, 2], zn=o[:, 3], nm=o[:, 4], nd=o[:, 5])

    def run(self, x):
        """Returns per-frame arrays: vad, raw, e, z, nf, zn."""
        e, z = self.features(x)
        return self.decide(e, z)
