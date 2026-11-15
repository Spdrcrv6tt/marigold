"""Bit-exact reference of rtl/src/mac_lane.sv."""
from fixed import sat, rshift_round


class MacLane:
    def __init__(self, AW=8, WW=8, ACCW=24, OW=8, PIPE=0):
        self.AW, self.WW, self.ACCW, self.OW, self.PIPE = AW, WW, ACCW, OW, PIPE
        self.pend, self.pend_en = 0, 0
        self.acc = 0
        self.sat_flag = 0

    def step(self, load, en, a, w, bias):
        if self.PIPE:
            # the term accepted last clock is added now; a load drops it
            en_now, prod = self.pend_en, self.pend
            self.pend_en = int(bool(en) and not load)
            if en:
                self.pend = a * w
        else:
            en_now, prod = en, a * w
        if load:
            self.acc = bias
            self.sat_flag = 0
        elif en_now:
            s = self.acc + prod
            t = sat(s, self.ACCW)
            self.sat_flag |= int(t != s)
            self.acc = t

    def q(self, shift):
        return sat(rshift_round(self.acc, shift), self.OW)


def dot_requant(a_vec, w_vec, bias, shift, **kw):
    """A whole dot product the way the array will use a lane."""
    m = MacLane(**kw)
    m.step(1, 0, 0, 0, bias)
    for a, w in zip(a_vec, w_vec):
        m.step(0, 1, a, w, 0)
    if m.PIPE:
        m.step(0, 0, 0, 0, 0)
    return m.q(shift), m.acc, m.sat_flag
