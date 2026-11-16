"""Wake-gated capture ring (mg_capture) against the segment rule model (rtl/model/mg_model.py Segmenter): segment
bounds, delivered samples, TIMEOUT flags, WAKE/TIMEOUT events, and core power (osc_en) around each segment."""
import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ReadOnly
import mg_model as M

P = dict(pre=300, mn=1500, cap=6000, hang=3, onset=2, listen=2500)
PER = 24                                    # clocks per sample


def sample(i):
    v = (i * 7919 + 13) & 0xFFFF
    return v - 65536 if v >= 32768 else v


class Core:
    """Answers every delivered sample; after an end marker it asks to sleep (dq_t)."""

    def __init__(self, dut):
        self.dut, self.segs, self.cur, self.err = dut, [], [], []
        self.osc_seen = []

    async def run(self):
        dut = self.dut
        last = 0
        while True:
            await RisingEdge(dut.clk)
            if not int(dut.osc_en.value):        # switched off: the core's toggles are back at reset
                last = 0
                dut.ack_t.value = 0
                dut.dq_t.value = 0
                continue
            t = int(dut.q_tog.value)
            if t == last:
                continue
            last = t
            idx, data, seg = int(dut.q_idx.value), dut.q_data.value.to_signed(), int(dut.q_seg.value)
            if not int(dut.osc_en.value):
                self.err.append("delivery while the core is off at %d" % idx)
            if seg:
                if data != sample(idx):
                    self.err.append("sample %d: %d, want %d" % (idx, data, sample(idx)))
                self.cur.append(idx)
            else:
                d = int(dut.q_data.value)
                self.segs.append((self.cur[0], self.cur[-1] + 1, bool(d & 2), bool(d & 1)))
                self.cur = []
            for _ in range(2):
                await RisingEdge(dut.clk)
            dut.ack_t.value = 1 - int(dut.ack_t.value)
            if not seg:                              # done with the segment: ask to be switched off
                for _ in range(20):
                    await RisingEdge(dut.clk)
                dut.dq_t.value = 1 - int(dut.dq_t.value)


async def drive(dut, n, wakes, raw):
    seg = M.Segmenter(**P)
    nw = nt = 0
    for w in range(n):
        dut.in_valid.value = 1
        dut.in_data.value = sample(w) & 0xFFFF
        rp = raw[w // 256 - 1] if (w > 0 and w % 256 == 0) else 0
        dut.in_raw.value = rp
        dut.in_wake.value = 1 if w in wakes else 0
        seg.step(w, w in wakes, rp)
        await RisingEdge(dut.clk)
        dut.in_valid.value = 0
        dut.in_wake.value = 0
        for _ in range(PER - 1):
            await RisingEdge(dut.clk)
            nw += int(dut.ev_wake.value)
            nt += int(dut.ev_tmo.value)
    return seg, nw, nt


@cocotb.test()
async def segments_random(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    for s in ("in_valid", "in_data", "in_raw", "in_wake", "ack_t", "dq_t"):
        getattr(dut, s).value = 0
    dut.pre.value, dut.mn.value, dut.cap.value = P["pre"], P["mn"], P["cap"]
    dut.hang.value, dut.onset.value, dut.listen.value = P["hang"], P["onset"], P["listen"]
    dut.rst_n.value = 0
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    for _ in range(5):
        await RisingEdge(dut.clk)
    core = Core(dut)
    cocotb.start_soon(core.run())
    rng = random.Random(4)
    n = 60000
    nf = n // 256 + 2
    raw, f = [], 0
    while len(raw) < nf:                        # speech bursts and pauses of random length
        raw += [1] * rng.randint(1, 40) + [0] * rng.choice([1, 2, 3, 5, 12, 30, 60])
    raw = raw[:nf]
    wakes = set()
    w = 400
    while w < n - 1000:
        wakes.add(w)
        if rng.random() < 0.3:
            wakes.add(w + rng.randint(1, 900))  # a second wake, maybe inside the open segment
        if rng.random() < 0.25:                 # silence after the wake: TIMEOUT
            for fr in range(w // 256, min(nf, w // 256 + P["listen"] // 256 + 3)):
                raw[fr] = 0
        w += rng.randint(2000, 9000)
    seg, nw, nt = await drive(dut, n, wakes, raw)
    for _ in range(20000):
        await RisingEdge(dut.clk)
    want = [(a, b, t) for a, b, t in seg.segs]
    got = [(a, b, t) for a, b, t, _ in core.segs]
    assert not core.err, core.err[:5]
    assert not any(o for *_, o in core.segs), "overrun"
    assert got == want, "segments differ:\nrtl   %s\nmodel %s" % (got, want)
    mw = sum(1 for _, e in seg.events if e == M.EV_WAKE)
    mt = sum(1 for _, e in seg.events if e == M.EV_TIMEOUT)
    assert (nw, nt) == (mw, mt), ((nw, nt), (mw, mt))
    assert int(dut.osc_en.value) == int(seg.on), "core power %d with a segment %s at the end" % (
        int(dut.osc_en.value), "open" if seg.on else "closed")
    dut._log.info("%d segments (%d TIMEOUT) and %d WAKE events identical, %d samples checked", len(got),
                  sum(t for *_, t in want), nw, sum(b - a for a, b, _ in want))
