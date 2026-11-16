"""mg_wake against the golden model (sw/wake/golden): every feature, logit and score of every frame, and every
wake sample index, must match the vectors in rtl/wake/tb/vectors/<case>/expected.txt exactly.
Negative controls (must fail to match): one corrupted input sample, and the expected trace of a different sens.
Samples arrive one per 192 clocks (16 kHz on clk_fe = 3.072 MHz)."""
import json
import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer

HERE = os.path.dirname(os.path.abspath(__file__))
VEC = os.path.join(HERE, os.environ.get("WAKE_VECDIR", "vectors"))
RES = os.path.join(HERE, "results")


async def drive(dut, x, segs, sens, vads=None):
    dut.rst_n.value = 0
    dut.en.value = 0
    dut.in_valid.value = 0
    dut.in_data.value = 0
    dut.in_vad.value = 0
    dut.sens.value = sens
    await Timer(100, "ns")
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)
    for si, (a, b) in enumerate(segs):
        dut.en.value = 1
        await ClockCycles(dut.clk, 3)
        vad = vads[si] if vads else []
        for n, v in enumerate(x[a:b]):
            f = n // 256 - 1
            dut.in_vad.value = vad[f] if 0 <= f < len(vad) else 0
            dut.in_data.value = v & 0xFFFF
            dut.in_valid.value = 1
            await RisingEdge(dut.clk)
            dut.in_valid.value = 0
            await ClockCycles(dut.clk, 191)
        await ClockCycles(dut.clk, 64000)        # let the last triggered frame finish
        dut.en.value = 0
        await ClockCycles(dut.clk, 10)


def read_trace(p):
    with open(p) as f:
        return [l.strip() for l in f if l.strip()]


async def run_case(dut, name, corrupt=False):
    d = os.path.join(VEC, name)
    c = json.load(open(os.path.join(d, "case.json")))
    x = [int(v) for v in open(os.path.join(d, "x.txt"))]
    if corrupt:
        x[len(x) // 2] ^= 0x0400
    cocotb.start_soon(Clock(dut.clk, 326, "ns").start())
    vp = os.path.join(d, "vad.json")
    await drive(dut, x, c["segs"], c["sens"], json.load(open(vp)) if os.path.exists(vp) else None)
    got = read_trace(os.environ["WAKE_TRACE"])
    exp = read_trace(os.path.join(d, "expected.txt"))
    bad = [i for i, (g, e) in enumerate(zip(got, exp)) if g != e]
    return got, exp, bad


def check(dut, name, got, exp, bad):
    nw = sum(l.startswith("W") for l in exp)
    msg = f"{name}: {len(got)} trace lines vs {len(exp)} expected, {len(bad)} mismatches, {nw} wakes expected"
    dut._log.info(msg)
    if bad:
        i = bad[0]
        dut._log.error(f"first mismatch at line {i}: got {got[i]!r}, expected {exp[i]!r}")
    assert not any("OVERRUN" in l for l in got), "frame overrun"
    assert len(got) == len(exp) and not bad, msg


def case_test(name, corrupt=False, expect_fail=False):
    async def t(dut):
        got, exp, bad = await run_case(dut, name, corrupt)
        if corrupt or expect_fail:
            assert bad or len(got) != len(exp), "negative control: the trace still matched"
            dut._log.info(f"{name} negative control: mismatch detected as required ({len(bad)} lines differ)")
        else:
            check(dut, name, got, exp, bad)
    t.__name__ = f"test_{name}{'_corrupt' if corrupt else ''}"
    return t


CASES = json.load(open(os.path.join(VEC, "cases.json")))
CASE = os.environ.get("WAKE_CASE", CASES[0])
test_main = cocotb.test()(case_test(CASE, os.environ.get("WAKE_CORRUPT", "0") == "1", os.environ.get("WAKE_EXPECT_FAIL", "0") == "1"))
