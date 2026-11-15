"""MAC lane against rtl/model/mac_model.py: accumulator, saturation flag and requantised
output every clock, for int8 (WW=8) and int4 (WW=4) weights."""
import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer

from mac_model import MacLane


async def setup(dut):
    dut.rst_n.value = 0
    for s in (dut.load, dut.en, dut.a, dut.w, dut.bias, dut.shift):
        s.value = 0
    cocotb.start_soon(Clock(dut.clk, 20, "ns").start())
    await Timer(50, "ns")
    dut.rst_n.value = 1
    await FallingEdge(dut.clk)
    p = dict(AW=int(dut.AW.value), WW=int(dut.WW.value), ACCW=int(dut.ACCW.value), OW=int(dut.OW.value), PIPE=int(dut.PIPE.value))
    return p, MacLane(**p)


async def drive(dut, m, p, load, en, a, w, bias, shift):
    """Apply one clock of inputs, then compare state and q (q is checked at every shift given)."""
    # called just after a falling edge (setup and the previous call both end on one)
    dut.load.value = load
    dut.en.value = en
    dut.a.value = a & ((1 << p["AW"]) - 1)
    dut.w.value = w & ((1 << p["WW"]) - 1)
    dut.bias.value = bias & ((1 << p["ACCW"]) - 1)
    dut.shift.value = shift
    await RisingEdge(dut.clk)
    m.step(load, en, a, w, bias)
    await FallingEdge(dut.clk)
    assert dut.acc.value.to_signed() == m.acc, "acc rtl %d model %d" % (dut.acc.value.to_signed(), m.acc)
    assert int(dut.sat_flag.value) == m.sat_flag
    assert dut.q.value.to_signed() == m.q(shift), "q rtl %d model %d (acc %d shift %d)" % (
        dut.q.value.to_signed(), m.q(shift), m.acc, shift)


def rnd_s(rng, bits, extreme=0.2):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if rng.random() < extreme:
        return rng.choice((lo, hi, 0, -1, 1))
    return rng.randint(lo, hi)


@cocotb.test()
async def random_ops(dut):
    """20,000 random clocks of load / accumulate / idle with random shifts."""
    p, m = await setup(dut)
    rng = random.Random(1)
    for _ in range(20000):
        r = rng.random()
        await drive(dut, m, p, int(r < 0.03), int(r < 0.9), rnd_s(rng, p["AW"]), rnd_s(rng, p["WW"]),
                    rnd_s(rng, p["ACCW"], 0.05), rng.randint(0, 23))


@cocotb.test()
async def saturation(dut):
    """Drive the accumulator into both rails and check it sticks, flags, and recovers on load."""
    p, m = await setup(dut)
    amin, wmin = -(1 << (p["AW"] - 1)), -(1 << (p["WW"] - 1))
    top = (1 << (p["ACCW"] - 1)) - 1
    await drive(dut, m, p, 1, 0, 0, 0, top - 5, 0)
    for _ in range(4):
        await drive(dut, m, p, 0, 1, amin, wmin, 0, 0)        # (+) product
    await drive(dut, m, p, 0, 0, 0, 0, 0, 0)                  # let a pipelined term land
    assert m.acc == top and m.sat_flag == 1
    await drive(dut, m, p, 1, 0, 0, 0, -top + 3, 4)
    assert m.sat_flag == 0
    for _ in range(4):
        await drive(dut, m, p, 0, 1, amin, -wmin - 1, 0, 4)   # (-) product
    await drive(dut, m, p, 0, 0, 0, 0, 0, 4)
    assert m.acc == -top - 1 and m.sat_flag == 1
    for sh in range(0, 24):                                   # requantise both rails at every shift
        await drive(dut, m, p, 0, 0, 0, 0, 0, sh)
    await drive(dut, m, p, 1, 0, 0, 0, top, 0)
    for sh in range(0, 24):
        await drive(dut, m, p, 0, 0, 0, 0, 0, sh)


@cocotb.test()
async def dot_products(dut):
    """300 dot products of 1 to 400 terms with a bias, as the array will run them."""
    p, m = await setup(dut)
    rng = random.Random(2)
    for _ in range(300):
        n = rng.randint(1, 400)
        sh = rng.randint(0, 12)
        await drive(dut, m, p, 1, 0, 0, 0, rng.randint(-4000, 4000), sh)
        for _ in range(n):
            await drive(dut, m, p, 0, 1, rnd_s(rng, p["AW"], 0.02), rnd_s(rng, p["WW"], 0.02), 0, sh)
        await drive(dut, m, p, 0, 0, 0, 0, 0, sh)
