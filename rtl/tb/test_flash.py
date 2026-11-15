"""SPI NOR read controller against the behavioural flash in flash_model.sv.

Checks the byte stream (order, values, no loss or duplication under random backpressure),
seeks (stop, start elsewhere), every read mode the build supports, round-trip delays that
need rx_dly, and measures the sustained rate in bytes per clk cycle with the consumer
always ready.
"""
import json
import os
import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer

CLK_PS = 20000        # 50 MHz
MODES = {0: ("single 0x03", 0), 1: ("single fast 0x0B", 8), 2: ("quad out 0x6B", 8),
         3: ("quad I/O 0xEB", 4), 4: ("octal out 0x8B", 8), 5: ("octal I/O 0xCB", 16)}
OPC24 = {0: 0x03, 1: 0x0B, 2: 0x6B, 3: 0xEB, 4: 0x8B, 5: 0xCB}
OPC32 = {0: 0x13, 1: 0x0C, 2: 0x6C, 3: 0xEC, 4: 0x7C, 5: 0xCC}


def pat(a):
    return (((a * 0x9E3779B1) & 0xFFFFFFFF) >> 24) ^ (a & 0xFF)


def modes_for(dut):
    iow = int(dut.IOW.value)
    return [m for m in MODES if (m < 4 or iow == 8)]


async def reset(dut, t_out=3000, t_in=2000, t_clqv=7000, rx_dly=0):
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.stop.value = 0
    dut.addr.value = 0
    dut.mode.value = 0
    dut.dummy.value = 0
    dut.rx_dly.value = rx_dly
    dut.out_ready.value = 0
    dut.t_out.value = t_out
    dut.t_in.value = t_in
    dut.t_clqv.value = t_clqv
    cocotb.start_soon(Clock(dut.clk, CLK_PS, "ps").start())
    await Timer(5 * CLK_PS, "ps")
    dut.rst_n.value = 1
    for _ in range(3):
        await RisingEdge(dut.clk)


async def begin(dut, addr, mode):
    dut.addr.value = addr
    dut.mode.value = mode
    dut.dummy.value = MODES[mode][1]
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0


async def take(dut, n, p_ready=1.0, rng=None):
    """Collect n bytes; returns (bytes, clk cycles from first to last byte)."""
    got, first, cyc = [], None, 0
    while len(got) < n:
        await RisingEdge(dut.clk)
        cyc += 1
        rdy = 1 if (p_ready >= 1.0 or rng.random() < p_ready) else 0
        dut.out_ready.value = rdy
        await FallingEdge(dut.clk)          # mid-cycle: what the next rising edge will see
        if rdy and dut.out_valid.value == 1:
            got.append(int(dut.out_data.value))
            if first is None:
                first = cyc
    await RisingEdge(dut.clk)
    dut.out_ready.value = 0
    return got, cyc - first


async def end(dut):
    dut.stop.value = 1
    await RisingEdge(dut.clk)
    dut.stop.value = 0
    for _ in range(200):
        await RisingEdge(dut.clk)
        if dut.busy.value == 0:
            return
    assert False, "controller did not go idle after stop"


def check(dut, got, addr, what):
    exp = [pat(addr + i) for i in range(len(got))]
    bad = [i for i in range(len(got)) if got[i] != exp[i]]
    assert not bad, "%s: byte %d is %02x, expected %02x (%d of %d wrong)" % (
        what, bad[0], got[bad[0]], exp[bad[0]], len(bad), len(got))
    assert dut.bad_cmd.value == 0, what + ": flash saw an unknown or unsupported opcode"
    assert dut.contention.value == 0, what + ": both ends drove an IO line"


@cocotb.test()
async def all_modes_stream(dut):
    """Each mode: 600 bytes with the consumer always ready; report bytes per clk cycle."""
    await reset(dut)
    aw = int(dut.AW.value)
    rates = {}
    for m in modes_for(dut):
        addr = 0x012345 + 0x1000 * m
        await begin(dut, addr, m)
        got, cyc = await take(dut, 600)
        check(dut, got, addr, MODES[m][0])
        assert int(dut.last_cmd.value) == (OPC32 if aw == 32 else OPC24)[m]
        await end(dut)
        rate = (len(got) - 1) / cyc
        rates[MODES[m][0]] = rate
        dut._log.info("%-18s %.4f bytes/clk -> %.3f MB/s at 50 MHz clk (SCK = clk/%d)", MODES[m][0], rate,
                      rate * 50.0, 2 * int(dut.HALF.value))
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "flash_rate_%s.json" % os.environ.get("VARIANT", "x")), "w") as f:
        json.dump(dict(half=int(dut.HALF.value), bytes_per_clk=rates), f, indent=1)


@cocotb.test()
async def backpressure(dut):
    """Random stalls (ready 30% then 85% of the time): nothing lost, repeated or reordered."""
    await reset(dut)
    rng = random.Random(4)
    for m in modes_for(dut):
        for p in (0.3, 0.85):
            addr = rng.randrange(0, 1 << 24)
            await begin(dut, addr, m)
            got, _ = await take(dut, 400, p, rng)
            # long stall, then carry on
            for _ in range(rng.randrange(20, 200)):
                await RisingEdge(dut.clk)
            more, _ = await take(dut, 200, 1.0)
            check(dut, got + more, addr, "%s ready=%.2f" % (MODES[m][0], p))
            await end(dut)


@cocotb.test()
async def seek(dut):
    """Stop mid-stream (also while stalled and during the command) and restart elsewhere."""
    await reset(dut)
    rng = random.Random(9)
    sel0 = int(dut.n_selects.value)
    n = 0
    for m in modes_for(dut):
        for k in range(6):
            addr = rng.randrange(0, (1 << 24) - 4096)
            await begin(dut, addr, m)
            n += 1
            if k == 0:
                for _ in range(rng.randrange(1, 12)):     # stop during command / address
                    await RisingEdge(dut.clk)
            elif k == 1:
                for _ in range(300):                      # stop while stalled with a full buffer
                    await RisingEdge(dut.clk)
            else:
                got, _ = await take(dut, rng.randrange(1, 40), 0.7, rng)
                check(dut, got, addr, "seek %s" % MODES[m][0])
            await end(dut)
            assert dut.out_valid.value == 0
    assert int(dut.n_selects.value) - sel0 == n


@cocotb.test()
async def round_trip_delay(dut):
    """Pad + board + flash delay sweep: find which rx_dly settings read correctly."""
    m = 4 if int(dut.IOW.value) == 8 else 3
    half = int(dut.HALF.value)
    table = {}
    for total_ns in (6, 12, 18, 24, 30, 36):
        ok = []
        for dly in range(4):
            t = total_ns * 1000
            await reset(dut, t_out=t // 4, t_in=t // 4, t_clqv=t - 2 * (t // 4), rx_dly=dly)
            await begin(dut, 0x00ABCD, m)
            got, _ = await take(dut, 64)
            await end(dut)
            if got == [pat(0x00ABCD + i) for i in range(64)] and dut.contention.value == 0:
                ok.append(dly)
        table[total_ns] = ok
        dut._log.info("round trip %2d ns (SCK out + flash tCLQV + data in), SCK %.1f MHz: rx_dly that work: %s",
                      total_ns, 1e6 / (2 * half * CLK_PS), ok)
        assert ok, "no rx_dly setting works at %d ns" % total_ns
    assert 0 in table[12], "default sampling must work at 12 ns round trip"
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                           "flash_rxdly_%s.json" % os.environ.get("VARIANT", "x")), "w") as f:
        json.dump(table, f)
