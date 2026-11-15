"""UART TX: a receiver model samples each bit mid-cell and checks framing (start, 8 data bits
LSB first, stop), the exact bit time, back-to-back bytes and idle gaps."""
import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer

CLK_NS = 20


async def setup(dut, div):
    dut.rst_n.value = 0
    dut.div.value = div
    dut.in_data.value = 0
    dut.in_valid.value = 0
    cocotb.start_soon(Clock(dut.clk, CLK_NS, "ns").start())
    await Timer(50, "ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def receiver(dut, div, out, frame_times):
    bit_ns = (div + 1) * CLK_NS
    while True:
        await FallingEdge(dut.tx_o)                       # start bit
        t0 = cocotb.utils.get_sim_time("ns")
        frame_times.append(t0)
        await Timer(bit_ns // 2, "ns")
        assert dut.tx_o.value == 0, "start bit not held"
        v = 0
        for i in range(8):
            await Timer(bit_ns, "ns")
            v |= int(dut.tx_o.value) << i
        await Timer(bit_ns, "ns")
        assert dut.tx_o.value == 1, "stop bit missing"
        out.append(v)


async def send(dut, data, gap_p=0.0, rng=None):
    for b in data:
        while True:
            await FallingEdge(dut.clk)
            if dut.in_ready.value == 1 and not (rng and rng.random() < gap_p):
                break
        dut.in_data.value = b
        dut.in_valid.value = 1
        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)
        dut.in_valid.value = 0
        assert dut.in_ready.value == 0


async def run(dut, div, data, gap_p=0.0):
    await setup(dut, div)
    out, times = [], []
    rx = cocotb.start_soon(receiver(dut, div, out, times))
    rng = random.Random(div)
    await send(dut, data, gap_p, rng)
    await Timer((div + 1) * CLK_NS * 12, "ns")
    rx.cancel()
    assert out == list(data), "got %s" % out[:16]
    assert dut.tx_o.value == 1 and dut.in_ready.value == 1
    return times


@cocotb.test()
async def text_115200(dut):
    """50 MHz / 115200 baud (div 433): a line of text back to back; frames are exactly 10 bits apart."""
    msg = b"floor 4, not 3. cancel that.\r\n" + bytes(range(256))
    t = await run(dut, 433, msg)
    gaps = {round(b - a) for a, b in zip(t, t[1:])}
    # back to back: next start bit follows after 10 bit times, plus the one clock the handshake takes
    assert gaps <= {10 * 434 * CLK_NS, 10 * 434 * CLK_NS + CLK_NS}, gaps
    dut._log.info("frame spacing %s ns for a 10-bit frame of %d ns", sorted(gaps), 10 * 434 * CLK_NS)


@cocotb.test()
async def fast_and_gappy(dut):
    """Small dividers (0, 1, 7) and random idle gaps between bytes."""
    rng = random.Random(8)
    for div in (7, 1, 0):
        data = bytes(rng.randrange(256) for _ in range(300)) + b"\x00\xff\x55\xaa"
        if div == 0:
            # one clock per bit: check by clock-exact sampling instead of the mid-cell receiver
            await setup(dut, 0)
            for b in data[:50]:
                await FallingEdge(dut.clk)
                dut.in_data.value = b
                dut.in_valid.value = 1
                await RisingEdge(dut.clk)
                bits = []
                for _ in range(10):
                    await FallingEdge(dut.clk)
                    dut.in_valid.value = 0
                    bits.append(int(dut.tx_o.value))
                assert bits == [0] + [(b >> i) & 1 for i in range(8)] + [1], (b, bits)
                await RisingEdge(dut.clk)
        else:
            await run(dut, div, data, gap_p=0.3)
