"""latch_ram (latch and flip-flop builds): random writes and reads against a dict, including
a read of a word in the cycle right after it was written, and back-to-back writes."""
import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge


@cocotb.test()
async def random_rw(dut):
    words, width = int(dut.WORDS.value), int(dut.WIDTH.value)
    rng = random.Random(3)
    for s in (dut.we, dut.waddr, dut.wdata, dut.re, dut.raddr):
        s.value = 0
    cocotb.start_soon(Clock(dut.clk, 20, "ns").start())
    for _ in range(3):
        await FallingEdge(dut.clk)
    # fill every word first (no reset in the RAM)
    ref = {}
    for a in range(words):
        v = rng.getrandbits(width)
        dut.we.value, dut.waddr.value, dut.wdata.value = 1, a, v
        ref[a] = v
        await FallingEdge(dut.clk)
    dut.we.value = 0
    await FallingEdge(dut.clk)

    last_written = None     # address written in the previous cycle
    checks = 0
    for _ in range(20000):
        # called just after a falling edge: set the inputs the next rising edge will sample
        do_w = rng.random() < 0.4
        do_r = rng.random() < 0.6
        wa, wv = rng.randrange(words), rng.getrandbits(width)
        if rng.random() < 0.3 and last_written is not None:
            ra = last_written            # read the word written one cycle ago
        else:
            ra = rng.randrange(words)
        if do_w and do_r and ra == wa:
            do_r = False                 # same-edge read of the word being written is not defined
        dut.we.value, dut.waddr.value, dut.wdata.value = int(do_w), wa, wv
        dut.re.value, dut.raddr.value = int(do_r), ra
        exp = ref[ra]
        await RisingEdge(dut.clk)
        if do_w:
            ref[wa] = wv
        last_written = wa if do_w else None
        await FallingEdge(dut.clk)
        if do_r:
            assert int(dut.rdata.value) == exp, "rdata %x expected %x" % (int(dut.rdata.value), exp)
            checks += 1
    dut._log.info("%d reads checked, %d words x %d bits", checks, words, width)
