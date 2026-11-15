"""PDM front end: RTL against rtl/model/pdm_model.py, sample for sample.

A mic model drives one PDM bit per rising edge of pdm_clk_o. Stimulus comes from the
sigma-delta modulator model (tones, speech-like noise, DC offset) plus raw bit patterns
(all ones, all zeros, random) that exercise saturation.
"""
import os
import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly, Timer, First

import pdm_model as pm

FS_PDM = 3072000


def tone_bits(freq, amp, n_bits, dc=0.0):
    t = np.arange(n_bits) / FS_PDM
    return pm.sdm(amp * np.sin(2 * np.pi * freq * t) + dc)


def noise_bits(n_bits, amp, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_bits // 192 + 8)
    u = pm.upsample_to_pdm(x)[:n_bits]
    u = amp * u / np.max(np.abs(u))
    return pm.sdm(u)


async def run_case(dut, bits, gain=0, dc_en=1):
    dut.rst_n.value = 1
    dut.en.value = 0
    dut.gain.value = gain
    dut.dc_en.value = dc_en
    dut.pdm_dat_i.value = 0
    cocotb.start_soon(Clock(dut.clk, 20, "ns").start())
    await Timer(7, "ns")
    dut.rst_n.value = 0          # a real falling edge: the gated clocks are stopped during reset
    await Timer(100, "ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    dut.en.value = 1

    got = []
    done = cocotb.triggers.Event()

    async def mic():
        for b in bits:
            await RisingEdge(dut.pdm_clk_o)
            dut.pdm_dat_i.value = int(b)
        # one more PDM period so the last bit is taken, stop the PDM clock (with DIV = 1 every
        # further clock would be another bit), then let the filters drain
        await RisingEdge(dut.pdm_clk_o)
        dut.en.value = 0
        for _ in range(400):
            await RisingEdge(dut.clk)
        done.set()

    async def mon():
        while True:
            await RisingEdge(dut.pcm_valid)
            await ReadOnly()
            got.append(dut.pcm_data.value.to_signed())

    m = cocotb.start_soon(mon())
    cocotb.start_soon(mic())
    await done.wait()
    m.cancel()
    assert int(dut.overrun.value) == 0, "FIR overrun"

    ref = pm.pdm_decimate(bits, gain=gain, dc_en=bool(dc_en))
    assert len(got) == len(ref), "sample count: rtl %d, model %d" % (len(got), len(ref))
    bad = [i for i in range(len(ref)) if got[i] != ref[i]]
    assert not bad, "first mismatch at sample %d: rtl %d model %d (%d of %d differ)" % (
        bad[0], got[bad[0]], ref[bad[0]], len(bad), len(ref))
    dut._log.info("bit-exact over %d PCM samples (%d PDM bits), peak %d", len(ref), len(bits),
                  max(abs(v) for v in ref))
    return got


LONG = os.environ.get("VARIANT", "div4") in ("div1", "div4", "cov")
NB = 192 * (1600 if LONG else 320)


@cocotb.test()
async def tone_1k(dut):
    """1 kHz at -6 dBFS with a DC offset, DC blocker on."""
    await run_case(dut, tone_bits(1000.0, 0.5, NB, dc=0.05))


@cocotb.test()
async def tone_1k_gain(dut):
    """Quiet 1 kHz tone (-46 dBFS) with gain 2^5, DC blocker off."""
    await run_case(dut, tone_bits(1000.0, 0.005, NB), gain=5, dc_en=0)


@cocotb.test()
async def noise(dut):
    """Band-limited noise at full modulator range."""
    await run_case(dut, noise_bits(NB, 0.7, seed=3))


@cocotb.test()
async def activity_stimulus(dut):
    """Conversation-level input (noise at about -30 dBFS rms with a small DC offset). Also the
    stimulus for the switching-activity run behind the power estimate."""
    await run_case(dut, noise_bits(192 * 800, 0.12, seed=21) if LONG else noise_bits(NB, 0.12, seed=21))


@cocotb.test()
async def rails_and_random(dut):
    """All ones, all zeros, then random bits: saturation paths and CIC wrap-around."""
    n = NB // 4
    rng = np.random.default_rng(7)
    bits = np.concatenate((np.ones(n, np.uint8), np.zeros(n, np.uint8),
                           rng.integers(0, 2, n).astype(np.uint8), np.ones(n, np.uint8)))
    await run_case(dut, bits, gain=7)
