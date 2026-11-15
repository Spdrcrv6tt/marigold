"""VAD: RTL against rtl/model/vad_model.py, every frame (decision, raw flag, level, zero
crossings, noise floor, zero-crossing average, noise mean and deviation). The lockout_* tests also check
that the noise statistics never learn speech (check_no_lockout), on a loud start and on the utterance that locked the
old detector out."""
import glob
import os
import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly, Timer

from vad_model import Vad

LS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "librispeech", "LibriSpeech", "test-clean")


async def run_case(dut, pcm, gap=0):
    dut.rst_n.value = 1
    dut.in_valid.value = 0
    dut.in_data.value = 0
    cocotb.start_soon(Clock(dut.clk, 20, "ns").start())
    await Timer(7, "ns")
    dut.rst_n.value = 0          # a real falling edge: the gated clock is stopped during reset
    await Timer(100, "ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    got = []

    async def mon():
        while True:
            await RisingEdge(dut.frame_stb)
            await ReadOnly()
            got.append((int(dut.vad_o.value), int(dut.raw_o.value), int(dut.e_o.value), int(dut.z_o.value),
                        int(dut.nf_o.value), int(dut.zn_o.value), int(dut.nm_o.value), int(dut.nd_o.value)))

    m = cocotb.start_soon(mon())
    for v in pcm.tolist():
        dut.in_valid.value = 1
        dut.in_data.value = v & 0xFFFF
        await RisingEdge(dut.clk)
        if gap:
            dut.in_valid.value = 0
            for _ in range(gap):
                await RisingEdge(dut.clk)
    dut.in_valid.value = 0
    for _ in range(8):
        await RisingEdge(dut.clk)
    m.cancel()

    r = Vad().run(pcm)
    ref = list(zip(r["vad"].tolist(), r["raw"].tolist(), r["e"].tolist(), r["z"].tolist(), r["nf"].tolist(), r["zn"].tolist(),
                   r["nm"].tolist(), r["nd"].tolist()))
    assert len(got) == len(ref), "frames: rtl %d model %d" % (len(got), len(ref))
    for i, (a, b) in enumerate(zip(got, ref)):
        assert a == b, "frame %d: rtl %s model %s" % (i, a, b)
    dut._log.info("bit-exact over %d frames, %d active, %d on/off edges", len(ref), int(r["vad"].sum()),
                  int(np.abs(np.diff(r["vad"])).sum()))
    return r


def bursts(rng, n, level):
    """Speech-like bursts: voiced (low zero-crossing) and hiss segments over a noise floor."""
    x = level * rng.standard_normal(n)
    t = np.arange(n) / 16000.0
    pos = 8000
    while pos < n - 16000:
        ln = int(rng.integers(1500, 9000))
        amp = float(rng.uniform(300, 12000))
        if rng.random() < 0.6:
            seg = amp * np.sin(2 * np.pi * rng.uniform(100, 400) * t[:ln])
        else:
            seg = amp * rng.standard_normal(ln)
        x[pos:pos + ln] += seg * np.hanning(ln)
        pos += ln + int(rng.integers(500, 14000))
    return np.clip(np.round(x), -32768, 32767).astype(np.int64)


@cocotb.test()
async def synthetic_bursts(dut):
    rng = np.random.default_rng(11)
    await run_case(dut, np.concatenate((bursts(rng, 160000, 30.0), bursts(rng, 160000, 800.0))))


@cocotb.test()
async def rails_and_silence(dut):
    """Digital silence, full-scale square wave (-32768 / 32767), a level step, silence again."""
    rng = np.random.default_rng(5)
    sq = np.where((np.arange(40000) // 7) % 2 == 0, -32768, 32767)
    x = np.concatenate((np.zeros(9000), sq, np.zeros(9000), 2000 * rng.standard_normal(30000),
                        20 * rng.standard_normal(30000), np.full(5000, -32768), np.zeros(8000)))
    await run_case(dut, np.clip(np.round(x), -32768, 32767).astype(np.int64))


@cocotb.test()
async def activity_stimulus(dut):
    """Same real-speech stream with idle clocks between samples (3 idle per sample), as the
    gated clock sees it in the chip; also the stimulus for the switching-activity run."""
    await _speech(dut, gap=3)


@cocotb.test()
async def librispeech_in_noise(dut):
    """Three real utterances with gaps, pink-ish noise at about 10 dB SNR."""
    await _speech(dut)


async def _speech(dut, gap=0):
    files = sorted(glob.glob(os.path.join(LS, "*", "*", "*.flac")))
    if not files:
        dut._log.warning("LibriSpeech not found under data/, using synthetic bursts instead")
        rng = np.random.default_rng(12)
        await run_case(dut, bursts(rng, 320000, 200.0), gap)
        return
    import soundfile as sf
    rng = np.random.default_rng(3)
    parts = [np.zeros(16000)]
    for f in (files[0], files[700], files[1900]):
        parts += [sf.read(f, dtype="float64")[0], np.zeros(24000)]
    x = np.concatenate(parts)
    nz = np.cumsum(rng.standard_normal(len(x)))
    nz = nz - np.convolve(nz, np.ones(64) / 64, "same")
    x = x / np.std(x) * 0.05 + nz / np.std(nz) * 0.012
    r = await run_case(dut, np.clip(np.round(x * 32768), -32768, 32767).astype(np.int64), gap)
    assert 0.2 < r["vad"].mean() < 0.98


def check_no_lockout(dut, r):
    """The rules that keep the noise statistics free of speech, on every frame after the first (r = RTL = model):
    a frame at K_G/4 times the floor or more leaves NM, ND and ZN unchanged; a frame over K_CAP/4 times the floor
    and over E_MIN is raw speech, whatever NM and ND hold. Returns how often each rule applied."""
    p = Vad().p
    e, nf, raw = r["e"][1:], r["nf"][:-1], r["raw"][1:]
    far = e >= (nf * p["K_G"]) >> 2
    same = (r["nm"][1:] == r["nm"][:-1]) & (r["nd"][1:] == r["nd"][:-1]) & (r["zn"][1:] == r["zn"][:-1])
    assert same[far].all(), "noise statistics learned a frame far above the floor at frame %d" % (1 + np.flatnonzero(far & ~same)[0])
    over = (e > (nf * p["K_CAP"]) >> 2) & (e > p["E_MIN"])
    assert raw[over].all(), "frame %d is over the cap but not raw" % (1 + np.flatnonzero(over & (raw == 0))[0])
    dut._log.info("%d frames far above the floor (not learned), %d over the cap (raw)", int(far.sum()), int(over.sum()))
    return int(far.sum()), int(over.sum())


@cocotb.test()
async def lockout_loud_start(dut):
    """A loud first frame (it sets the floor and the noise mean), a quiet room, then speech-like bursts:
    the gate and the cap both act, bit-exact, and the rules of check_no_lockout hold."""
    rng = np.random.default_rng(21)
    x = np.concatenate((20000 * rng.standard_normal(256), 30 * rng.standard_normal(16000), bursts(rng, 96000, 30.0)))
    r = await run_case(dut, np.clip(np.round(x), -32768, 32767).astype(np.int64))
    far, over = check_no_lockout(dut, r)
    assert far > 0 and over > 0


@cocotb.test()
async def lockout_utterance(dut):
    """6930-75918-0004 from reset: its loud first frame raised the old detector's noise mean and deviation, which
    then learned the speech, and the old detector stayed off from 3.8 s to 7.5 s mid-sentence. Now it is on from
    4.16 to 6.62 s ("love for a country where they had both been")."""
    f = glob.glob(os.path.join(LS, "6930", "75918", "6930-75918-0004.flac"))
    if not f:
        dut._log.warning("LibriSpeech not found under data/, skipped")
        return
    import soundfile as sf
    pcm = sf.read(f[0], dtype="int16")[0].astype(np.int64)
    r = await run_case(dut, pcm)
    check_no_lockout(dut, r)
    a, b = int(4.16 * 16000) >> 8, int(6.62 * 16000) >> 8
    old = Vad(K_G=0, K_CAP=0).run(pcm)["vad"][a:b].mean()
    dut._log.info("on from 4.16 to 6.62 s: %.2f (without the fix %.2f)", r["vad"][a:b].mean(), old)
    assert r["vad"][a:b].mean() > 0.9 and old < 0.1
