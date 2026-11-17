import os
"""Config loader (CRC and checks) and matcher against rtl/model/mg_model.py, with a higher-priority reader
stealing config-SRAM cycles at random."""
import random
import struct
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly, Timer
import mg_model as M
from mg_util import rand_cfg, tune_thresholds


async def setup(dut):
    cocotb.start_soon(Clock(dut.clk, 326, unit="ns").start())
    for s in ("begin_i", "commit_i", "in_valid", "in_data", "noise_req", "go", "q_we", "q_waddr", "q_wdata",
              "v_we", "v_waddr", "v_wdata"):
        getattr(dut, s).value = 0
    dut.rst_n.value = 0
    await Timer(1, unit="us")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def pulse(dut, sig):
    getattr(dut, sig).value = 1
    await RisingEdge(dut.clk)
    getattr(dut, sig).value = 0


async def load(dut, blob):
    await pulse(dut, "begin_i")
    for x in blob:
        dut.in_valid.value = 1
        dut.in_data.value = x
        await RisingEdge(dut.clk)
    dut.in_valid.value = 0
    await pulse(dut, "commit_i")
    await RisingEdge(dut.clk)
    return int(dut.valid.value), int(dut.err.value)


async def noise(dut, rng, p):
    while True:
        dut.noise_req.value = 1 if rng.random() < p else 0
        await RisingEdge(dut.clk)


@cocotb.test()
async def loader_checks(dut):
    """Good blobs load; each corruption gives the model's error code."""
    await setup(dut)
    rng = random.Random(5)
    n = 0
    for t in range(40):
        cfg = rand_cfg(rng)
        blob = M.build_blob(cfg)
        cases = [("good", blob)]
        b = bytearray(blob); b[1] ^= 0x10; cases.append(("magic", bytes(b)))
        b = bytearray(blob); b[4] = 1; cases.append(("version", bytes(b)))
        b = bytearray(blob); b[10] = 65; cases.append(("n_actions", bytes(b)))
        b = bytearray(blob); b[11] = 9; cases.append(("n_slots", bytes(b)))
        b = bytearray(blob); b[6:8] = struct.pack("<H", 257); cases.append(("D", bytes(b)))
        cases.append(("short", blob[:-1]))
        cases.append(("long", blob + b"\x00"))
        cases.append(("header only", blob[:20]))
        b = bytearray(blob); i = rng.randrange(len(b)); b[i] ^= 1 << rng.randrange(8); cases.append(("bit %d" % i, bytes(b)))
        b = bytearray(blob); b[-1] ^= 0x80; cases.append(("crc", bytes(b)))
        for name, bb in cases:
            v, e = await load(dut, bb)
            ee, _ = M.parse_blob(bb)
            assert e == ee and v == (ee == 0), "%s: rtl err %d valid %d, model err %d" % (name, e, v, ee)
            assert int(dut.cnt.value) == len(bb)
            n += 1
    dut._log.info("%d loads checked", n)


@cocotb.test()
async def match_random(dut):
    """Random configs, q and values: result registers equal the model (accept / every reject path / clamping)."""
    await setup(dut)
    rng = random.Random(11)
    cocotb.start_soon(noise(dut, random.Random(99), 0.3))
    stats = dict(accept=0, reject=0, s2none=0)
    for t in range(150):
        cfg = rand_cfg(rng, D=rng.choice([1, 3, 4, 8, 13, 32, 64]) if t % 10 else 256)
        q = [rng.randint(-128, 127) for _ in range(cfg["D"])]
        if t % 7 == 0:          # all vectors one action
            a = cfg["vectors"][0][0] or cfg["actions"][0][0]
            cfg["vectors"] = [(a, f, bb, v) for (_, f, bb, v) in cfg["vectors"]]
        tune_thresholds(rng, cfg, q)
        vals = [rng.choice([rng.randint(-3000, 3000), -rng.randint(1, 5), -rng.randint(1, 6)]) for _ in range(8)]
        blob = M.build_blob(cfg)
        v, e = await load(dut, blob)
        assert v == 1 and e == 0
        _, pc = M.parse_blob(blob)
        want = M.match(pc, q, vals)
        qq = q + [0] * (256 - len(q))
        for w in range(64):
            dut.q_we.value = 1; dut.q_waddr.value = w
            dut.q_wdata.value = int.from_bytes(struct.pack("<4b", *qq[4 * w:4 * w + 4]), "little")
            await RisingEdge(dut.clk)
        dut.q_we.value = 0
        for k in range(8):
            dut.v_we.value = 1; dut.v_waddr.value = k; dut.v_wdata.value = vals[k] & 0xFFFFFFFF
            await RisingEdge(dut.clk)
        dut.v_we.value = 0
        await pulse(dut, "go")
        lim = 3 * len(cfg["vectors"]) * (cfg["D"] + 1) + 2000
        for _ in range(lim):
            await RisingEdge(dut.clk)
            if int(dut.done.value):
                break
        else:
            raise AssertionError("matcher did not finish (trial %d)" % t)
        await ReadOnly()
        acc = int(dut.accept.value)
        got = dict(b=int(dut.best.value), sb=dut.sb.value.to_signed(), conf=int(dut.conf.value),
                   has=int(dut.has_s2.value), action=int(dut.action.value), mask=int(dut.slot_mask.value))
        if acc != (want["ev"] == M.EV_COMMAND):
            dut._log.error("cfg D %d N %d thr %d margin %d actions %s vt %s", pc["D"], pc["N"], pc["reject_thr"],
                           pc["margin"], pc["actions"], [x[0] for x in pc["vt"]])
        assert acc == (want["ev"] == M.EV_COMMAND), "trial %d accept %d rtl %s s2 %d, model %s" % (
            t, acc, got, dut.s2.value.to_signed(), want)
        assert got["b"] == want["b"] and got["sb"] == want["sb"] and got["conf"] == want["conf"], (t, got, want)
        assert got["has"] == (want["s2"] is not None), (t, got, want)
        if want["s2"] is not None:
            assert dut.s2.value.to_signed() == want["s2"], (t, dut.s2.value.to_signed(), want)
        else:
            stats["s2none"] += 1
        if acc:
            stats["accept"] += 1
            assert got["action"] == want["action"] and got["mask"] == want["slot_mask"], (t, got, want)
            vv = int(dut.vals.value)
            for k in range(8):
                x = (vv >> (32 * k)) & 0xFFFFFFFF
                x = x - (1 << 32) if x >> 31 else x
                assert x == want["vals8"][k], "trial %d slot %d: rtl %d model %d" % (t, k, x, want["vals8"][k])
        else:
            stats["reject"] += 1
        await RisingEdge(dut.clk)
    dut._log.info("150 matches identical: %s", stats)


@cocotb.test()
async def model_d_check(dut):
    """Review R4: with MODEL_D = 128 (variant d128) a blob whose D is not the model's is refused (error 6); without
    it (MODEL_D = 0, unit tests of other D) any D in 1..256 loads."""
    md = int(os.environ.get("MODEL_D", "0"))
    await setup(dut)
    rng = random.Random(21)
    for D in (128, 256, 64, 1):
        cfg = rand_cfg(rng, D=D, N=3, n_gpio=2)
        blob = M.build_blob(cfg)
        v, e = await load(dut, blob)
        ee, _ = M.parse_blob(blob, model_d=md or None)
        assert e == ee and v == (ee == 0), "D %d: rtl err %d valid %d, model err %d (MODEL_D %d)" % (D, e, v, ee, md)
    dut._log.info("D check with MODEL_D %d matches the model", md)


@cocotb.test()
async def reload_during_match(dut):
    """Review R6: CTRL begin (a config reload) while the matcher runs stops it at once with no result; the next
    load and match are correct."""
    await setup(dut)
    rng = random.Random(33)
    for t in range(3):
        cfg = rand_cfg(rng, D=256, N=60, n_actions=4, n_slots=2, n_gpio=2)
        q = [rng.randint(-128, 127) for _ in range(256)]
        tune_thresholds(rng, cfg, q)
        vals = [rng.randint(-100, 100) for _ in range(8)]
        assert (await load(dut, M.build_blob(cfg)))[0] == 1
        for w in range(64):
            dut.q_we.value = 1; dut.q_waddr.value = w
            dut.q_wdata.value = int.from_bytes(struct.pack("<4b", *q[4 * w:4 * w + 4]), "little")
            await RisingEdge(dut.clk)
        dut.q_we.value = 0
        for k in range(8):
            dut.v_we.value = 1; dut.v_waddr.value = k; dut.v_wdata.value = vals[k] & 0xFFFFFFFF
            await RisingEdge(dut.clk)
        dut.v_we.value = 0
        await pulse(dut, "go")
        for _ in range(200 + 500 * t):
            await RisingEdge(dut.clk)
        assert int(dut.busy.value), "matcher finished too early for the test"
        await pulse(dut, "begin_i")                      # a reload starts while the matcher runs
        for i in range(40):
            await RisingEdge(dut.clk)
            if not int(dut.busy.value):
                break
        else:
            raise AssertionError("matcher still busy 40 clocks after CTRL begin (review R6)")
        assert not int(dut.done.value), "an aborted match reported a result"
        # a new config and a full match afterwards are right
        cfg2 = rand_cfg(rng, D=32, N=8, n_actions=3, n_slots=2, n_gpio=2)
        q2 = [rng.randint(-128, 127) for _ in range(32)]
        tune_thresholds(rng, cfg2, q2)
        blob2 = M.build_blob(cfg2)
        assert (await load(dut, blob2))[0] == 1
        qq = q2 + [0] * 224
        for w in range(64):
            dut.q_we.value = 1; dut.q_waddr.value = w
            dut.q_wdata.value = int.from_bytes(struct.pack("<4b", *qq[4 * w:4 * w + 4]), "little")
            await RisingEdge(dut.clk)
        dut.q_we.value = 0
        for k in range(8):
            dut.v_we.value = 1; dut.v_waddr.value = k; dut.v_wdata.value = vals[k] & 0xFFFFFFFF
            await RisingEdge(dut.clk)
        dut.v_we.value = 0
        await pulse(dut, "go")
        for _ in range(20000):
            await RisingEdge(dut.clk)
            if int(dut.done.value):
                break
        await ReadOnly()
        _, pc = M.parse_blob(blob2)
        want = M.match(pc, q2, vals)
        assert int(dut.accept.value) == (want["ev"] == M.EV_COMMAND) and dut.sb.value.to_signed() == want["sb"], (t, want)
        await RisingEdge(dut.clk)
    dut._log.info("3 reloads during a match: matcher stopped, next match identical")
