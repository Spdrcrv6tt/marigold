"""Island control (mg_ctl): I2C target and registers, config load over I2C, events and frames (I2C reads and UART),
GPIO rules, against rtl/model/mg_model.py. The I2C master is bit-banged at fast-mode (400 kHz) minimum timing."""
import random
import struct
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Edge, Timer, ReadOnly
from cocotb.utils import get_sim_time
import mg_model as M
from mg_util import rand_cfg, tune_thresholds

CLK_NS = 326


class I2C:
    """Bit-banged I2C master on an open-drain bus with a pull-up: a released SDA rises `rise` ns after the release
    (both sides let go), a pull-down is immediate; resolution half a clk_fe period. For every bit the master leaves
    released (the target's data and ACK bits) it records the data-valid time (SCL fall to the line settled) and the
    setup time before SCL rises (review R7)."""

    def __init__(self, dut, t_low=1300, t_high=600, rise=0):
        self.dut, self.tl, self.th, self.rise = dut, t_low, t_high, rise
        self.msda = 1
        self.line = 1
        self.t_change = self.t_fall = 0
        self.tvd_max, self.tsu_min = 0, 10 ** 9
        dut.scl_i.value = 1
        dut.sda_i.value = 1
        cocotb.start_soon(self._wire())

    async def _wire(self):
        rel = None
        while True:
            await Edge(self.dut.clk)
            now = get_sim_time("ns")
            if self.msda and not int(self.dut.sda_oe.value):
                rel = now if rel is None else rel
                v = 1 if now - rel >= self.rise else 0
            else:
                rel, v = None, 0
            if v != self.line:
                self.line, self.t_change = v, now
            self.dut.sda_i.value = v

    async def _bit(self, b):
        self.msda = b
        await Timer(self.tl, unit="ns")
        if b:                                     # the target owns this bit: data valid and setup
            now = get_sim_time("ns")
            if self.t_change > self.t_fall:
                self.tvd_max = max(self.tvd_max, self.t_change - self.t_fall)
            self.tsu_min = min(self.tsu_min, now - max(self.t_change, self.t_fall))
        self.dut.scl_i.value = 1
        await Timer(self.th // 2, unit="ns")
        v = int(self.dut.sda_i.value)
        await Timer(self.th - self.th // 2, unit="ns")
        self.dut.scl_i.value = 0
        self.t_fall = get_sim_time("ns")
        return v

    async def start(self):
        self.msda = 1
        await Timer(self.tl, unit="ns")
        self.dut.scl_i.value = 1
        await Timer(self.th, unit="ns")
        self.msda = 0
        await Timer(self.th, unit="ns")
        self.dut.scl_i.value = 0

    async def stop(self):
        self.msda = 0
        await Timer(self.tl, unit="ns")
        self.dut.scl_i.value = 1
        await Timer(self.th, unit="ns")
        self.msda = 1
        await Timer(2 * self.tl, unit="ns")

    async def wbyte(self, x):
        for i in range(8):
            await self._bit((x >> (7 - i)) & 1)
        return await self._bit(1) == 0           # ACK

    async def rbyte(self, ack):
        v = 0
        for _ in range(8):
            v = (v << 1) | await self._bit(1)
        await self._bit(0 if ack else 1)
        return v

    async def write(self, dev, reg, data=b""):
        await self.start()
        assert await self.wbyte(dev << 1), "no ACK for address 0x%02x" % dev
        assert await self.wbyte(reg)
        for x in data:
            assert await self.wbyte(x)
        await self.stop()

    async def read(self, dev, reg, n):
        await self.start()
        assert await self.wbyte(dev << 1)
        assert await self.wbyte(reg)
        await self.start()                        # repeated start
        assert await self.wbyte((dev << 1) | 1)
        out = [await self.rbyte(i < n - 1) for i in range(n)]
        await self.stop()
        return out


async def setup(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    for s in ("ev_wake", "ev_tmo", "spot_req", "spot_addr", "br_req", "br_we", "br_addr", "br_wdata"):
        getattr(dut, s).value = 0
    dut.rst_n.value = 0
    i2c = I2C(dut)
    await Timer(2, unit="us")
    dut.rst_n.value = 1
    await Timer(2, unit="us")
    return i2c


async def bridge(dut, addr, data=None):
    dut.br_req.value = 1
    dut.br_we.value = 0 if data is None else 1
    dut.br_addr.value = addr
    dut.br_wdata.value = 0 if data is None else data & 0xFFFFFFFF
    await RisingEdge(dut.clk)
    dut.br_req.value = 0
    for _ in range(100):
        await ReadOnly()
        if int(dut.br_ack.value):
            v = int(dut.br_rdata.value)
            await RisingEdge(dut.clk)
            return v
        await RisingEdge(dut.clk)
    raise AssertionError("bridge timeout")


async def pulse(dut, sig):
    getattr(dut, sig).value = 1
    await RisingEdge(dut.clk)
    getattr(dut, sig).value = 0


async def load(i2c, blob, dev=0x2A, after=None):
    await i2c.write(dev, 0x02, b"\x01")
    await i2c.write(dev, 0x03, blob)
    await i2c.write(dev, 0x02, b"\x02")
    st = (await i2c.read(after or dev, 0x01, 1))[0]
    return st


async def settle(dut, cycles=600):
    for _ in range(cycles):
        await RisingEdge(dut.clk)


async def uart_rx(dut, out, bit=27):
    """Decode 8N1 frames on uart_tx_o into the list out."""
    while True:
        await FallingEdge(dut.uart_tx_o)
        for _ in range(bit // 2):
            await RisingEdge(dut.clk)
        v = 0
        for i in range(8):
            for _ in range(bit):
                await RisingEdge(dut.clk)
            v |= int(dut.uart_tx_o.value) << i
        for _ in range(bit):
            await RisingEdge(dut.clk)
        assert int(dut.uart_tx_o.value) == 1, "stop bit"
        out.append(v)


async def read_frames(i2c, dev):
    frames = []
    while True:
        n = (await i2c.read(dev, 0x04, 1))[0]
        if n == 0:
            return frames
        frames.append(bytes(await i2c.read(dev, 0x05, n)))


@cocotb.test()
async def i2c_and_config(dut):
    """ID and STATUS registers, config load over I2C (good and corrupted), address change after commit."""
    i2c = await setup(dut)
    assert (await i2c.read(0x2A, 0x00, 1))[0] == 0x4D
    assert (await i2c.read(0x2A, 0x01, 1))[0] == 0
    rng = random.Random(3)
    for t in range(6):
        cfg = rand_cfg(rng, D=rng.choice([2, 5, 8]), N=rng.randint(1, 6), n_gpio=4)
        blob = M.build_blob(cfg)
        bad = bytearray(blob)
        bad[-2] ^= 0x40
        dev = 0x2A
        st = await load(i2c, bytes(bad), dev)
        assert st >> 4 == M.ERR_CRC and st & 1 == 0, hex(st)
        st = await load(i2c, blob, dev, cfg["i2c_addr"])
        assert st & 0x0B == 0x09 and st >> 4 == 0, hex(st)          # valid, listening
        dev = cfg["i2c_addr"]
        assert (await i2c.read(dev, 0x06, 2)) == [len(blob) & 0xFF, len(blob) >> 8]
        await i2c.write(dev, 0x02, b"\x01")                          # back to the default address
    dut._log.info("I2C registers and 12 loads correct")


@cocotb.test()
async def events_and_gpio(dut):
    """WAKE, TIMEOUT and match events: frames read over I2C or received on the UART, irq_o and GPIO pins equal
    the model, for random configs (host none, I2C, UART; random flags)."""
    i2c = await setup(dut)
    rx = []
    cocotb.start_soon(uart_rx(dut, rx))
    rng = random.Random(8)
    nframes = nev = 0
    dev = 0x2A
    for t in range(16):
        cfg = rand_cfg(rng, D=rng.choice([3, 4, 8]), N=rng.randint(1, 8), n_gpio=rng.randint(0, 12))
        cfg["host"] = t % 3
        q = [rng.randint(-128, 127) for _ in range(cfg["D"])]
        tune_thresholds(rng, cfg, q)
        blob = M.build_blob(cfg)
        await i2c.write(dev, 0x02, b"\x01")
        dev = 0x2A
        st = await load(i2c, blob, dev, cfg["i2c_addr"])
        assert st & 1, hex(st)
        dev = cfg["i2c_addr"]
        _, pc = M.parse_blob(blob)
        gp = M.Gpio()
        gp.pins = int(dut.gpio_o.value)
        gp.pulse = {}
        rx.clear()
        want = []
        for step in range(4):
            kind = rng.choice(["wake", "tmo", "match", "match"])
            if kind == "match":
                vals = [rng.choice([rng.randint(-3000, 3000), -rng.randint(1, 5)]) for _ in range(8)]
                qq = q + [0] * (256 - len(q))
                for w in range(64):
                    await bridge(dut, 0x8100 + 4 * w, int.from_bytes(struct.pack("<4b", *qq[4 * w:4 * w + 4]), "little"))
                for k in range(8):
                    await bridge(dut, 0x8040 + 4 * k, vals[k])
                await bridge(dut, 0x8008, 1)
                for _ in range(500):
                    st = await bridge(dut, 0x8008)
                    if st & 8:                        # accepted: values are in, emit
                        await bridge(dut, 0x8008, 2)
                    if st & 2:
                        break
                r = M.match(pc, q, vals)
                ev = r["ev"]
                f = M.frame(ev, r["action"], r["values"], r["conf"])
                gid = r["action"] if ev == M.EV_COMMAND else 0xF1
                for _ in range(20000):
                    await RisingEdge(dut.clk)
                    if int(dut.u_m.done.value):
                        break
                st = await bridge(dut, 0x8008)
                assert st & 2 and bool(st & 4) == (ev == M.EV_NOMATCH), (st, r)
            else:
                ev = M.EV_WAKE if kind == "wake" else M.EV_TIMEOUT
                await pulse(dut, "ev_wake" if kind == "wake" else "ev_tmo")
                f, gid, r = M.frame(ev), M.EV_ID[ev], dict(slot_mask=0, vals8=[0] * 8)
            nev += 1
            await settle(dut)
            gp.apply(pc, gid, r["slot_mask"], r["vals8"])
            assert int(dut.gpio_o.value) == gp.pins, "t %d %s: gpio %02x, model %02x" % (t, kind, int(dut.gpio_o.value), gp.pins)
            await Timer(6, unit="ms")                 # pulses (<= 5 ms) run out
            gp.settle()
            assert int(dut.gpio_o.value) == gp.pins, "t %d %s settled: gpio %02x, model %02x" % (t, kind, int(dut.gpio_o.value), gp.pins)
            if M.host_gets(pc, ev):
                want.append(f)
            if cfg["host"] == 1 and want:
                exp_irq = (pc["flags"] & 1)
                assert int(dut.irq_o.value) == exp_irq
                got = await read_frames(i2c, dev)
                assert got == want, "t %d: frames %s, model %s" % (t, [g.hex() for g in got], [w.hex() for w in want])
                nframes += len(got)
                want = []
                assert int(dut.irq_o.value) == 0
            if cfg["host"] == 2 and want:
                for _ in range(len(want[0]) * 300 + 2000):
                    await RisingEdge(dut.clk)
                    if len(rx) >= len(want[0]):
                        break
                await settle(dut, 400)
                assert bytes(rx) == want[0], "t %d uart %s model %s" % (t, bytes(rx).hex(), want[0].hex())
                nframes += 1
                rx.clear()
                want = []
            if cfg["host"] == 0:
                assert (await i2c.read(dev, 0x04, 1))[0] == 0
    dut._log.info("%d events, %d frames identical, GPIO equal", nev, nframes)


@cocotb.test()
async def i2c_timing(dut):
    """Review R7: data-valid and setup times of the target's bits on a bus with rise time. Standard mode (100 kHz:
    tLOW 4.7 us, tHIGH 4.0 us, rise 1000 ns, the spec's worst case) must meet tVD;DAT <= 3.45 us and tSU;DAT >= 250 ns.
    Fast mode (400 kHz: tLOW 1.3 us, tHIGH 0.6 us, rise 300 ns) is measured and reported, not required: the target
    changes SDA 2 to 3 clk_fe periods after SCL falls, so with a 300 ns rise its bits miss tVD;DAT (0.9 us) and
    setup; iteration 1 supports standard mode only (docs/architecture.md section 6)."""
    i2c = await setup(dut)
    rng = random.Random(9)
    res = {}
    for mode, tl, th, rise, tvd, tsu in (("standard", 4700, 4000, 1000, 3450, 250), ("fast", 1300, 600, 300, 900, 100)):
        i2c.tl, i2c.th, i2c.rise = tl, th, rise
        i2c.tvd_max, i2c.tsu_min = 0, 10 ** 9
        try:
            assert (await i2c.read(0x2A, 0x00, 1))[0] == 0x4D
            cfg = rand_cfg(rng, D=4, N=2, n_gpio=2)
            blob = M.build_blob(cfg)
            st = await load(i2c, blob, 0x2A, cfg["i2c_addr"])
            assert st & 1, hex(st)
            await i2c.write(cfg["i2c_addr"], 0x02, b"\x01")       # back to the default address
            ok = True
        except AssertionError:
            if mode == "standard":
                raise
            ok = False
            dut._log.info("fast mode, rise %d ns: the transfer FAILED (a target bit or ACK not valid in time)", rise)
            i2c.msda = 1
            await i2c.stop()
            dut.rst_n.value = 0
            await Timer(2, unit="us")
            dut.rst_n.value = 1
            await Timer(2, unit="us")
        res[mode] = (i2c.tvd_max, i2c.tsu_min, ok)
        dut._log.info("%s mode, rise %d ns: data valid max %d ns (limit %d), setup min %d ns (limit %d)",
                      mode, rise, i2c.tvd_max, tvd, i2c.tsu_min, tsu)
    assert res["standard"][0] <= 3450 and res["standard"][1] >= 250, res
