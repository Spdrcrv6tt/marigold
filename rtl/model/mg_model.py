"""Reference model of Marigold's island rules (docs/config-blob.md, docs/architecture.md section 8): config blob
layout and checks, CRC-32 and CRC-8, the matching rule, event frames, GPIO rules and the wake-gated segmenter.
The cocotb tests in rtl/tb check the RTL against it bit for bit."""
import struct
import zlib

MAGIC = b"MGC0"
EV_COMMAND, EV_WAKE, EV_NOMATCH, EV_TIMEOUT = 0, 1, 2, 3
EV_ID = {EV_WAKE: 0xF0, EV_NOMATCH: 0xF1, EV_TIMEOUT: 0xF2}
ERR_NONE, ERR_MAGIC, ERR_VERSION, ERR_RANGE, ERR_LENGTH, ERR_CRC = 0, 1, 2, 3, 4, 5


def crc32(b):
    return zlib.crc32(bytes(b)) & 0xFFFFFFFF


def crc8(b):
    c = 0
    for x in b:
        c ^= x
        for _ in range(8):
            c = ((c << 1) ^ 0x07) & 0xFF if c & 0x80 else (c << 1) & 0xFF
    return c


def blob_len(D, N, n_actions, n_slots, n_gpio, wake_len):
    return 0x20 + 4 * N + N * D + 4 * n_actions + 12 * n_slots + 8 * n_gpio + wake_len + 4


def build_blob(cfg):
    """cfg: dict with D, vectors [(action, flags, bias, [D int8])], actions [(id, slot_mask, gpio_first, gpio_count)],
    slots [(type, min, max)], gpio [(pin_mask, mode, level, slot, pulse_ms)], wake (bytes), host, i2c_addr,
    reject_thr, margin, wake_sens, flags. Returns the blob bytes with its CRC."""
    D, vec = cfg["D"], cfg["vectors"]
    h = MAGIC + struct.pack("<HHHBBBBBBiiHHB3x", 0, D, len(vec), len(cfg["actions"]), len(cfg["slots"]),
                            len(cfg["gpio"]), len(cfg["wake"]), cfg["host"], cfg["i2c_addr"], cfg["reject_thr"],
                            cfg["margin"], cfg["wake_sens"], cfg["flags"], cfg.get("conf_sh", 8))
    b = bytearray(h)
    for a, f, bias, _ in vec:
        b += struct.pack("<BBh", a, f, bias)
    for *_, v in vec:
        b += struct.pack("<%db" % D, *v)
    for a in cfg["actions"]:
        b += struct.pack("<BBBB", *a)
    for sl in cfg["slots"]:
        t, mn, mx = sl[:3]
        lv = tuple(sl[3]) if len(sl) > 3 else (0, 0, 0)
        b += struct.pack("<BBBBii", t, lv[0], lv[1], lv[2], mn, mx)
    for pm, mode, lvl, slot, ms in cfg["gpio"]:
        b += struct.pack("<BBBBHH", pm, mode, lvl, slot, ms, 0)
    b += bytes(cfg["wake"])
    b += struct.pack("<I", crc32(b))
    return bytes(b)


def parse_blob(b):
    """Returns (err, cfg) with the checks of docs/architecture.md section 8.6, in that order."""
    b = bytes(b)
    if len(b) < 0x20 or b[:4] != MAGIC:
        return ERR_MAGIC, None
    ver, D, N, na, ns, ng, wl, host, i2c, rthr, mrg, sens, flags, csh = struct.unpack_from("<HHHBBBBBBiiHHB", b, 4)
    if ver != 0:
        return ERR_VERSION, None
    if not (1 <= na <= 64 and ns <= 8 and ng <= 32 and host <= 2 and 1 <= D <= 256 and N >= 1):
        return ERR_RANGE, None
    L = blob_len(D, N, na, ns, ng, wl)
    if L > 32768 or L != len(b):
        return ERR_LENGTH, None
    if crc32(b[:-4]) != struct.unpack_from("<I", b, L - 4)[0]:
        return ERR_CRC, None
    o = 0x20
    vt = [struct.unpack_from("<BBh", b, o + 4 * i) for i in range(N)]
    o += 4 * N
    vec = [list(struct.unpack_from("<%db" % D, b, o + i * D)) for i in range(N)]
    o += N * D
    acts = [struct.unpack_from("<BBBB", b, o + 4 * j) for j in range(na)]
    o += 4 * na
    slots = []
    for k in range(ns):
        t, l, m, h, mn, mx = struct.unpack_from("<BBBBii", b, o + 12 * k)
        slots.append((t, mn, mx, (l, m, h)))
    o += 12 * ns
    gp = [struct.unpack_from("<BBBBH", b, o + 8 * r) for r in range(ng)]
    o += 8 * ng
    cfg = dict(D=D, N=N, vt=vt, vec=vec, actions=acts, slots=slots, gpio=gp, wake=b[o:o + wl], host=host,
               i2c_addr=i2c, reject_thr=rthr, margin=mrg, wake_sens=sens, flags=flags, conf_sh=csh)
    return ERR_NONE, cfg


def find_action(cfg, aid):
    for a in cfg["actions"]:
        if a[0] == aid:
            return a
    return None


def island_value(raw, slot):
    """A raw slot value as the island emits it (ruling of Nov 15): level codes -1 low, -2 medium, -3 high, -4 max,
    -5 min mapped with the slot's level bytes (0 = default: low = min, medium = floor((min + max) / 2), high = max),
    then clamped as min(max(v, min), max). Same as sw/intent/values.py island_value."""
    _, lo, hi, (l, m, h) = slot
    if -5 <= raw <= -1:
        raw = {-1: l or lo, -2: m or (lo + hi) // 2, -3: h or hi, -4: hi, -5: lo}[raw]
    return min(max(raw, lo), hi)


def match(cfg, q, raw_vals):
    """The matching rule. Returns dict(ev, action, b, sb, s2, conf, values [(slot, clamped)], slot_mask, vals8)."""
    D = cfg["D"]
    s = [sum(q[d] * v[d] for d in range(D)) + vt[2] for vt, v in zip(cfg["vt"], cfg["vec"])]
    b = max(range(len(s)), key=lambda i: (s[i], -i))
    ab = cfg["vt"][b][0]
    others = [s[j] for j in range(len(s)) if cfg["vt"][j][0] != ab]
    s2 = max(others) if others else None
    conf = max(0, min(255, s[b] >> cfg["conf_sh"]))
    rej = ab == 0 or s[b] < cfg["reject_thr"] or (s2 is not None and s[b] - s2 < cfg["margin"])
    act = None if rej else find_action(cfg, ab)
    r = dict(b=b, sb=s[b], s2=s2, conf=conf, action=0, values=[], slot_mask=0, vals8=[0] * 8, ev=EV_NOMATCH)
    if act is None:
        return r
    vals8 = [0] * 8
    values = []
    for k in range(8):
        if act[1] >> k & 1:
            v = island_value(raw_vals[k], cfg["slots"][k]) if k < len(cfg["slots"]) else 0
            vals8[k] = v
            values.append((k, v))
    r.update(ev=EV_COMMAND, action=ab, values=values, slot_mask=act[1], vals8=vals8)
    return r


def frame(ev, action=0, values=(), conf=0):
    f = bytearray([0xA5, ev, action if ev == EV_COMMAND else 0, len(values)])
    for k, v in values:
        f += struct.pack("<Bi", k, v)
    f.append(conf if ev in (EV_COMMAND, EV_NOMATCH) else 0)
    f.append(crc8(f))
    return bytes(f)


def host_gets(cfg, ev):
    if cfg["host"] == 0:
        return False
    return ev == EV_COMMAND or bool(cfg["flags"] >> {EV_WAKE: 1, EV_NOMATCH: 2, EV_TIMEOUT: 3}[ev] & 1)


class Gpio:
    """GPIO rule engine state (section 8.5). pins: current levels; pulses: pin -> (end level, ms left)."""

    def __init__(self):
        self.pins = 0
        self.pulse = {}

    def apply(self, cfg, aid, slot_mask=0, vals8=(0,) * 8):
        act = find_action(cfg, aid)
        if act is None:
            return
        for r in range(act[2], act[2] + act[3]):
            if r >= len(cfg["gpio"]):
                break
            pm, mode, lvl, slot, ms = cfg["gpio"][r]
            pins = [p for p in range(8) if pm >> p & 1]
            if mode == 0:
                for p in pins:
                    self._set(p, lvl & 1)
                    self.pulse[p] = ((lvl & 1) ^ 1, ms)
                continue
            for p in pins:
                self.pulse.pop(p, None)
            if mode == 1:
                for p in pins:
                    self._set(p, lvl & 1)
            elif mode == 2:
                for p in pins:
                    self._set(p, (self.pins >> p & 1) ^ 1)
            elif mode == 3 and slot < 8 and slot_mask >> slot & 1:
                v = max(0, min((1 << len(pins)) - 1, vals8[slot]))
                for j, p in enumerate(pins):
                    self._set(p, v >> j & 1)

    def _set(self, p, v):
        self.pins = (self.pins & ~(1 << p)) | (v << p)

    def settle(self):
        """All pulses run out."""
        for p, (lv, _) in list(self.pulse.items()):
            self._set(p, lv)
        self.pulse = {}


class Segmenter:
    """Wake-gated segment rule (docs/architecture.md section 8.1). step(w, wake, raw_prev) per sample w, where wake is
    the spotter's flag with sample w and raw_prev the raw flag bundled with it (frame w/256 - 1, used only when w is
    a positive multiple of 256). Segments: list of (start, end, timeout); events: list of (sample, ev)."""

    def __init__(self, pre=4800, mn=8000, cap=96000, hang=8, onset=3, listen=64000):
        self.p = dict(pre=pre, mn=mn, cap=cap, hang=hang, onset=onset, listen=listen)
        self.on = False
        self.last_end = 0
        self.h = 0
        self.segs, self.events = [], []

    def step(self, w, wake, raw_prev):
        p = self.p
        fb = w > 0 and w % 256 == 0
        if fb:
            self.h = 0 if raw_prev else self.h + 1
        if not self.on:
            if wake:
                self.on, self.k, self.sp = True, w, 0
                self.start = max(w - p["pre"], self.last_end, 0)
                self.events.append((w, EV_WAKE))
            return
        close = tmo = False
        if fb:
            if raw_prev and w - 256 >= self.k:
                self.sp += 1
            if self.sp < p["onset"]:
                if w - self.k >= p["listen"]:
                    close = tmo = True
            elif w - self.start >= p["mn"] and self.h >= p["hang"]:
                close = True
        if p["cap"] and w - self.start == p["cap"]:
            close = True
        if close:
            self.segs.append((self.start, w, tmo))
            if tmo:
                self.events.append((w, EV_TIMEOUT))
            self.last_end = w
            self.on = False
