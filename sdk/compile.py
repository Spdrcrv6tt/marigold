#!/usr/bin/env python
"""Marigold SDK: product config JSON (sdk/config-format.md) -> config blob v0 (docs/config-blob.md), and back.

  python sdk/compile.py CONFIG.json [-o OUT.mgc] [--embedder DIR|stub] [--expansion FILE|none] [--dim 256]
                        [--projection random|none|FILE.npy] [--protos 0=auto] [--reject-protos 16]
                        [--q-scale S] [--reject-rate 0.90] [--margin 0.0]
  python sdk/compile.py --parse BLOB.mgc

What the compiler does: validate the config (SDK rules), gather each action's phrases (the typed
examples plus the SDK expansion of that config, sdk/expand.py), embed them with the command
embedder (newest data/gate0b/models/* by default), project to D dimensions, cluster each action
into up to --protos prototypes, add --reject-protos action-0 prototypes from the expansion's
non-command set, quantize to int8, calibrate the reject threshold so that --reject-rate of the
non-commands are rejected under the matching rule, and pack the blob. A JSON report goes next to
the blob (OUT.mgc.json). The compiler never reads eval/.

PLACEHOLDERS until the model agent writes docs/golden-format.md and docs/golden-wake.md:
  - projection: a fixed, seeded random orthonormal matrix (embedder dim -> D), the same for every
    product (never fit per product), or a .npy matrix [embedder_dim, D] supplied by the model agent;
  - q scale (the encoder's int8 output scale): default 127 / max |component| over the calibration
    phrases; set --q-scale once golden-format.md fixes it;
  - wake template: byte 0x00 (placeholder format tag) + ASCII wake phrase; wake_sens = sensitivity x 65535;
  - GPIO rules without a slot carry slot 0.
"""
import argparse, json, struct, sys, zlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAGIC, VERSION, MAX_BLOB = b"MGC0", 0, 32768
HDR = struct.Struct("<4sHHHBBBBBBiiHHI")      # 32 bytes, docs/config-blob.md offsets 0x00-0x1F
VEC = struct.Struct("<BBh")                   # action u8, flags u8, bias i16
ACT = struct.Struct("<BBBB")                  # id, slot_mask, gpio_first, gpio_count
SLOT = struct.Struct("<B3xii")                # type, pad 3, min, max
GPIO = struct.Struct("<BBBBHH")               # pin_mask, mode, level, slot, pulse_ms, pad
SLOT_TYPES = {"number": 0, "duration": 1, "level": 2}
GPIO_MODES = {"pulse": 0, "latch": 1, "toggle": 2, "binary": 3}
HOSTS = {"none": 0, "i2c": 1, "uart": 2}
EVENT_IDS = {"wake": 0xF0, "nomatch": 0xF1, "timeout": 0xF2}      # action-table event entries (config-blob.md)
EVENT_FLAG_BITS = {"wake": 1, "nomatch": 2, "timeout": 3}          # flags bits: send the event to the host
MAX_ACTION_ID = 0xEF                                               # 0xF0-0xFF are reserved for events


# ------------------------------------------------------------------ packing (exact layout)
def pack(h, vec_action, vec_flags, bias, V, actions, slots, gpio, wake):
    """h: dict with D, reject_thr, margin, wake_sens, flags, host, i2c_addr. V: int8 [N, D]."""
    V = np.asarray(V, dtype=np.int8)
    N, D = V.shape
    assert D == h["D"] and len(vec_action) == len(vec_flags) == len(bias) == N
    assert 1 <= len(actions) <= 64 and len(slots) <= 8 and len(gpio) <= 32 and len(wake) <= 255
    out = bytearray(HDR.pack(MAGIC, VERSION, D, N, len(actions), len(slots), len(gpio), len(wake), h["host"],
                             h["i2c_addr"], int(h["reject_thr"]), int(h["margin"]), h["wake_sens"], h["flags"], 0))
    for a, f, b in zip(vec_action, vec_flags, bias):
        out += VEC.pack(int(a), int(f), int(b))
    out += V.tobytes(order="C")
    for a in actions:
        out += ACT.pack(a["id"], a["slot_mask"], a["gpio_first"], a["gpio_count"])
    for s in slots:
        out += SLOT.pack(s["type"], int(s["min"]), int(s["max"]))
    for g in gpio:
        out += GPIO.pack(g["pin_mask"], g["mode"], g["level"], g["slot"], g["pulse_ms"], 0)
    out += bytes(wake)
    out += struct.pack("<I", zlib.crc32(bytes(out)) & 0xFFFFFFFF)
    return bytes(out)


def parse(blob):
    """Inverse of pack(); checks magic, version, sizes and CRC. Returns a dict of numpy arrays and tables."""
    if len(blob) < HDR.size + 4:
        raise ValueError("blob too short")
    crc = struct.unpack_from("<I", blob, len(blob) - 4)[0]
    if zlib.crc32(blob[:-4]) & 0xFFFFFFFF != crc:
        raise ValueError("CRC-32 mismatch")
    magic, ver, D, N, na, ns, ng, wl, host, i2c, thr, margin, wsens, flags, res = HDR.unpack_from(blob, 0)
    if magic != MAGIC or ver != VERSION:
        raise ValueError(f"bad magic/version {magic!r} {ver}")
    # the chip's config checks (docs/architecture.md 8.6)
    if not (1 <= na <= 64 and ns <= 8 and ng <= 32 and host <= 2 and 1 <= D <= 256):
        raise ValueError(f"config check failed: n_actions {na}, n_slots {ns}, n_gpio {ng}, host {host}, D {D}")
    if 0x20 + 4 * N + N * D + 4 * na + 12 * ns + 8 * ng + wl + 4 != len(blob) or len(blob) > MAX_BLOB:
        raise ValueError("config check failed: length")
    o = HDR.size
    vt = [VEC.unpack_from(blob, o + i * VEC.size) for i in range(N)]
    o += N * VEC.size
    V = np.frombuffer(blob, dtype=np.int8, count=N * D, offset=o).reshape(N, D).copy()
    o += N * D
    actions = [dict(zip(("id", "slot_mask", "gpio_first", "gpio_count"), ACT.unpack_from(blob, o + i * ACT.size))) for i in range(na)]
    o += na * ACT.size
    slots = [dict(zip(("type", "min", "max"), SLOT.unpack_from(blob, o + i * SLOT.size))) for i in range(ns)]
    o += ns * SLOT.size
    gpio = [dict(zip(("pin_mask", "mode", "level", "slot", "pulse_ms"), GPIO.unpack_from(blob, o + i * GPIO.size)[:5])) for i in range(ng)]
    o += ng * GPIO.size
    wake = blob[o:o + wl]
    o += wl
    if o + 4 != len(blob):
        raise ValueError(f"size mismatch: parsed {o + 4} bytes, blob has {len(blob)}")
    return dict(D=D, N=N, host=host, i2c_addr=i2c, reject_thr=thr, margin=margin, wake_sens=wsens, flags=flags,
                reserved=res, vec_action=np.array([v[0] for v in vt], dtype=np.int32), vec_flags=np.array([v[1] for v in vt]),
                bias=np.array([v[2] for v in vt], dtype=np.int32), vectors=V, actions=actions, slots=slots, gpio=gpio,
                wake=bytes(wake), crc32=crc, size=len(blob))


def match(b, q):
    """Matching rule of docs/config-blob.md, bit for bit: returns the action id, or None for reject."""
    s = b["vectors"].astype(np.int32) @ np.asarray(q, dtype=np.int32) + b["bias"]
    i = int(np.argmax(s))                      # first maximum: lowest index wins ties
    a = int(b["vec_action"][i])
    if a == 0 or s[i] < b["reject_thr"]:
        return None
    other = s[b["vec_action"] != a]
    if other.size and s[i] - other.max() < b["margin"]:
        return None
    return a


def crc8(data):
    """CRC-8/SMBus PEC: polynomial 0x07, init 0, no reflection, no final XOR (architecture 8.4)."""
    c = 0
    for byte in data:
        c ^= byte
        for _ in range(8):
            c = ((c << 1) ^ 0x07) & 0xFF if c & 0x80 else (c << 1) & 0xFF
    return c


def host_frame(etype, action=0, values=(), confidence=0):
    """Host frame (config-blob.md): 0xA5, type, action, n_values, (slot u8, i32 LE)*, confidence, crc8."""
    f = bytearray([0xA5, etype, action, len(values)])
    for slot, v in values:
        f += struct.pack("<Bi", slot, v)
    f.append(confidence)
    f.append(crc8(f))
    return bytes(f)


# ------------------------------------------------------------------ config rules
def validate(c):
    err = []
    if c.get("format") != "marigold-config/0":
        err.append("format must be marigold-config/0")
    acts = c.get("actions", [])
    if not 1 <= len(acts) <= 64:
        err.append("1 to 64 actions")
    ids = [a.get("id") for a in acts]
    if len(set(ids)) != len(ids) or not all(isinstance(i, int) and 1 <= i <= MAX_ACTION_ID for i in ids):
        err.append("action ids must be unique integers 1 to 239 (0xF0 and up are reserved for events)")
    slots = list(c.get("values", {}))
    ev = c.get("events", {})
    if not isinstance(ev, dict) or set(ev) - set(EVENT_IDS):
        err.append("events may only contain wake, nomatch and timeout")
    else:
        for n, e in ev.items():
            if set(e) - {"host", "gpio"}:
                err.append(f"event {n}: only 'host' and 'gpio' are allowed")
            for g in gpio_list(e):
                err += gpio_errors(f"event {n}", g, slots, allow_binary=False)
    if len(acts) + sum(1 for e in ev.values() if isinstance(e, dict) and gpio_list(e)) > 64:
        err.append("actions plus event entries exceed 64")
    if sum(len(gpio_list(a)) for a in acts) + sum(len(gpio_list(e)) for e in ev.values() if isinstance(e, dict)) > 32:
        err.append("more than 32 GPIO rules")
    slots = list(c.get("values", {}))
    if len(slots) > 8:
        err.append("at most 8 value slots")
    for n, v in c.get("values", {}).items():
        if v.get("type") not in SLOT_TYPES:
            err.append(f"slot {n}: type must be number, duration or level")
    owner = {}
    for a in acts:
        ex = a.get("examples", [])
        if len(ex) < 5:
            err.append(f"action {a.get('name')}: needs at least 5 examples")
        for e in ex:
            k = " ".join(e.lower().split())
            if k in owner and owner[k] != a.get("name"):
                err.append(f"example '{e}' appears under {owner[k]} and {a.get('name')}")
            owner[k] = a.get("name")
        for s in a.get("values", []):
            if s not in slots:
                err.append(f"action {a.get('name')}: unknown slot {s}")
        for g in gpio_list(a):
            err += gpio_errors(f"action {a.get('name')}", g, slots)
    host = c.get("output", {}).get("host", "none")
    if host not in HOSTS:
        err.append("output.host must be i2c, uart or none")
    if err:
        raise ValueError("; ".join(err))


def gpio_level(g):
    """Pulse: active level, default 1 (high for pulse_ms, then low; architecture 8.5). Latch: the level. Toggle: 0."""
    return int(g.get("level", 1)) if g["mode"] == "pulse" else int(g.get("level", 0)) if g["mode"] == "latch" else 0


def gpio_errors(where, g, slots, allow_binary=True):
    e = []
    if not g.get("pins") or not all(isinstance(p, int) and 0 <= p <= 7 for p in g["pins"]) or len(set(g["pins"])) != len(g["pins"]):
        e.append(f"{where}: GPIO pins must be distinct, 0 to 7")
    if g.get("mode") not in ("pulse", "latch", "toggle"):
        e.append(f"{where}: GPIO mode must be pulse, latch or toggle")
    if g.get("encode") not in (None, "binary") or (g.get("encode") and not allow_binary):
        e.append(f"{where}: encode must be 'binary' (actions only)")
    if g.get("encode") == "binary" and g.get("value") not in slots:
        e.append(f"{where}: binary GPIO needs a known value slot")
    if g.get("mode") == "pulse" and not (isinstance(g.get("pulse_ms"), int) and 1 <= g["pulse_ms"] <= 65535):
        e.append(f"{where}: pulse needs pulse_ms 1 to 65535")
    if g.get("mode") == "latch" and not g.get("encode") and g.get("level") not in (0, 1):
        e.append(f"{where}: latch needs level 0 or 1")
    if "level" in g and g["level"] not in (0, 1):
        e.append(f"{where}: level must be 0 or 1")
    return e


def gpio_list(a):
    g = a.get("gpio")
    return [] if not g else (g if isinstance(g, list) else [g])


def tables(c):
    names = list(c.get("values", {}))
    slots = []
    for n in names:
        v = c["values"][n]
        lo, hi = (v["min_s"], v["max_s"]) if v["type"] == "duration" else (v["min"], v["max"])
        slots.append(dict(type=SLOT_TYPES[v["type"]], min=int(lo), max=int(hi)))
    actions, gpio = [], []
    for a in c["actions"]:
        first = len(gpio)
        for g in gpio_list(a):
            binary = g.get("encode") == "binary"
            gpio.append(dict(pin_mask=sum(1 << p for p in g["pins"]), mode=GPIO_MODES["binary" if binary else g["mode"]],
                             level=0 if binary else gpio_level(g), slot=names.index(g["value"]) if binary else 0,
                             pulse_ms=int(g.get("pulse_ms", 0))))
        actions.append(dict(id=a["id"], slot_mask=sum(1 << names.index(s) for s in a.get("values", [])),
                            gpio_first=first if len(gpio) > first else 0, gpio_count=len(gpio) - first))
    flags = 0
    for n, e in sorted(c.get("events", {}).items(), key=lambda kv: EVENT_IDS[kv[0]]):
        if e.get("host"):
            flags |= 1 << EVENT_FLAG_BITS[n]
        rules = gpio_list(e)
        if rules:                                  # event entry: no vectors point at it; its rules fire on the event
            first = len(gpio)
            for g in rules:
                gpio.append(dict(pin_mask=sum(1 << p for p in g["pins"]), mode=GPIO_MODES[g["mode"]], level=gpio_level(g),
                                 slot=0, pulse_ms=int(g.get("pulse_ms", 0))))
            actions.append(dict(id=EVENT_IDS[n], slot_mask=0, gpio_first=first, gpio_count=len(rules)))
    out = c.get("output", {})
    wake = c.get("wake", {})
    w = bytes([0x00]) + wake.get("phrase", "").encode("ascii", "replace")[:254]
    h = dict(host=HOSTS[out.get("host", "none")], i2c_addr=int(str(out.get("i2c_address", "0")), 0) & 0x7F,
             flags=flags | (1 if out.get("irq_pin") else 0), wake_sens=int(round(float(wake.get("sensitivity", 0.5)) * 65535)))
    return h, actions, slots, gpio, w


# ------------------------------------------------------------------ embedding side
class StubEmbedder:
    """Deterministic hashed bag of words, for tests only (no model)."""
    dim = 384

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim))
        for i, t in enumerate(texts):
            for w in t.lower().split():
                h = zlib.crc32(w.encode())
                out[i, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)


class STEmbedder:
    def __init__(self, path):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(str(path), device="cpu")
        self.dim = self.m.get_sentence_embedding_dimension()

    def encode(self, texts):
        return self.m.encode(list(texts), batch_size=64, normalize_embeddings=True, convert_to_numpy=True)


def newest_embedder():
    """The model agent's frozen teacher (data/models/teacher-v3) if present, else the newest data/gate0b/models/*."""
    t = ROOT / "data/models/teacher-v3"
    if (t / "modules.json").exists():
        return t
    ms = sorted((p for p in (ROOT / "data/gate0b/models").glob("*") if (p / "modules.json").exists()), key=lambda p: p.stat().st_mtime)
    return ms[-1] if ms else None


def projection(kind, dim_in, D):
    if kind == "none":
        return None
    if kind == "random":
        g = np.random.default_rng(20261115).standard_normal((dim_in, D))
        q, _ = np.linalg.qr(g)                 # orthonormal columns, fixed seed: same for every product
        return q[:, :D]
    P = np.load(kind)
    assert P.shape == (dim_in, D), f"projection {kind} has shape {P.shape}, need {(dim_in, D)}"
    return P


def unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def kmeans(X, k, iters=50):
    """Spherical k-means, deterministic farthest-point start."""
    k = min(k, len(X))
    C = [unit(X.mean(0, keepdims=True))[0]]
    C = [X[int(np.argmax(X @ C[0]))]]
    while len(C) < k:
        C.append(X[int(np.argmin(np.max(X @ np.array(C).T, axis=1)))])
    C = np.array(C)
    for _ in range(iters):
        a = np.argmax(X @ C.T, axis=1)
        new = unit(np.array([X[a == j].mean(0) if np.any(a == j) else C[j] for j in range(k)]))
        if np.allclose(new, C):
            break
        C = new
    return C


def compile_config(c, key, embedder, expansion=None, D=256, proj="random", protos=0, reject_protos=16,
                   q_scale=None, reject_rate=0.90, margin=0.0):
    validate(c)
    h, actions, slots, gpio, wake = tables(c)
    exp_acts = (expansion or {}).get("actions", {}).get(key, {})
    negs = list((expansion or {}).get("negatives", {}).get(key, []))
    phrases, labels = [], []
    for a in c["actions"]:
        ex = list(dict.fromkeys(a["examples"] + exp_acts.get(a["name"], [])))
        phrases += ex
        labels += [a["id"]] * len(ex)
    labels = np.array(labels)
    E = embedder.encode(phrases + negs)
    P = projection(proj, E.shape[1], D)
    if P is None:
        D = E.shape[1]
    E = unit(E @ P) if P is not None else unit(E)
    if not 1 <= D <= 256:
        raise ValueError(f"D = {D}: the chip accepts D <= 256 (architecture 8.6); use --projection with --dim")
    Ep, En = E[:len(phrases)], E[len(phrases):]
    fixed = HDR.size + len(actions) * ACT.size + len(slots) * SLOT.size + len(gpio) * GPIO.size + len(wake) + 4
    if protos <= 0:                            # auto: fill the 32 KB blob, at most 32 prototypes per action
        budget = (MAX_BLOB - fixed) // (VEC.size + D) - (reject_protos if len(En) else 0)
        protos = max(1, min(32, budget // len(c["actions"])))
    while True:                                # shrink prototypes until the blob fits
        C, va = [], []
        for a in c["actions"]:
            cent = kmeans(Ep[labels == a["id"]], protos)
            C.append(cent)
            va += [a["id"]] * len(cent)
        if len(En) and reject_protos:
            cent = kmeans(En, reject_protos)
            C.append(cent)
            va += [0] * len(cent)
        C = np.concatenate(C)
        size = fixed + len(C) * (VEC.size + D)
        if size <= MAX_BLOB or (protos == 1 and reject_protos <= 1):
            break
        protos, reject_protos = max(1, protos - 1), max(1, reject_protos - 1)
    if size > MAX_BLOB:
        raise ValueError(f"blob would be {size} bytes, over {MAX_BLOB}")
    sv = 127.0 / float(np.max(np.abs(C)))
    V = np.clip(np.round(C * sv), -127, 127).astype(np.int8)
    sq = float(q_scale) if q_scale else 127.0 / float(np.max(np.abs(E)))
    qi = lambda X: np.clip(np.round(X * sq), -127, 127).astype(np.int32)
    va = np.array(va, dtype=np.int32)
    Sp, Sn = qi(Ep) @ V.astype(np.int32).T, qi(En) @ V.astype(np.int32).T if len(En) else None
    m_int = int(round(margin * sq * sv))
    thr_cfg = c.get("match", {}).get("reject_threshold")
    calib = {}
    if thr_cfg is not None:
        thr = int(round(float(thr_cfg) * sq * sv))
        calib["source"] = "config reject_threshold (cosine units) x q_scale x vector scale"
    elif Sn is not None:
        bi = np.argmax(Sn, axis=1)
        best = Sn[np.arange(len(Sn)), bi]
        open_ = np.array([va[bi[i]] != 0 and (not np.any(va != va[bi[i]]) or best[i] - Sn[i][va != va[bi[i]]].max() >= m_int)
                          for i in range(len(Sn))], dtype=bool)   # not already rejected by a reject vector or the margin
        allowed = int(np.floor((1 - reject_rate) * len(Sn)))
        cand = np.sort(best[open_])[::-1]
        thr = int(cand[allowed]) + 1 if len(cand) > allowed else int(Sp.max(axis=1).min())
        calib["source"] = f"calibrated: {reject_rate:.0%} of {len(Sn)} expansion non-commands rejected"
    else:
        thr = int(round(0.5 * sq * sv))
        calib["source"] = "PLACEHOLDER: no non-command set, cosine 0.5"
    h.update(D=D, reject_thr=thr, margin=m_int)
    blob = pack(h, va, np.zeros(len(va), int), np.zeros(len(va), int), V, actions, slots, gpio, wake)
    b = parse(blob)
    pred = np.array([match(b, q) or 0 for q in qi(Ep)])
    calib.update(pos_phrases=len(phrases), pos_accept=float(np.mean(pred != 0)), pos_correct=float(np.mean(pred == labels)),
                 note="pos_* are on the SDK's own phrases (examples + expansion), not a test")
    if len(En):
        calib.update(neg_phrases=len(negs), neg_rejected=float(np.mean([match(b, q) is None for q in qi(En)])))
    report = dict(product=c.get("product"), key=key, size_bytes=len(blob), crc32=f"{b['crc32']:08x}", D=D, N=b["N"],
                  protos_per_action=protos, reject_protos=int(np.sum(va == 0)), vector_scale=sv, q_scale=sq,
                  q_scale_source="--q-scale" if q_scale else "PLACEHOLDER 127/max|calibration component|",
                  projection=proj if P is not None else "none", embedder_dim=int(E.shape[1]) if P is None else int(P.shape[0]),
                  reject_thr=thr, margin=m_int, calibration=calib,
                  value_words={n: v.get("words") for n, v in c.get("values", {}).items() if v.get("words")},
                  placeholders=["projection", "q_scale", "wake template format", "wake_sens units"])
    return blob, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("-o", "--out")
    ap.add_argument("--parse")
    ap.add_argument("--embedder", default=None, help="sentence-transformers dir, or 'stub' (tests); default newest data/gate0b/models/*")
    ap.add_argument("--expansion", default=str(ROOT / "data/expand/expanded_qwen25.json"))
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--projection", default="random")
    ap.add_argument("--protos", type=int, default=0, help="prototypes per action; 0 = fill the 32 KB budget (max 32)")
    ap.add_argument("--reject-protos", type=int, default=16)
    ap.add_argument("--q-scale", type=float, default=None)
    ap.add_argument("--reject-rate", type=float, default=0.90, help="share of the expansion non-commands to reject (0.90 chosen on the eval DEV split)")
    ap.add_argument("--margin", type=float, default=0.0)
    a = ap.parse_args()
    if a.parse:
        b = parse(Path(a.parse).read_bytes())
        print(json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) and v.ndim == 1 else
                              f"int8[{v.shape[0]}x{v.shape[1]}]" if isinstance(v, np.ndarray) else
                              v.hex() if isinstance(v, bytes) else v) for k, v in b.items()}, indent=1))
        return
    cfg = Path(a.config)
    c = json.load(open(cfg, encoding="utf-8"))
    emb_path = None if a.embedder == "stub" else Path(a.embedder) if a.embedder else newest_embedder()
    if a.embedder != "stub" and emb_path is None:
        sys.exit("no embedder: pass --embedder DIR or --embedder stub")
    emb = StubEmbedder() if a.embedder == "stub" else STEmbedder(emb_path)
    exp = json.load(open(a.expansion, encoding="utf-8")) if a.expansion != "none" and Path(a.expansion).exists() else None
    blob, rep = compile_config(c, cfg.stem, emb, exp, a.dim, a.projection, a.protos, a.reject_protos, a.q_scale, a.reject_rate, a.margin)
    rep.update(config=str(cfg), embedder=str(emb_path) if emb_path else "stub", expansion=a.expansion if exp else None)
    out = Path(a.out) if a.out else ROOT / "data/sdk" / f"{cfg.stem}.mgc"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    json.dump(rep, open(f"{out}.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: rep[k] for k in ("product", "size_bytes", "D", "N", "reject_thr", "calibration")}, indent=1))


if __name__ == "__main__":
    main()
