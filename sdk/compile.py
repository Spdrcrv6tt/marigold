#!/usr/bin/env python
"""Marigold SDK: product config JSON (sdk/config-format.md) -> config blob v1 (docs/config-blob.md), and back.

  python sdk/compile.py CONFIG.json [-o OUT.mgc] [--model-tag TAG] [--embedder DIR|stub] [--expansion FILE|none] [--dim D]
                        [--projection random|none|FILE.npy] [--protos 0=auto] [--reject-protos 16]
                        [--q-scale S] [--reject-rate R] [--margin 0.0]
                        [--enroll audio|text] [--enroll-clips MANIFEST] [--golden MODEL.npz]
  python sdk/compile.py --parse BLOB.mgc

Enrollment (review K2, Nov 17, 2026): the default, --enroll audio, takes the prototypes and the reject threshold from
speech: TTS recordings of the config's phrases by the SDK's own training voices (data/tts/jobs_enroll.jsonl, from
sw/intent/enroll_jobs.py and the TTS build), run through the chip model's integer export (weights/mg-<tag>.npz). The
threshold and margin are calibrated on those audio scores with one voice held out at a time (enroll_audio). Text
prototypes with a text-calibrated threshold, --enroll text, rejected 13 of 13 spoken commands on a real voice (John's
first recording, Nov 17) and nearly all from new synthetic voices (the review); they remain for tests and research.
The audio default rejects 95% of the non-commands (--reject-rate 0.95, plan target 4; default_reject_rate).
Everything below describes the text step, which still builds the blob's tables and wake template in both modes.

What the compiler does: validate the config (SDK rules), gather each action's phrases (the typed
examples plus the SDK expansion of that config, sdk/expand.py), embed them with the teacher of the
chip model the blob is for (--model-tag, default the shipping model; data/models/<its teacher> by
default, sdk/chip_models.py; an --embedder from another teacher is refused), project to D dimensions, cluster each action
into up to --protos prototypes, add --reject-protos action-0 prototypes from the expansion's
non-command set, quantize to int8, calibrate the reject threshold so that --reject-rate of the
non-commands are rejected under the matching rule, and pack the blob. A JSON report goes next to
the blob (OUT.mgc.json). The blob records the model and the teacher (header bytes 0x1D and 0x1E; review K1), and
the chip refuses a blob made for another model; every compile prints both. The compiler never reads eval/.

Wake template and wake_sens: sw/wake/template.py (format in docs/golden-wake.md): CMUdict phones of the
typed phrase (up to 4 pronunciation variants; compound split or letter rules for unknown words, with a
warning in the report), wake_sens from the config's sensitivity. A config may give "phones" (a list of
ARPAbet strings) under "wake" to override the lexicon.

Vectors (docs/golden-format.md section 5, model agent, Nov 15): with the frozen teacher
(data/models/teacher-v3) the projection is the teacher's own marigold_proj.npz (mu, P), applied as
v = P (t - mu), v -= mean(v), v /= rms(v), so phrases live in the encoder's unit-RMS, zero-mean D-space;
prototypes are spherical k-means centroids put back on that rule; int8 = clip(round(32 v), +-127)
(the encoder's q has 5 fractional bits). conf_sh (byte 0x1C) = log2(D) + 2 (9 at D = 128). Level slots
carry the config's low / medium / high word values in slot bytes 1-3 (0 = default rule).
--projection random|none|FILE.npy and the stub embedder remain for tests. GPIO rules without a slot carry slot 0.
"""
import argparse, json, struct, sys, zlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sw/wake"))
import template as wake_tpl                   # sw/wake/template.py (wake agent)
import chip_models as CM                      # sdk/chip_models.py: the models, their ids and teachers (K1)


def wake_template(wake):
    """Config "wake" object -> (template bytes, wake_sens u16, report)."""
    return wake_tpl.build(wake.get("phrase", ""), float(wake.get("sensitivity", 0.5)), wake.get("phones"))
MAGIC, VERSION, MAX_BLOB = b"MGC0", 1, 32768     # version 1: model and teacher ids (K1, Nov 17, 2026)
DEFAULT_EXPANSION = ROOT / "data/expand/expanded_qwen25.json"      # written by sdk/expand.py (DEFAULT_OUT)
GOLDEN_D = 128          # docs/golden-format.md v1.0: the model's q has D = 128 (review K7: was a 256 default)
HDR = struct.Struct("<4sHHHBBBBBBiiHHBBBx")   # 32 bytes, docs/config-blob.md 0x00-0x1F (0x1C conf_sh, 0x1D model, 0x1E teacher)
VEC = struct.Struct("<BBh")                   # action u8, flags u8, bias i16
ACT = struct.Struct("<BBBB")                  # id, slot_mask, gpio_first, gpio_count
SLOT = struct.Struct("<BBBBii")               # type, low, medium, high (level words), min, max
GPIO = struct.Struct("<BBBBHH")               # pin_mask, mode, level, slot, pulse_ms, pad
SLOT_TYPES = {"number": 0, "duration": 1, "level": 2}
GPIO_MODES = {"pulse": 0, "latch": 1, "toggle": 2, "binary": 3}
HOSTS = {"none": 0, "i2c": 1, "uart": 2}
EVENT_IDS = {"wake": 0xF0, "nomatch": 0xF1, "timeout": 0xF2}      # action-table event entries (config-blob.md)
EVENT_FLAG_BITS = {"wake": 1, "nomatch": 2, "timeout": 3}          # flags bits: send the event to the host
MAX_ACTION_ID = 0xEF                                               # 0xF0-0xFF are reserved for events


# ------------------------------------------------------------------ packing (exact layout)
def pack(h, vec_action, vec_flags, bias, V, actions, slots, gpio, wake):
    """h: dict with D, reject_thr, margin, wake_sens, flags, host, i2c_addr, model_id, teacher_id (no ids: 0, a blob
    no chip loads). V: int8 [N, D]."""
    V = np.asarray(V, dtype=np.int8)
    N, D = V.shape
    assert D == h["D"] and len(vec_action) == len(vec_flags) == len(bias) == N
    assert 1 <= len(actions) <= 64 and len(slots) <= 8 and len(gpio) <= 32 and len(wake) <= 255
    out = bytearray(HDR.pack(MAGIC, VERSION, D, N, len(actions), len(slots), len(gpio), len(wake), h["host"],
                             h["i2c_addr"], int(h["reject_thr"]), int(h["margin"]), h["wake_sens"], h["flags"],
                             int(h.get("conf_sh", conf_shift(D))), int(h.get("model_id", 0)), int(h.get("teacher_id", 0))))
    for a, f, b in zip(vec_action, vec_flags, bias):
        out += VEC.pack(int(a), int(f), int(b))
    out += V.tobytes(order="C")
    for a in actions:
        out += ACT.pack(a["id"], a["slot_mask"], a["gpio_first"], a["gpio_count"])
    for s in slots:
        out += SLOT.pack(s["type"], int(s.get("low", 0)), int(s.get("medium", 0)), int(s.get("high", 0)), int(s["min"]), int(s["max"]))
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
    magic, ver, D, N, na, ns, ng, wl, host, i2c, thr, margin, wsens, flags, conf_sh, mid, tid = HDR.unpack_from(blob, 0)
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
    slots = [dict(zip(("type", "low", "medium", "high", "min", "max"), SLOT.unpack_from(blob, o + i * SLOT.size))) for i in range(ns)]
    o += ns * SLOT.size
    gpio = [dict(zip(("pin_mask", "mode", "level", "slot", "pulse_ms"), GPIO.unpack_from(blob, o + i * GPIO.size)[:5])) for i in range(ng)]
    o += ng * GPIO.size
    wake = blob[o:o + wl]
    o += wl
    if o + 4 != len(blob):
        raise ValueError(f"size mismatch: parsed {o + 4} bytes, blob has {len(blob)}")
    return dict(D=D, N=N, host=host, i2c_addr=i2c, reject_thr=thr, margin=margin, wake_sens=wsens, flags=flags,
                conf_sh=conf_sh, model_id=mid, teacher_id=tid, vec_action=np.array([v[0] for v in vt], dtype=np.int32), vec_flags=np.array([v[1] for v in vt]),
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


def conf_shift(D):
    """conf_sh so that a perfect match (about D * 2^10 with 5-bit q and v) maps to about 255 (golden-format 4)."""
    return int(round(np.log2(D))) + 2


def confidence(b, s_best):
    """architecture.md 8.3 with conf_sh from the blob."""
    return int(min(max(int(s_best) >> b["conf_sh"], 0), 255))


def golden_vec(T, mu, P):
    """golden-format section 5: teacher vectors (L2-normalised) -> unit-RMS, zero-mean D vectors (float)."""
    v = (np.asarray(T, np.float64) - mu) @ P.T
    v = v - v.mean(1, keepdims=True)
    return v / np.sqrt((v * v).mean(1, keepdims=True))


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
        w = v.get("words", {}) if v["type"] == "level" else {}
        slots.append(dict(type=SLOT_TYPES[v["type"]], min=int(lo), max=int(hi),
                          low=int(w.get("low", 0)) & 0xFF, medium=int(w.get("medium", 0)) & 0xFF, high=int(w.get("high", 0)) & 0xFF))
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
    w, wsens, _ = wake_template(wake)
    h = dict(host=HOSTS[out.get("host", "none")], i2c_addr=int(str(out.get("i2c_address", "0")), 0) & 0x7F,
             flags=flags | (1 if out.get("irq_pin") else 0), wake_sens=wsens)
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
    """The SDK's teacher embedding, deterministic (Nov 17): the same phrase gives the same bits in any call, any process
    and whatever the caller's batching, so the same config always compiles to the same blob bytes. Each text runs alone
    (a batch of one: no padding and no batch-composition effects on the kernels), with deterministic algorithms and TF32
    off. bf16 weights for teachers over 2 GB (teacher-v4: 8 GB) on the GPU when present. The caller's batch size
    (MARIGOLD_EMBED_BATCH, default 64) only groups calls; sdk/tests/test_determinism.py checks byte identity across
    processes and batch sizes."""
    def __init__(self, path):
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS (before the first CUDA call)
        import torch
        from sentence_transformers import SentenceTransformer
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        big = sum(f.stat().st_size for f in Path(path).glob("*.safetensors")) > 2e9
        self.m = SentenceTransformer(str(path), device="cuda" if torch.cuda.is_available() else "cpu",
                                     model_kwargs={"dtype": torch.bfloat16} if big else {})
        self.path = Path(path)
        self.dim = self.m.get_sentence_embedding_dimension()
        self.batch = int(os.environ.get("MARIGOLD_EMBED_BATCH", "64"))

    def encode(self, texts):
        texts = list(texts)
        out = []
        for i in range(0, len(texts), self.batch):            # grouping only; every text is its own forward pass
            out += [self.m.encode([t], batch_size=1, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
                    for t in texts[i:i + self.batch]]
        return np.stack(out) if out else np.zeros((0, self.dim), np.float32)


def default_embedder(tag=CM.SHIPPING):
    """The teacher of chip model `tag` (sdk/chip_models.py), data/models/<teacher>, if present (review K1: one default,
    the model's own teacher; it no longer changes with whatever teacher directory exists)."""
    t = ROOT / "data/models" / CM.TEACHER[tag]
    return t if (t / "modules.json").exists() else None


newest_embedder = default_embedder           # old name (the slow tests): the shipping model's teacher


def model_ids(model_tag, embedder):
    """(tag, model id, teacher name, teacher id) for a blob compiled with this embedder for chip model model_tag. A real
    embedder must be that model's teacher (review K1); the stub has teacher id 0. model_tag None: a blob for no chip
    model (teacher research, sw/teacher4/text_gate.py): model id 0, which every chip refuses, and no pairing check."""
    path = getattr(embedder, "path", None)
    name = "stub" if path is None else Path(path).name
    if model_tag is None:
        return None, 0, name, CM.teacher_id(name)
    tag = model_tag
    if tag not in CM.TEACHER:
        raise ValueError(f"model {tag!r} is not in sdk/chip_models.py (models {', '.join(CM.TEACHER)})")
    if path is None:
        return tag, CM.model_id(tag), "stub", 0
    if name != CM.TEACHER[tag]:
        raise ValueError(f"the embedder is {name}, but model {tag} pairs with {CM.TEACHER[tag]} (sdk/chip_models.py): "
                         f"a blob built with another teacher matches garbage on that chip")
    return tag, CM.model_id(tag), name, CM.teacher_id(name)


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


def enroll_protos(b0, byact, qb=None, K_audio=32, K_text=0, K_bank=0, cal=False, text_reject=False):
    """SDK audio enrollment (the matching variants sw/intent/evaluate.py chooses on dev; one implementation for both).
    byact {action id: [int8 q of the config's phrases spoken by TTS train voices through the chip model]} (id 0 = the
    expansion non-commands), qb int8 q of a generic speech bank (LibriSpeech training utterances). Prototypes: K_audio
    k-means centroids per action of the enrolled q, plus K_text centroids of the text blob b0's vectors, 16 reject
    centroids of the enrolled non-commands, K_bank reject centroids of the bank; cal = cohort bias normalisation
    (each action prototype's mean score on the bank, centred on the median, subtracted as the vector's bias).
    Returns a parsed-blob dict with the new vectors (pack() it with the dev threshold and margin)."""
    D = b0["D"]
    Cs, va = [], []
    # review K3: the 32 KB budget holds on every path (the blob's fixed part from b0's own tables)
    fixed = HDR.size + len(b0["actions"]) * ACT.size + len(b0["slots"]) * SLOT.size + len(b0["gpio"]) * GPIO.size + len(b0["wake"]) + 4
    max_n = (MAX_BLOB - fixed) // (VEC.size + D)
    R0 = b0["vectors"][b0["vec_action"] == 0].astype(np.float64) if text_reject else np.zeros((0, D))
    n_act = sum(1 for x in byact if x != 0)
    room = max_n - len(R0) - (16 if 0 in byact else 0) - K_bank
    K_audio = max(1, min(K_audio, room // max(1, n_act) - K_text))
    if text_reject:      # iteration-2 reject fix: keep the text blob's action-0 prototypes (expansion + reject bank)
        Cs.append(unit(R0)); va += [0] * len(R0)
    for aid in sorted(byact):
        if aid == 0:
            continue
        X = unit(np.array(byact[aid], np.float64)); cent = kmeans(X, K_audio); Cs.append(cent); va += [aid] * len(cent)
        if K_text:
            T = b0["vectors"][b0["vec_action"] == aid].astype(np.float64)
            ct = kmeans(unit(T), K_text); Cs.append(ct); va += [aid] * len(ct)
    if 0 in byact:
        X = unit(np.array(byact[0], np.float64)); cent = kmeans(X, 16); Cs.append(cent); va += [0] * len(cent)
    if K_bank:
        cent = kmeans(unit(qb.astype(np.float64)), K_bank); Cs.append(cent); va += [0] * len(cent)
    Cm = np.concatenate(Cs)
    V = np.clip(np.round(golden_vec(Cm, 0.0, np.eye(D)) * 32.0), -127, 127).astype(np.int8)
    va = np.array(va, np.int32)
    bias = np.zeros(len(va), np.int32)
    if cal:
        mu = (qb.astype(np.int64) @ V.astype(np.int64).T).mean(0)
        act = va != 0
        bias[act] = np.clip(np.round(-(mu[act] - np.median(mu[act]))), -32767, 32767).astype(np.int32)
    if len(va) > max_n:
        raise ValueError(f"enrollment does not fit the blob: it needs {len(va)} vectors, the 32 KB budget holds {max_n}")
    bb = dict(b0); bb.update(vectors=V, vec_action=va, vec_flags=np.zeros(len(va), int), bias=bias, N=len(va))
    return bb


# ------------------------------------------------------------------ matching scores (one implementation: the SDK and sw/intent/evaluate.py)
def scores(b, Q):
    """Best score, its action, and the best score of a different action (minus infinity if none), per row."""
    S = Q.astype(np.int64) @ b["vectors"].astype(np.int64).T + b["bias"][None].astype(np.int64)
    va = np.asarray(b["vec_action"])
    bi = S.argmax(1); best = S[np.arange(len(S)), bi]; act = va[bi]
    other = np.full(len(S), np.iinfo(np.int64).min // 4)
    for x in np.unique(va):
        rows_ = act == x
        if rows_.any() and (va != x).any():
            other[rows_] = S[rows_][:, va != x].max(1)
    return best, act, other


def pure_action(b, Q):
    """Review S7: the best ACTION prototype, reject (action-0) prototypes masked: pure action confusion."""
    S = Q.astype(np.int64) @ b["vectors"].astype(np.int64).T + b["bias"][None].astype(np.int64)
    va = np.asarray(b["vec_action"])
    S[:, va == 0] = np.iinfo(np.int64).min // 4
    return va[S.argmax(1)]


def decide(b, Q, thr, margin=None):
    """Matching rule (config-blob.md, arch 8.2): returns (action or 0, best score)."""
    m = b["margin"] if margin is None else margin
    best, act, other = scores(b, Q)
    out = np.where((act == 0) | (best < thr) | (best - other < m), 0, act)
    return out, best


def pick_thr_margin(b, Qn, Qp, yp, rate=0.95):
    """For each margin in a grid, the lowest threshold rejecting >= rate of the non-commands Qn; keep the
    (threshold, margin) pair with the best command accuracy on Qp (labels yp). Returns (threshold, margin, accuracy)."""
    bn, an, on = scores(b, Qn); bp, ap_, op = scores(b, Qp)
    allowed = len(Qn) - int(np.ceil(rate * len(Qn)))
    best_cfg = None
    for m in [0, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 13000, 16000, 20000, 25000]:
        passing = np.sort(bn[(an != 0) & (bn - on >= m)])[::-1]
        thr = int(passing[allowed]) + 1 if len(passing) > allowed else -10 ** 9
        acc = float(((ap_ == yp) & (bp >= thr) & (bp - op >= m)).mean())
        if best_cfg is None or acc > best_cfg[2]:
            best_cfg = (thr, m, acc)
    return best_cfg


def thr_for_rate(b, Qn, m, rate):
    """The lowest threshold that rejects >= rate of the rows Qn at margin m (pick_thr_margin's rule, one margin)."""
    bn, an, on = scores(b, Qn)
    allowed = len(Qn) - int(np.ceil(rate * len(Qn)))
    passing = np.sort(bn[(an != 0) & (bn - on >= m)])[::-1]
    return int(passing[allowed]) + 1 if len(passing) > allowed else -10 ** 9


# ------------------------------------------------------------------ audio enrollment, the default (review K2, Nov 17, 2026)
DEFAULT_ENROLL = ROOT / "data/tts/jobs_enroll.jsonl"     # sw/intent/enroll_jobs.py, then the TTS build (train voices only)
SPEECH_BANK = ROOT / "sw/intent/speech_bank.txt"           # 1,500 frozen LibriSpeech training utterances (review S6)


def golden_path(tag):
    """The chip model's integer export: weights/ (tracked), else data/golden/."""
    return next((p for p in (ROOT / f"weights/mg-{tag}.npz", ROOT / f"data/golden/mg-{tag}.npz") if p.exists()), None)


def _q_worker(args):
    gpath, paths = args
    sys.path.insert(0, str(ROOT / "sw/golden"))
    import soundfile as sf
    import intent_np as S
    I = S.IntIntent(dict(np.load(gpath)))
    return [np.asarray(I.run(sf.read(str(p), dtype="int16")[0].astype(np.int64))[0], np.int8) for p in paths]


def cache_load(cache):
    """{path: q} from a chip_q cache file, read fully and CLOSED (Windows cannot replace a file that is still open)."""
    if not Path(cache).exists():
        return {}
    with np.load(cache) as z:
        return dict(zip(z["paths"].tolist(), np.array(z["q"])))


def cache_save(cache, have):
    """Write {path: q} atomically: a temporary file, then a replace."""
    cache = Path(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.stem + ".tmp.npz")
    np.savez(tmp, paths=np.array(list(have)), q=np.stack(list(have.values())))
    tmp.replace(cache)


def chip_q(gpath, paths, procs=12):
    """int8 q of each audio file through the chip model's integer golden (sw/golden/intent_np.py): what the chip computes.
    Results are cached per export in data/sdk/cache/ (keyed by a hash of the export and the file path), saved after
    every chunk of 600 clips so an interrupted run keeps its work."""
    import hashlib
    from multiprocessing import Pool
    h = hashlib.sha1(Path(gpath).read_bytes()).hexdigest()[:12]
    cache = ROOT / "data/sdk/cache" / f"q-{h}.npz"
    have = cache_load(cache)
    todo = [p for p in dict.fromkeys(str(p) for p in paths) if p not in have]
    if todo:
        print(f"chip model on {len(todo)} clips ({len(have)} cached)", file=sys.stderr, flush=True)
        with Pool(procs) as pool:
            for s in range(0, len(todo), 600):
                part = todo[s:s + 600]
                chunks = [c for c in (part[i::procs] for i in range(procs)) if c]
                for c, r in zip(chunks, pool.map(_q_worker, [(str(gpath), c) for c in chunks])):
                    have.update(zip(c, r))
                cache_save(cache, have)
                print(f"  {min(s + 600, len(todo))} of {len(todo)}", file=sys.stderr, flush=True)
    return np.stack([have[str(p)] for p in paths])


def enroll_audio(b0, c, clips, Q, qb, reject_rate, K_audio=32):
    """Review K2: action prototypes from TTS recordings of the config's phrases through the chip model ("audio", the
    variant sw/intent/evaluate.py chose on dev for model v3), and a reject threshold and margin calibrated on AUDIO
    scores. clips: enrollment rows of this product (action name, or None for a non-command; voice); Q: their int8 q;
    qb: q of the generic speech bank. Calibration leaves one voice out: prototypes from the other voices, the held-out
    voice's non-commands and the speech bank as negatives, reject_rate of EACH set rejected (the non-commands pick the
    threshold and margin, the bank can only raise the threshold), its commands as positives (pick_thr_margin).
    The blob gets prototypes from every voice and the median threshold and margin.
    Returns (parsed-blob dict, report)."""
    name2id = {a["name"]: a["id"] for a in c["actions"]}
    unknown = sorted({r["action"] for r in clips if r["action"] and r["action"] not in name2id})
    if unknown:
        raise ValueError(f"enrollment clips name actions the config does not have: {unknown}")
    y = np.array([name2id[r["action"]] if r["action"] else 0 for r in clips])
    voices = sorted({r["voice"] for r in clips})
    if len(voices) < 2:
        raise ValueError(f"audio calibration needs clips from 2 or more voices, found {voices}")

    def protos(ix):
        byact = {}
        for i in ix:
            byact.setdefault(int(y[i]), []).append(Q[i])
        return enroll_protos(b0, byact, qb, K_audio, 0)

    held = []
    for v in voices:
        tr = [i for i, r in enumerate(clips) if r["voice"] != v]
        ho = np.array([i for i, r in enumerate(clips) if r["voice"] == v])
        bv = protos(tr)
        neg, pos = ho[y[ho] == 0], ho[y[ho] != 0]
        # each negative set on its own (John's first recording, Nov 17: pooled with the 1,500-line bank, the product's
        # own near-miss non-commands set almost nothing and real chatter passed): the product's non-commands pick the
        # threshold and margin, then the bank may only raise the threshold
        thr, m, _ = pick_thr_margin(bv, Q[neg] if len(neg) else qb, Q[pos], y[pos], reject_rate)
        if len(neg) and len(qb):
            thr = max(thr, thr_for_rate(bv, qb, m, reject_rate))
        held.append(dict(voice=v, reject_thr=int(thr), margin=int(m),
                         commands=int(len(pos)), commands_right=float((decide(bv, Q[pos], thr, m)[0] == y[pos]).mean()),
                         commands_right_rejection_off=float((pure_action(bv, Q[pos]) == y[pos]).mean()),
                         noncommands=int(len(neg)),
                         noncommands_rejected=float((decide(bv, Q[neg], thr, m)[0] == 0).mean()) if len(neg) else None,
                         bank_rejected=float((decide(bv, qb, thr, m)[0] == 0).mean()) if len(qb) else None))
    b = protos(range(len(clips)))
    b["reject_thr"] = int(np.median([h["reject_thr"] for h in held]))
    b["margin"] = int(np.median([h["margin"] for h in held]))
    rep = dict(enroll="audio", clips=len(clips), voices=voices, reject_rate=reject_rate, speech_bank=int(len(qb)),
               source=f"audio: leave one voice out; {reject_rate:.0%} of the held-out voice's non-commands rejected, and "
                      f"{reject_rate:.0%} of the speech bank, each set on its own; median over voices", held_out=held,
               note="held_out figures are on the SDK's own synthetic TTS voices, not a test; rejection_off = closest action "
                    "prototype (review S7)")
    return b, rep


def default_reject_rate(enroll):
    """--reject-rate when none is given. audio 0.95: the SDK's own TTS voices are easier to reject than real talk, and
    0.95 is the lowest of 0.80/0.85/0.90/0.95/0.98 whose eval DEV rejection meets plan target 4 (95%) for both reference
    configs (fan 95.5%, microwave 96.3%; at 0.90: 93.9% and 89.8%). text 0.90, chosen on the eval DEV split (Nov 15)."""
    return 0.95 if enroll == "audio" else 0.90


HDR_KEYS = ("D", "reject_thr", "margin", "wake_sens", "flags", "host", "i2c_addr", "conf_sh", "model_id", "teacher_id")


def repack(b):
    """A parsed (and edited) blob dict -> blob bytes, checked by the chip's own rules."""
    blob = pack({k: b[k] for k in HDR_KEYS}, b["vec_action"], b["vec_flags"], b["bias"], b["vectors"], b["actions"],
                b["slots"], b["gpio"], b["wake"])
    parse(blob)                     # 32 KB budget, sizes, CRC
    return blob


def audio_inputs(key, model_tag, manifest, golden):
    """What audio enrollment needs, checked before any model loads: (export path, clip rows, speech bank paths)."""
    gpath = Path(golden) if golden else golden_path(model_tag)
    if gpath is None or not gpath.exists():
        raise ValueError(f"no chip model export for {model_tag} (weights/mg-{model_tag}.npz or data/golden/mg-{model_tag}.npz); "
                         f"pass --golden FILE, or --enroll text")
    if not Path(manifest).exists():
        raise ValueError(f"enrollment manifest not found: {manifest} (sw/intent/enroll_jobs.py and the TTS build write it); "
                         f"or pass --enroll text")
    base = Path(manifest).parent
    rows = [json.loads(l) for l in open(manifest, encoding="utf-8")]
    clips = [dict(r, path=str(base / r["out"])) for r in rows if r.get("product") == key and (base / r["out"]).exists()]
    if not clips:
        raise ValueError(f"no enrollment clips for {key!r} in {manifest}: synthesize them (sw/intent/enroll_jobs.py, then the "
                         f"TTS build), or pass --enroll text")
    bank = [str(ROOT / l.strip()) for l in open(SPEECH_BANK, encoding="utf-8") if l.strip() and not l.startswith("#")]
    bank = [p for p in bank if Path(p).exists()]           # LibriSpeech under data/librispeech; without it, non-commands only
    if not bank:
        print("speech bank: no LibriSpeech files found (data/librispeech); calibrating on the enrolled non-commands only",
              file=sys.stderr)
    return gpath, clips, bank


def enroll_audio_blob(blob0, c, gpath, clips, bank, reject_rate, manifest=None):
    """The compile's text blob -> the audio-enrolled blob (review K2). Returns (blob, report)."""
    Q = chip_q(gpath, [r["path"] for r in clips] + bank).astype(np.int64)
    b, rep = enroll_audio(parse(blob0), c, clips, Q[:len(clips)], Q[len(clips):], reject_rate)
    rep.update(golden=str(gpath), manifest=str(manifest))
    return repack(b), rep


def compile_config(c, key, embedder, expansion=None, D=None, proj="random", protos=0, reject_protos=16,
                   q_scale=None, reject_rate=0.90, margin=0.0, reject_thr=None, reject_bank=None, bank_protos=0, bank_keep=0.8,
                   model_tag=CM.SHIPPING):
    """model_tag: the chip model the blob is for (sdk/chip_models.py; default the shipping model; None: no chip model,
    for teacher research only, see model_ids)."""
    validate(c)
    if reject_bank is not None:      # review K4: a bank needs prototypes, and must come from this embedder's teacher
        if bank_protos <= 0:
            raise ValueError("a reject bank was given with bank_protos = 0: pass --bank-protos N (N > 0)")
        files = getattr(reject_bank, "files", reject_bank)
        bt = str(reject_bank["teacher"]) if "teacher" in files else None
        et = Path(embedder.path).name if getattr(embedder, "path", None) else None
        if bt and et and bt != et:
            raise ValueError(f"reject bank built with teacher {bt!r}, but the embedder is {et!r}")
    tag, mid, teacher, tid = model_ids(model_tag, embedder)          # review K1: the blob records both
    D = GOLDEN_D if D is None else int(D)
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
    golden = proj == "teacher"
    if golden:                                 # golden-format section 5 (unit RMS, zero mean, D from the teacher)
        z = np.load(Path(embedder.path) / "marigold_proj.npz")
        P = z["P"]; D = int(P.shape[0])
        E = unit(golden_vec(E, z["mu"], P))
    else:
        P = projection(proj, E.shape[1], D)
        if P is None:
            D = E.shape[1]
        E = unit(E @ P) if P is not None else unit(E)
    if not 1 <= D <= 256:
        raise ValueError(f"D = {D}: the chip accepts D <= 256 (architecture 8.6); use --projection with --dim")
    Ep, En = E[:len(phrases)], E[len(phrases):]
    Rb = None
    if reject_bank is not None and bank_protos > 0:
        # iteration 2 reject fix: action-0 prototypes from a generic bank (commands of the 120 training appliances,
        # their non-commands and LibriSpeech snippets; data/sdk/reject_bank-<teacher>.npz, unit vectors in this teacher's
        # chip space). Bank lines that sit close to one of this product's own phrases (cosine >= bank_keep) are left out,
        # so "turn it on" from a lamp never becomes a reject prototype for a fan's "on".
        Vb = np.asarray(reject_bank["V"], np.float64)
        Vb = Vb[(Vb @ Ep.T).max(1) < bank_keep]
        Rb = kmeans(Vb, bank_protos) if len(Vb) else None
    fixed = HDR.size + len(actions) * ACT.size + len(slots) * SLOT.size + len(gpio) * GPIO.size + len(wake) + 4
    if protos <= 0:                            # auto: fill the 32 KB blob, at most 32 prototypes per action
        budget = (MAX_BLOB - fixed) // (VEC.size + D) - (reject_protos if len(En) else 0) - (len(Rb) if Rb is not None else 0)
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
        if Rb is not None:
            C.append(Rb)
            va += [0] * len(Rb)
        C = np.concatenate(C)
        size = fixed + len(C) * (VEC.size + D)
        if size <= MAX_BLOB or (protos == 1 and reject_protos <= 1):
            break
        protos, reject_protos = max(1, protos - 1), max(1, reject_protos - 1)
    if size > MAX_BLOB:
        raise ValueError(f"blob would be {size} bytes, over {MAX_BLOB}")
    if golden:                                 # centroids back on the golden rule; q and v at 5 fractional bits
        sv = sq = 32.0 * np.sqrt(D)
        V = np.clip(np.round(golden_vec(C, 0.0, np.eye(D)) * 32.0), -127, 127).astype(np.int8)
    else:
        sv = 127.0 / float(np.max(np.abs(C)))
        V = np.clip(np.round(C * sv), -127, 127).astype(np.int8)
        sq = float(q_scale) if q_scale else 127.0 / float(np.max(np.abs(E)))
    qi = lambda X: np.clip(np.round(X * sq), -127, 127).astype(np.int32)
    va = np.array(va, dtype=np.int32)
    Sp, Sn = qi(Ep) @ V.astype(np.int32).T, qi(En) @ V.astype(np.int32).T if len(En) else None
    m_int = int(round(margin * sq * sv))
    thr_cfg = c.get("match", {}).get("reject_threshold")
    calib = {}
    if reject_thr is not None:
        thr = int(reject_thr)
        calib["source"] = "--reject-thr (integer score units)"
    elif thr_cfg is not None:
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
    h.update(D=D, reject_thr=thr, margin=m_int, conf_sh=conf_shift(D), model_id=mid, teacher_id=tid)
    blob = pack(h, va, np.zeros(len(va), int), np.zeros(len(va), int), V, actions, slots, gpio, wake)
    b = parse(blob)
    pred = np.array([match(b, q) or 0 for q in qi(Ep)])
    calib.update(pos_phrases=len(phrases), pos_accept=float(np.mean(pred != 0)), pos_correct=float(np.mean(pred == labels)),
                 note="pos_* are on the SDK's own phrases (examples + expansion), not a test")
    if len(En):
        calib.update(neg_phrases=len(negs), neg_rejected=float(np.mean([match(b, q) is None for q in qi(En)])))
    report = dict(product=c.get("product"), key=key, model=tag, model_id=mid, teacher=teacher, teacher_id=tid, size_bytes=len(blob), crc32=f"{b['crc32']:08x}", D=D, N=b["N"],
                  protos_per_action=protos, reject_protos=int(np.sum(va == 0)), vector_scale=sv, q_scale=sq,
                  q_scale_source="golden-format 5 (32 per unit RMS)" if golden else "--q-scale" if q_scale else "127/max|calibration component|",
                  projection=proj if P is not None else "none", embedder_dim=int(E.shape[1]) if P is None else int(P.shape[1] if golden else P.shape[0]),
                  conf_sh=conf_shift(D),
                  reject_thr=thr, margin=m_int, calibration=calib,
                  value_words={n: v.get("words") for n, v in c.get("values", {}).items() if v.get("words")},
                  wake=wake_template(c.get("wake", {}))[2],
                  placeholders=[] if golden else ["projection", "q_scale"])
    return blob, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("-o", "--out")
    ap.add_argument("--parse")
    ap.add_argument("--embedder", default=None, help="the model's teacher (sentence-transformers dir; default data/models/<teacher "
                    "of --model-tag>), or 'stub' (tests; no chip loads such a blob)")
    ap.add_argument("--expansion", default=str(DEFAULT_EXPANSION), help="sdk/expand.py output, or 'none'")
    ap.add_argument("--dim", type=int, default=None, help="D without a teacher projection (default: the model's D)")
    ap.add_argument("--model-tag", default=CM.SHIPPING, help="the chip model the blob is for (sdk/chip_models.py; default the "
                    "shipping model): sets the teacher, the ids the blob records and D")
    ap.add_argument("--projection", default="auto", help="auto = the teacher's marigold_proj.npz when present (golden-format 5), else random")
    ap.add_argument("--reject-thr", type=int, default=None, help="absolute reject threshold in score units (overrides calibration)")
    ap.add_argument("--protos", type=int, default=0, help="prototypes per action; 0 = fill the 32 KB budget (max 32)")
    ap.add_argument("--reject-protos", type=int, default=16)
    ap.add_argument("--q-scale", type=float, default=None)
    ap.add_argument("--reject-rate", type=float, default=None,
                    help="share of the non-commands to reject; default 0.95 for --enroll audio (the lowest rate that meets plan "
                         "target 4, 95%% rejected, on the eval DEV split for both configs; review K2) and 0.90 for --enroll text")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--reject-bank", default=None, help="iteration 2: generic reject bank npz (sw/teacher4/make_reject_bank.py)")
    ap.add_argument("--bank-protos", type=int, default=0, help="action-0 prototypes taken from the reject bank")
    ap.add_argument("--bank-keep", type=float, default=0.8, help="bank lines closer than this to the product's own phrases are dropped")
    ap.add_argument("--enroll", default="audio", choices=("audio", "text"),
                    help="audio (default, review K2): prototypes and the reject threshold from TTS recordings of the config's "
                         "phrases through the chip model; text: from the teacher's text vectors (rejects most spoken commands)")
    ap.add_argument("--enroll-clips", default=str(DEFAULT_ENROLL), help="enrollment manifest (sw/intent/enroll_jobs.py + TTS build)")
    ap.add_argument("--golden", default=None, help="the chip model's integer export (default weights/mg-<tag>.npz, else data/golden/)")
    a = ap.parse_args()
    if a.reject_rate is None:
        a.reject_rate = default_reject_rate("text" if a.embedder == "stub" else a.enroll)   # the stub compiles text only
    if a.parse:
        b = parse(Path(a.parse).read_bytes())
        print(json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) and v.ndim == 1 else
                              f"int8[{v.shape[0]}x{v.shape[1]}]" if isinstance(v, np.ndarray) else
                              v.hex() if isinstance(v, bytes) else v) for k, v in b.items()}, indent=1))
        return
    cfg = Path(a.config)
    c = json.load(open(cfg, encoding="utf-8"))
    tag = a.model_tag                          # review K1: the model and its teacher, checked before any model loads
    if tag not in CM.TEACHER:
        sys.exit(f"--model-tag {tag}: not a chip model in sdk/chip_models.py (models {', '.join(CM.TEACHER)})")
    teacher = CM.TEACHER[tag]
    emb_path = None if a.embedder == "stub" else Path(a.embedder) if a.embedder else default_embedder(tag)
    if a.embedder != "stub" and emb_path is None:
        sys.exit(f"no embedder: data/models/{teacher} (model {tag}'s teacher) not found; pass --embedder DIR or --embedder stub")
    if emb_path is not None and emb_path.name != teacher:
        sys.exit(f"--embedder {emb_path} is not model {tag}'s teacher {teacher} (sdk/chip_models.py): a blob built with another "
                 f"teacher matches garbage on that chip; pass --model-tag for the model it belongs to")
    ain = None
    if a.enroll == "audio" and emb_path is not None:     # review K2: checked before the teacher loads
        try:
            ain = audio_inputs(cfg.stem, tag, a.enroll_clips, a.golden)
        except ValueError as e:
            sys.exit(f"audio enrollment: {e}")
    print(f"compiling for model {tag} (id {CM.model_id(tag)}) with " + (f"{teacher} (teacher id {CM.teacher_id(teacher)})"
          if emb_path else "the stub embedder (teacher id 0: tests only, no chip loads this blob)"), file=sys.stderr)
    emb = StubEmbedder() if a.embedder == "stub" else STEmbedder(emb_path)
    if a.expansion != "none" and not Path(a.expansion).exists():     # review K5: no silent fallback to typed examples only
        sys.exit(f"expansion file not found: {a.expansion} (run sdk/expand.py, which writes it, or pass --expansion none)")
    exp = json.load(open(a.expansion, encoding="utf-8")) if a.expansion != "none" else None
    if exp is None:     # say so rather than build from the typed examples silently (Mythos's check, Nov 17)
        print("no expansion: the prototypes come from the typed examples only (run sdk/expand.py, or pass --expansion FILE)",
              file=sys.stderr)
    if a.dim is None:                          # the model's D: weights/ (tracked), else data/golden/
        mj = next((p for p in (ROOT / f"weights/mg-{tag}.json", ROOT / f"data/golden/mg-{tag}.json") if p.exists()), None)
        a.dim = int(json.load(open(mj, encoding="utf-8"))["cfg"]["D"]) if mj else None
    proj = a.projection
    if proj == "auto":
        proj = "teacher" if emb_path and (Path(emb_path) / "marigold_proj.npz").exists() else "random"
    blob, rep = compile_config(c, cfg.stem, emb, exp, a.dim, proj, a.protos, a.reject_protos, a.q_scale, a.reject_rate, a.margin, a.reject_thr,
                               np.load(a.reject_bank) if a.reject_bank else None, a.bank_protos, a.bank_keep, model_tag=tag)
    rep.update(config=str(cfg), embedder=str(emb_path) if emb_path else "stub", expansion=a.expansion if exp else None)
    if a.enroll == "audio" and emb_path is None:
        print("stub embedder: text prototypes (tests only; no chip loads this blob)", file=sys.stderr)
        rep["enroll"] = "text (stub)"
    elif a.enroll == "audio":
        try:
            blob, arep = enroll_audio_blob(blob, c, *ain, a.reject_rate, manifest=a.enroll_clips)
        except ValueError as e:
            sys.exit(f"audio enrollment: {e}")
        b = parse(blob)
        if a.reject_thr is not None:            # an explicit threshold still wins
            b["reject_thr"] = int(a.reject_thr)
            blob = repack(b)
            arep["source"] = "--reject-thr (integer score units); prototypes from audio"
            b = parse(blob)
        rep.update(enroll="audio", text_calibration=rep["calibration"], calibration=arep, N=int(b["N"]), size_bytes=len(blob),
                   reject_thr=int(b["reject_thr"]), margin=int(b["margin"]), reject_protos=int(np.sum(b["vec_action"] == 0)),
                   crc32=f"{b['crc32']:08x}")
        for h_ in arep["held_out"]:
            pc = lambda x: "-" if x is None else format(x, ".0%")
            print(f"held-out voice {h_['voice']}: commands right {pc(h_['commands_right'])} (rejection off "
                  f"{pc(h_['commands_right_rejection_off'])}), non-commands rejected {pc(h_['noncommands_rejected'])}, "
                  f"speech bank rejected {pc(h_['bank_rejected'])}", file=sys.stderr)
    else:
        print("text enrollment: prototypes and threshold from text; on speech from a new voice most commands are rejected "
              "(review K2). A chip should use the default, --enroll audio", file=sys.stderr)
        rep["enroll"] = "text"
    out = Path(a.out) if a.out else ROOT / "data/sdk" / f"{cfg.stem}.mgc"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    json.dump(rep, open(f"{out}.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: rep[k] for k in ("product", "model", "teacher", "size_bytes", "D", "N", "reject_thr", "calibration")}, indent=1))


if __name__ == "__main__":
    main()
