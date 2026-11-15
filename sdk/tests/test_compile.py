"""Tests for sdk/compile.py: exact blob layout, round trip, CRC, matching rule, config rules.
Run: .venv/Scripts/python.exe -m pytest -q sdk/tests   (or python sdk/tests/test_compile.py)
The real-embedder test runs only when MARIGOLD_SLOW=1 and a model exists in data/gate0b/models.
"""
import json, os, struct, sys, zlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sdk"))
import compile as C  # noqa: E402

CFG = {p: json.load(open(ROOT / f"sdk/configs/{p}.json", encoding="utf-8")) for p in ("fan", "microwave")}
EXP = ROOT / "data/expand/expanded_qwen25.json"
EXPANSION = json.load(open(EXP, encoding="utf-8")) if EXP.exists() else None


def build(name, **kw):
    return C.compile_config(CFG[name], name, C.StubEmbedder(), EXPANSION, **kw)


def test_header_offsets_match_spec():
    assert C.HDR.size == 0x20 and C.VEC.size == 4 and C.ACT.size == 4 and C.SLOT.size == 12 and C.GPIO.size == 8
    blob, _ = build("microwave")
    assert blob[:4] == b"MGC0"
    D, N = struct.unpack_from("<HH", blob, 0x06)
    assert (D, N) == (256, (len(blob) - 0x20 - 4 * 4 - 2 * 12 - 0 - len(b"\x00hey kitchen") - 4) // (4 + D))
    assert blob[0x0E] == 1 and blob[0x0F] == 0x2A                # host I2C, address 0x2A
    assert struct.unpack_from("<H", blob, 0x1A)[0] == 0b1111     # IRQ pin + WAKE, NOMATCH, TIMEOUT events to the host
    assert struct.unpack_from("<I", blob, len(blob) - 4)[0] == zlib.crc32(blob[:-4]) & 0xFFFFFFFF


def test_round_trip_fan_and_microwave():
    for name in ("fan", "microwave"):
        blob, rep = build(name)
        b = C.parse(blob)
        c = CFG[name]
        assert b["size"] == len(blob) <= C.MAX_BLOB and rep["size_bytes"] == len(blob)
        assert [a["id"] for a in b["actions"] if a["id"] < 0xF0] == [a["id"] for a in c["actions"]]
        assert set(b["vec_action"].tolist()) - {0} == {a["id"] for a in c["actions"]}
        names = list(c["values"])
        for s, n in zip(b["slots"], names):
            v = c["values"][n]
            assert s["type"] == C.SLOT_TYPES[v["type"]]
            assert (s["min"], s["max"]) == ((v["min_s"], v["max_s"]) if v["type"] == "duration" else (v["min"], v["max"]))
        for a, ca in zip(b["actions"], c["actions"]):
            assert a["slot_mask"] == sum(1 << names.index(s) for s in ca["values"])
        assert b["wake"] == b"\x00" + c["wake"]["phrase"].encode()
        assert b["reject_thr"] == rep["reject_thr"] and b["margin"] == rep["margin"]
        # pack(parse(blob)) reproduces the blob byte for byte
        h = {k: b[k] for k in ("D", "reject_thr", "margin", "wake_sens", "flags", "host", "i2c_addr")}
        assert C.pack(h, b["vec_action"], b["vec_flags"], b["bias"], b["vectors"], b["actions"], b["slots"], b["gpio"], b["wake"]) == blob


def test_fan_gpio_rules():
    b = C.parse(build("fan")[0])
    assert b["host"] == 0 and b["flags"] == 0 and len(b["gpio"]) == 7      # 6 action rules + the NOMATCH buzzer
    by_id = {a["id"]: a for a in b["actions"]}
    on = b["gpio"][by_id[1]["gpio_first"]]
    assert (on["pin_mask"], on["mode"], on["level"]) == (0b1, 1, 1)
    faster = b["gpio"][by_id[3]["gpio_first"]]
    assert (faster["pin_mask"], faster["mode"], faster["pulse_ms"]) == (0b10, 0, 100)
    lvl = b["gpio"][by_id[5]["gpio_first"]]
    assert (lvl["pin_mask"], lvl["mode"], lvl["slot"]) == (0b11100000, 3, 0)
    osc = b["gpio"][by_id[6]["gpio_first"]]
    assert (osc["pin_mask"], osc["mode"]) == (0b1000, 2)


def test_crc_detects_corruption():
    blob = bytearray(build("fan")[0])
    blob[100] ^= 0x01
    try:
        C.parse(bytes(blob))
    except ValueError as e:
        assert "CRC" in str(e)
    else:
        raise AssertionError("corruption not detected")


def hand_blob(thr=-10**9, margin=0):
    V = np.array([[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10], [10, 0, 0, 0]], dtype=np.int8)
    va = [1, 1, 2, 0, 2]                       # vector 4 duplicates vector 0 under another action (tie test)
    h = dict(D=4, reject_thr=thr, margin=margin, wake_sens=0, flags=0, host=0, i2c_addr=0)
    acts = [dict(id=1, slot_mask=0, gpio_first=0, gpio_count=0), dict(id=2, slot_mask=0, gpio_first=0, gpio_count=0)]
    return C.parse(C.pack(h, va, [0] * 5, [0] * 5, V, acts, [], [], b""))


def test_matching_rule():
    b = hand_blob()
    assert C.match(b, [0, 5, 0, 0]) == 1                  # nearest vector's action
    assert C.match(b, [5, 0, 0, 0]) == 1                  # tie between vector 0 (action 1) and 4 (action 2): lowest index
    assert C.match(b, [0, 0, 0, 5]) is None               # reject class
    assert C.match(hand_blob(thr=60), [0, 5, 0, 0]) is None   # score 50 below threshold
    assert C.match(hand_blob(thr=50), [0, 5, 0, 0]) == 1      # equal to threshold passes
    assert C.match(hand_blob(margin=1), [0, 5, 5, 0]) is None # best 50 vs other action 50: margin 0 < 1
    assert C.match(hand_blob(margin=1), [5, 0, 0, 0]) is None # tie across actions also fails a positive margin
    b = hand_blob()
    b["bias"] = np.array([0, 0, 1, 0, 0], dtype=np.int32)
    assert C.match(b, [0, 5, 5, 0]) == 2                  # bias breaks the tie


def test_config_rules():
    import copy
    c = copy.deepcopy(CFG["fan"])
    c["actions"][1]["examples"].append(c["actions"][0]["examples"][0])
    for bad, msg in [(c, "appears under")]:
        try:
            C.validate(bad)
        except ValueError as e:
            assert msg in str(e)
        else:
            raise AssertionError(msg)
    c = copy.deepcopy(CFG["microwave"])
    c["actions"][0]["examples"] = c["actions"][0]["examples"][:4]
    try:
        C.validate(c)
    except ValueError as e:
        assert "at least 5" in str(e)
    else:
        raise AssertionError("short example list accepted")


def test_real_embedder():
    if os.environ.get("MARIGOLD_SLOW") != "1" or C.newest_embedder() is None:
        return
    emb = C.STEmbedder(C.newest_embedder())
    for name in ("fan", "microwave"):
        blob, rep = C.compile_config(CFG[name], name, emb, EXPANSION)
        assert rep["calibration"]["pos_correct"] > 0.8, rep["calibration"]


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)


def test_events():
    b = C.parse(build("fan")[0])
    ev = [a for a in b["actions"] if a["id"] >= 0xF0]
    assert [a["id"] for a in ev] == [0xF1] and ev[0]["gpio_count"] == 1
    g = b["gpio"][ev[0]["gpio_first"]]
    assert (g["pin_mask"], g["mode"], g["pulse_ms"]) == (0b10000, 0, 150)
    assert 0xF1 not in b["vec_action"].tolist() and b["flags"] == 0
    m = C.parse(build("microwave")[0])
    assert m["flags"] == 0b1111                               # IRQ + WAKE + NOMATCH + TIMEOUT to the host
    assert all(a["id"] < 0xF0 for a in m["actions"])          # host-only events need no action-table entry
    import copy
    c = copy.deepcopy(CFG["fan"])
    c["events"] = {"beep": {"host": True}}
    try:
        C.validate(c)
    except ValueError as e:
        assert "wake, nomatch and timeout" in str(e)
    else:
        raise AssertionError("unknown event accepted")
    c = copy.deepcopy(CFG["fan"])
    c["actions"][0]["id"] = 0xF1
    try:
        C.validate(c)
    except ValueError as e:
        assert "reserved" in str(e)
    else:
        raise AssertionError("reserved action id accepted")


def test_pulse_level_and_crc8():
    b = C.parse(build("fan")[0])
    by_id = {a["id"]: a for a in b["actions"]}
    assert b["gpio"][by_id[3]["gpio_first"]]["level"] == 1          # pulse defaults to active high (arch 8.5)
    assert b["gpio"][by_id[0xF1]["gpio_first"]]["level"] == 1
    assert C.crc8(b"123456789") == 0xF4                              # CRC-8/SMBUS check value
    f = C.host_frame(0, 5, [(0, 3)], 200)
    assert f[:4] == bytes([0xA5, 0, 5, 1]) and f[-1] == C.crc8(f[:-1]) and len(f) == 4 + 5 + 2


def test_chip_config_checks():
    blob = bytearray(build("microwave")[0])
    struct.pack_into("<H", blob, 0x06, 300)                        # D > 256
    blob[-4:] = struct.pack("<I", zlib.crc32(bytes(blob[:-4])) & 0xFFFFFFFF)
    try:
        C.parse(bytes(blob))
    except ValueError as e:
        assert "config check" in str(e)
    else:
        raise AssertionError("D > 256 accepted")
