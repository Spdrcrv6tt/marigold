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
sys.path.insert(0, str(ROOT / "sw/wake"))
import template as W  # noqa: E402

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
    assert (D, N) == (C.GOLDEN_D, (len(blob) - 0x20 - 4 * 4 - 2 * 12 - 0 - len(W.build("hey kitchen")[0]) - 4) // (4 + D))   # K7: D = 128
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
        tpl, sens, _ = W.build(c["wake"]["phrase"], c["wake"]["sensitivity"])
        assert b["wake"] == tpl and b["wake_sens"] == sens
        assert b["reject_thr"] == rep["reject_thr"] and b["margin"] == rep["margin"]
        # pack(parse(blob)) reproduces the blob byte for byte
        h = {k: b[k] for k in ("D", "reject_thr", "margin", "wake_sens", "flags", "host", "i2c_addr", "conf_sh")}
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
        print("SKIP test_real_embedder (needs MARIGOLD_SLOW=1 and a teacher in data/models or data/gate0b/models)")
        return
    emb = C.STEmbedder(C.newest_embedder())
    for name in ("fan", "microwave"):
        blob, rep = C.compile_config(CFG[name], name, emb, EXPANSION)
        assert rep["calibration"]["pos_correct"] > 0.8, rep["calibration"]




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


def test_wake_template():
    """Wake template bytes (docs/golden-wake.md): format 1, variants of CMUdict phones, parse round trip."""
    tpl, sens, rep = W.build("hey breeze", 0.5)
    P = {p: i + 1 for i, p in enumerate(W.PHONES)}
    assert tpl == bytes([1, 1, W.CLIP, W.REFRACTORY, 6] + [P[x] for x in "HH EY B R IY Z".split()])
    assert sens == W.sens_to_u16(0.5) and W.SENS_LO <= sens <= W.SENS_HI
    assert W.parse(tpl)["variants"] == [[P[x] for x in "HH EY B R IY Z".split()]]
    t2, _, _ = W.build("hello lumen", 1.0)                      # two pronunciations of "hello"
    assert t2[1] == 2 and len(W.parse(t2)["variants"]) == 2 and W.sens_to_u16(1.0) == W.SENS_HI
    _, _, r3 = W.build("hey zorblat")                           # unknown word: fallback with a warning
    assert any("zorblat" in n for n in r3["notes"])
    t4, _, _ = W.build("anything", phones=["HH EY K IH CH AH N"])
    assert t4[4:] == bytes([7] + [P[x] for x in "HH EY K IH CH AH N".split()])


def test_conf_sh_and_level_words():
    b = C.parse(build("microwave")[0])
    assert b["conf_sh"] == C.conf_shift(b["D"]) == C.conf_shift(C.GOLDEN_D) == 9   # D = 128 (golden format v1.0)
    power = b["slots"][1]
    assert (power["type"], power["low"], power["medium"], power["high"]) == (2, 3, 5, 10)
    assert (b["slots"][0]["low"], b["slots"][0]["medium"], b["slots"][0]["high"]) == (0, 0, 0)
    s_best = int((b["vectors"][0].astype(np.int32) ** 2).sum())
    assert C.confidence(b, s_best) == min(255, s_best >> b["conf_sh"])


# ---------------------------------------------------------------- review fixes K3, K4, K5 (Nov 17, 2026)
def seven_action_config():
    import copy
    c = copy.deepcopy(CFG["fan"])
    c["actions"].append(dict(id=max(a["id"] for a in c["actions"]) + 1, name="turbo", description="Run the fan at full blast",
                             values=[], examples=["turbo", "full blast", "max power now", "blast it", "turbo mode"]))
    return c


def test_enroll_budget_every_path():
    """K3: 7 actions x 32 audio prototypes + 16 + 32 bank prototypes used to pack 35.9 KB; every path now fits."""
    c = seven_action_config()
    b0 = C.parse(C.compile_config(c, "fan7", C.StubEmbedder(), None)[0])
    rng = np.random.default_rng(0)
    q = lambda n: rng.integers(-60, 60, (n, b0["D"])).astype(np.int8)
    byact = {a["id"]: list(q(40)) for a in c["actions"]}
    byact[0] = list(q(40))
    for text_reject in (False, True):
        bb = C.enroll_protos(b0, byact, q(300), K_audio=32, K_text=0, K_bank=32, cal=True, text_reject=text_reject)
        blob = C.pack({k: bb[k] for k in ("D", "reject_thr", "margin", "wake_sens", "flags", "host", "i2c_addr", "conf_sh")},
                      bb["vec_action"], bb["vec_flags"], bb["bias"], bb["vectors"], bb["actions"], bb["slots"], bb["gpio"], bb["wake"])
        assert len(blob) <= C.MAX_BLOB and C.parse(blob)["N"] == bb["N"], (text_reject, len(blob))


class NamedStub(C.StubEmbedder):
    def __init__(self, name):
        self.path = Path("data/models") / name


def test_bank_needs_protos_and_matching_teacher():
    """K4: a bank with 0 prototypes is an error, and a bank from another teacher is refused."""
    V = C.unit(np.random.default_rng(1).standard_normal((50, C.GOLDEN_D)))
    bank = {"V": V, "teacher": np.array("teacher-v4")}
    for kw, emb, msg in ((dict(bank_protos=0), NamedStub("teacher-v4"), "bank_protos = 0"),
                         (dict(bank_protos=4), NamedStub("teacher-v3"), "teacher")):
        try:
            C.compile_config(CFG["fan"], "fan", emb, None, reject_bank=bank, **kw)
        except ValueError as e:
            assert msg in str(e), e
        else:
            raise AssertionError(f"accepted: {kw}")
    blob, rep = C.compile_config(CFG["fan"], "fan", NamedStub("teacher-v4"), None, reject_bank=bank, bank_protos=4)
    assert rep["reject_protos"] >= 4


def test_expand_any_product_and_cross_action_dedupe():
    """K5: expand.py works for any product, drops a line written under two actions or equal to another action's example."""
    sys.path.insert(0, str(ROOT / "sdk"))
    import expand as X
    c = {"product": "Desk lamp (test)", "actions": [
        dict(id=1, name="on", description="Turn the lamp on", values=[], examples=["lamp on", "light on", "on", "switch on", "lights up"]),
        dict(id=2, name="off", description="Turn the lamp off", values=[], examples=["lamp off", "light off", "off", "switch off", "lights out"])]}

    def stub(model, prompt):
        if "action 'on'" in prompt:
            return "turn the lamp on\nlet there be light\nhit the switch\nlamp off"
        if "action 'off'" in prompt:
            return "turn the lamp off\nkill the light\nhit the switch"
        return "nice weather today\nturn the lamp on\nwhat's for dinner"
    acts, neg = X.expand_config(c, "stub", 3, ask_fn=stub, log=lambda s: None)
    assert acts["on"] == ["let there be light", "turn the lamp on"], acts       # "hit the switch" (both) and "lamp off" (off's example) dropped
    assert acts["off"] == ["kill the light", "turn the lamp off"], acts
    assert neg == ["nice weather today", "what's for dinner"], neg               # a command is never a negative
    assert X.DEFAULT_OUT.name == "expanded_qwen25.json"                          # the file compile.py reads by default


def test_cli_missing_expansion_is_an_error():
    """K5: the compiler no longer falls back silently to typed examples when the expansion file is missing."""
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run([sys.executable, str(ROOT / "sdk/compile.py"), str(ROOT / "sdk/configs/fan.json"), "--embedder", "stub",
                            "--expansion", str(Path(d) / "missing.json"), "-o", str(Path(d) / "f.mgc")], capture_output=True, text=True)
        assert r.returncode != 0 and "not found" in (r.stderr + r.stdout), r
        r = subprocess.run([sys.executable, str(ROOT / "sdk/compile.py"), str(ROOT / "sdk/configs/fan.json"), "--embedder", "stub",
                            "--expansion", "none", "-o", str(Path(d) / "f.mgc")], capture_output=True, text=True)
        assert r.returncode == 0 and C.parse((Path(d) / "f.mgc").read_bytes())["D"] == C.GOLDEN_D, r


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
