"""R16 (John's ruling, Nov 17, 2026; docs/decisions.md): a value nobody said must not look like the slot's minimum.
The value head marks "no value" with the reserved raw code INT32_MIN, and the island leaves that slot out of the
COMMAND frame (n_values counts only the values that were said) and out of the GPIO binary rules. Golden-model side:
sw/intent/values.py and rtl/model/mg_model.py; the RTL side is rtl/tb/test_cfg.py value_not_said and
rtl/tb/test_ctl.py value_not_said_frame. Run: pytest sdk/tests"""
import json, struct, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in ("sdk", "rtl/model", "sw/intent"):
    sys.path.insert(0, str(ROOT / p))
import compile as C  # noqa: E402
import mg_model as M  # noqa: E402
import values as VAL  # noqa: E402

NO_VALUE = -(1 << 31)          # docs/golden-format.md section 6: the raw code for "no value heard"
MW = json.load(open(ROOT / "sdk/configs/microwave.json", encoding="utf-8"))
FAN = json.load(open(ROOT / "sdk/configs/fan.json", encoding="utf-8"))


def island(config, key):
    """The product's action, slot and GPIO tables as the SDK compiles them, with one one-hot vector per action so a
    chosen q picks the action. Returns (parsed island config, {action id: q})."""
    b = C.parse(C.compile_config(config, key, C.StubEmbedder(), None)[0])
    ids = [a["id"] for a in b["actions"] if a["id"] < 0xF0]
    hot = {i: [int(j == k) for j in range(len(ids))] for k, i in enumerate(ids)}
    cfg = dict(D=len(ids), vectors=[(i, 0, 0, hot[i]) for i in ids],
               actions=[(a["id"], a["slot_mask"], a["gpio_first"], a["gpio_count"]) for a in b["actions"]],
               slots=[(s["type"], s["min"], s["max"], (s["low"], s["medium"], s["high"])) for s in b["slots"]],
               gpio=[(g["pin_mask"], g["mode"], g["level"], g["slot"], g["pulse_ms"]) for g in b["gpio"]],
               wake=b"", host=1, i2c_addr=0x2A, reject_thr=-(1 << 31), margin=0, wake_sens=0, flags=0xF, conf_sh=0)
    err, pc = M.parse_blob(M.build_blob(cfg))
    assert err == 0
    return pc, hot


def slot_specs(pc):
    return [dict(type=t, min=lo, max=hi, words={w: x for w, x in zip(("low", "medium", "high"), lv) if x})
            for t, lo, hi, lv in pc["slots"]]


def test_value_head_marks_no_value():
    pc, _ = island(MW, "microwave")
    sl = slot_specs(pc)                                   # 0 time (duration 1..5999 s), 1 power (level 1..10)
    for heard in ([], VAL.targets("power level"), VAL.targets("start the microwave")):
        assert VAL.raw_values(heard, sl) == [NO_VALUE, NO_VALUE], heard
    assert VAL.raw_values(VAL.targets("power level five"), sl)[1] == 5
    assert VAL.raw_values(VAL.targets("set it to zero"), [dict(type=0, min=-5, max=5)]) == [0]   # a said 0 stays 0
    assert VAL.parse([], sl) == [None, None]              # the host view: nothing to report


def test_review_case_unsaid_value_is_left_out_of_the_frame():
    """The review's case: "power level five" with the number missed arrived as power_level with power = 1, and a
    "start" with no time as start, 1 s."""
    pc, hot = island(MW, "microwave")
    none = [NO_VALUE] * 8
    r = M.match(pc, hot[4], none)                         # power_level, number missed
    assert r["ev"] == M.EV_COMMAND and r["action"] == 4
    assert r["values"] == [] and r["slot_mask"] == 0, r["values"]
    assert M.frame(r["ev"], r["action"], r["values"], r["conf"])[:4] == bytes([0xA5, 0, 4, 0])     # n_values 0
    r = M.match(pc, hot[1], none)                         # start, no time and no power said
    assert r["action"] == 1 and r["values"] == [] and r["slot_mask"] == 0, r["values"]


def test_said_values_still_go_out_and_are_clamped():
    pc, hot = island(MW, "microwave")
    r = M.match(pc, hot[1], [90, NO_VALUE] + [0] * 6)   # "start for 90 seconds": time only
    assert r["values"] == [(0, 90)] and r["slot_mask"] == 0b01
    r = M.match(pc, hot[4], [NO_VALUE, -3] + [0] * 6)   # "power high": the config's word value
    assert r["values"] == [(1, 10)]
    r = M.match(pc, hot[1], [0, 99] + [0] * 6)          # said values below min / above max are clamped, still sent
    assert r["values"] == [(0, 1), (1, 10)] and r["slot_mask"] == 0b11
    f = M.frame(r["ev"], r["action"], r["values"], r["conf"])
    assert f[3] == 2 and struct.unpack_from("<Bi", f, 4) == (0, 1)


def test_gpio_binary_rule_keeps_its_pins_for_an_unsaid_value():
    """The fan's set_level drives the speed on pins 5 to 7 in binary (GPIO mode 3, the one rule that uses values):
    with no level said the pins keep their state instead of showing the minimum."""
    pc, hot = island(FAN, "fan")
    gp = M.Gpio()
    r = M.match(pc, hot[5], [4] + [0] * 7)
    gp.apply(pc, r["action"], r["slot_mask"], r["vals8"])
    assert gp.pins == 4 << 5
    r = M.match(pc, hot[5], [NO_VALUE] + [0] * 7)
    gp.apply(pc, r["action"], r["slot_mask"], r["vals8"])
    assert r["values"] == [] and gp.pins == 4 << 5, (r["values"], bin(gp.pins))
