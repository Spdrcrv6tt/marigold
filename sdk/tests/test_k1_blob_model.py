"""K1 (John's ruling, Nov 17, 2026; docs/decisions.md): a config blob records the model and the teacher it was built
for (blob version 1, header bytes 0x1D and 0x1E), and the chip refuses a blob made for another model (STATUS cfg
error 7), the way it refuses another D (error 6). One model-to-teacher table (sdk/chip_models.py) serves
sdk/compile.py and sw/intent/evaluate.py. Golden side; the RTL side is rtl/tb/test_cfg.py model_id_check (cfg
variant id34). Run: pytest sdk/tests"""
import json, struct, subprocess, sys, zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in ("sdk", "rtl/model"):
    sys.path.insert(0, str(ROOT / p))
import compile as C  # noqa: E402
import mg_model as M  # noqa: E402

FAN = json.load(open(ROOT / "sdk/configs/fan.json", encoding="utf-8"))


def rand_blob(D, model_id, teacher_id):
    """A small island blob (rtl/model/mg_model.py build_blob) stamped for a model and teacher."""
    cfg = dict(D=D, vectors=[(1, 0, 0, [1] * D), (0, 0, 0, [-1] * D)], actions=[(1, 0, 0, 0)], slots=[], gpio=[],
               wake=b"", host=1, i2c_addr=0x2A, reject_thr=0, margin=0, wake_sens=0, flags=0, conf_sh=9,
               model_id=model_id, teacher_id=teacher_id)
    return M.build_blob(cfg)


class NamedStub(C.StubEmbedder):
    """The stub embedder under a teacher's directory name (no model is loaded)."""
    def __init__(self, name):
        self.path = Path("data/models") / name


def ids(blob):
    return struct.unpack_from("<H", blob, 4)[0], blob[0x1D], blob[0x1E], blob[0x1F]


def test_blob_records_model_and_teacher():
    assert ids(C.compile_config(FAN, "fan", NamedStub("teacher-v4"), None)[0]) == (1, 3, 4, 0)       # shipping v3
    assert ids(C.compile_config(FAN, "fan", NamedStub("teacher-v3"), None, model_tag="v1")[0]) == (1, 1, 3, 0)
    b = C.parse(C.compile_config(FAN, "fan", NamedStub("teacher-v4"), None, model_tag="v2")[0])
    assert (b["model_id"], b["teacher_id"]) == (2, 4)
    assert ids(C.compile_config(FAN, "fan", C.StubEmbedder(), None)[0]) == (1, 3, 0, 0)     # the stub is no teacher
    # teacher research (sw/teacher4/text_gate.py): a candidate teacher, a blob for no chip model, which no chip loads
    assert ids(C.compile_config(FAN, "fan", NamedStub("candidate-x"), None, model_tag=None)[0]) == (1, 0, 0, 0)


def test_compile_refuses_an_embedder_from_another_teacher():
    for tag, name in (("v3", "teacher-v3"), ("v1", "teacher-v4"), ("v2", "some-gate0b-model")):
        with pytest.raises(ValueError, match="teacher"):
            C.compile_config(FAN, "fan", NamedStub(name), None, model_tag=tag)
    with pytest.raises(ValueError, match="v9"):
        C.compile_config(FAN, "fan", C.StubEmbedder(), None, model_tag="v9")


def run_compile(tmp_path, *extra):
    return subprocess.run([sys.executable, str(ROOT / "sdk/compile.py"), str(ROOT / "sdk/configs/fan.json"),
                           "--expansion", "none", "-o", str(tmp_path / "fan.mgc"), *extra], capture_output=True, text=True)


def test_compile_prints_model_and_teacher_and_refuses_a_mismatch(tmp_path):
    r = run_compile(tmp_path, "--embedder", "stub")
    assert r.returncode == 0, r.stderr[-400:]
    assert "model v3 (id 3)" in r.stdout + r.stderr and "teacher" in r.stdout + r.stderr
    assert ids((tmp_path / "fan.mgc").read_bytes())[1:3] == (3, 0)
    r = run_compile(tmp_path, "--embedder", "stub", "--model-tag", "v1")
    assert r.returncode == 0 and "model v1 (id 1)" in r.stdout + r.stderr, r.stderr[-400:]
    fake = tmp_path / "teacher-v3"                     # refused by name, before any model is loaded
    fake.mkdir()
    r = run_compile(tmp_path, "--embedder", str(fake))
    assert r.returncode != 0 and "teacher-v4" in r.stderr, r.stderr[-400:]


def test_island_refuses_version_0_and_a_blob_for_another_model():
    chip = dict(model_d=128, model_id=3, teacher_id=4)
    good = rand_blob(128, 3, 4)
    assert ids(good) == (1, 3, 4, 0)
    assert M.parse_blob(good, **chip)[0] == 0
    for m, t in ((2, 4), (3, 3), (0, 0), (1, 3)):
        b = rand_blob(128, m, t)
        assert M.parse_blob(b, **chip)[0] == 7, (m, t)
        assert M.parse_blob(b)[0] == 0                           # format checks alone: it is a valid blob
    assert M.parse_blob(rand_blob(64, 2, 4), **chip)[0] == 6     # the D check comes first
    v0 = bytearray(good)
    v0[4:6] = b"\x00\x00"
    v0[-4:] = struct.pack("<I", zlib.crc32(bytes(v0[:-4])) & 0xFFFFFFFF)
    assert M.parse_blob(bytes(v0))[0] == M.ERR_VERSION == 2
    with pytest.raises(ValueError, match="version"):
        C.parse(bytes(v0))


def test_one_model_teacher_table():
    import chip_models as CM
    assert CM.SHIPPING == "v3" and CM.TEACHER == {"v1": "teacher-v3", "v2": "teacher-v4", "v3": "teacher-v4"}
    assert [CM.ids(t) for t in ("v1", "v2", "v3", "randtest")] == [(1, 3), (2, 4), (3, 4), (0, 0)]
    assert CM.rom_ids() == CM.ids(CM.rom_tag()) == CM.ids(CM.SHIPPING)      # the tracked LUT ROM is the shipping model's
    sys.path.insert(0, str(ROOT / "sw/intent"))
    import evaluate as E
    assert E.default_teacher("v3") == "data/models/teacher-v4" and E.default_teacher("v1") == "data/models/teacher-v3"


def test_committed_test_blob_is_a_v3_blob():
    """rtl/top/tb/fan-i2c.mgc comes from data/sdk/fan-v3.mgc: sdk/configs/fan.json compiled for model v3 with
    teacher-v4 (text prototypes, --enroll text, the default before review K2). The shipping chip loads it as it is."""
    b = (ROOT / "rtl/top/tb/fan-i2c.mgc").read_bytes()
    err, cfg = M.parse_blob(b, model_d=128, model_id=3, teacher_id=4)
    assert err == 0 and (cfg["model_id"], cfg["teacher_id"]) == (3, 4)
    assert b[14] == 1 and b[15] == 0x2A and struct.unpack_from("<H", b, 26)[0] == 0xF
    src = ROOT / "data/sdk/fan-v3.mgc"                   # data/ is not in git: checked where it exists
    if src.exists():
        s = src.read_bytes()
        assert len(s) == len(b) and s[0x20:-4] == b[0x20:-4]           # vectors and tables identical
        out = subprocess.run([sys.executable, str(ROOT / "rtl/top/tb/mkblob_i2c.py"), str(src), "-"], capture_output=True)
        assert out.returncode == 0 and out.stdout == b, out.stderr[-300:]
