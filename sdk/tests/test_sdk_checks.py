"""SDK checks from the Nov 17, 2026 review (reports/review-2026-11-17.md, K3 to K6), written by Mythos (deoxylabs/marigold
PR 1) and kept as written, except the two expand.py tests: master dedupes paraphrases and takes any product config
inside expand_config (marked ADAPTED). Run: pytest sdk/tests"""
import copy, json, re, subprocess, sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sdk"))
import compile as C  # noqa: E402
import expand as X  # noqa: E402

FAN = json.load(open(ROOT / "sdk/configs/fan.json", encoding="utf-8"))


def fan_blob():
    return C.parse(C.compile_config(FAN, "fan", C.StubEmbedder(), None)[0])


def fixed_bytes(b):
    """Everything in a blob except its vectors (docs/config-blob.md)."""
    return (C.HDR.size + len(b["actions"]) * C.ACT.size + len(b["slots"]) * C.SLOT.size
            + len(b["gpio"]) * C.GPIO.size + len(b["wake"]) + 4)


def unit_rows(rng, n, d):
    v = rng.standard_normal((n, d))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ---- K6: the script runner
def test_script_mode_runs_every_test():
    src = (ROOT / "sdk/tests/test_compile.py").read_text(encoding="utf-8")
    defined = set(re.findall(r"^def (test_\w+)\(", src, re.M))
    out = subprocess.run([sys.executable, str(ROOT / "sdk/tests/test_compile.py")], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-400:]
    reported = set(re.findall(r"^(?:ok|skipped) (test_\w+)", out.stdout, re.M))
    assert reported == defined


# ---- K5: expansion file and paraphrases shared by two actions
def test_expand_writes_the_file_compile_reads():
    assert Path(X.DEFAULT_OUT) == Path(C.DEFAULT_EXPANSION)


def stub_llm(answers):     # ADAPTED: expand_config takes the LLM call as ask_fn
    """The answer for the action named in the prompt; the non-command prompts get small talk."""
    def ask(model, prompt):
        m = re.search(r"\(action '([^']+)'\)", prompt)
        return answers[m.group(1)] if m else "what a nice day it is"
    return ask


def config(product, names):
    return {"product": product, "actions": [{"name": n, "description": f"{n} the {product}", "examples": [f"{n} now please"],
                                              "values": []} for n in names]}


def test_expand_drops_a_paraphrase_given_for_two_actions():     # ADAPTED: no drop_shared(); expand_config dedupes
    ask = stub_llm({"on": "switch it on\nmake it go", "off": "switch it off\nmake it go", "faster": "speed up"})
    got, _ = X.expand_config(config("fan", ["on", "off", "faster"]), "stub", 5, ask_fn=ask, log=lambda *a: None)
    assert got == {"on": ["switch it on"], "off": ["switch it off"], "faster": ["speed up"]}


def test_expand_takes_any_product_config():     # ADAPTED: expand.py takes config files, not --products names
    ask = stub_llm({"purify": "clean the air", "stop": "stop cleaning"})
    got, _ = X.expand_config(config("air purifier", ["purify", "stop"]), "stub", 5, ask_fn=ask, log=lambda *a: None)
    assert got == {"purify": ["clean the air"], "stop": ["stop cleaning"]}


def run_compile(tmp_path, *extra):
    return subprocess.run([sys.executable, str(ROOT / "sdk/compile.py"), str(ROOT / "sdk/configs/fan.json"),
                           "--embedder", "stub", "-o", str(tmp_path / "fan.mgc"), *extra], capture_output=True, text=True)


def test_compile_stops_when_a_named_expansion_file_is_missing(tmp_path):
    out = run_compile(tmp_path, "--expansion", str(tmp_path / "missing.json"))
    assert out.returncode != 0
    assert "expansion file not found" in out.stderr


def test_compile_says_so_when_it_builds_without_an_expansion(tmp_path):
    out = run_compile(tmp_path, "--expansion", "none")
    assert out.returncode == 0, out.stderr[-400:]
    assert "no expansion" in out.stderr


# ---- K3: the 32 KB budget on the audio-enrollment path
def test_enroll_protos_fits_the_blob():
    b0 = fan_blob()
    rng = np.random.default_rng(1)
    byact = {a["id"]: list(unit_rows(rng, 40, b0["D"])) for a in FAN["actions"]}
    byact[0] = list(unit_rows(rng, 40, b0["D"]))
    bank = unit_rows(rng, 200, b0["D"])
    b = C.enroll_protos(b0, byact, bank, K_audio=32, K_text=0, K_bank=32)
    size = fixed_bytes(b0) + len(b["vec_action"]) * (C.VEC.size + b0["D"])
    assert size <= C.MAX_BLOB
    assert set(b["vec_action"].tolist()) == {0} | {a["id"] for a in FAN["actions"]}


def test_enroll_protos_refuses_a_bank_that_cannot_fit():
    b0 = fan_blob()
    rng = np.random.default_rng(2)
    byact = {a["id"]: list(unit_rows(rng, 8, b0["D"])) for a in FAN["actions"]}
    bank = unit_rows(rng, 400, b0["D"])
    with pytest.raises(ValueError, match="does not fit"):
        C.enroll_protos(b0, byact, bank, K_audio=32, K_text=0, K_bank=300)


# ---- K4: reject bank options
def test_reject_bank_without_bank_protos_is_an_error():
    bank = {"V": unit_rows(np.random.default_rng(3), 50, 256)}
    with pytest.raises(ValueError, match="bank-protos"):
        C.compile_config(copy.deepcopy(FAN), "fan", C.StubEmbedder(), None, reject_bank=bank, bank_protos=0)


def test_reject_bank_from_another_teacher_is_an_error():
    class Emb(C.StubEmbedder):
        path = Path("data/models/teacher-v3")

    bank = {"V": unit_rows(np.random.default_rng(4), 50, 256), "teacher": np.array("teacher-v4")}
    with pytest.raises(ValueError, match="teacher-v4"):
        C.compile_config(copy.deepcopy(FAN), "fan", Emb(), None, reject_bank=bank, bank_protos=8)
