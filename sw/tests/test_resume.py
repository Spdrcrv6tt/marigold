"""Mythos's S1/S2 tests (deoxylabs/marigold PR 1, Nov 17, 2026), adapted: master has no resume.py: the same checks run against
train.py's own helpers (thr_init, ckpt_dict) and code. Master keeps no tensor names in the checkpoint (it reads them
from Moonshine on every --moonshine start), so the three tests of Mythos's pretrained_names() map to one code check
here and to the GPU test. Run: pytest sw/tests_port"""
import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sw" / "intent"))
import train as T  # noqa: E402

TRAIN = ROOT / "sw/intent/train.py"
TREE = ast.parse(TRAIN.read_text(encoding="utf-8"))


def calls(name):
    return [n for n in ast.walk(TREE) if isinstance(n, ast.Call) and ast.unparse(n.func) == name]


# ---- S1: the learned reject threshold
def test_checkpoint_carries_the_threshold():
    ck = T.ckpt_dict({"w": 1}, {"D": 128}, 5000, False, 0.31, 0.5)
    assert ck["thr"] == 0.31
    assert (ck["state"], ck["cfg"], ck["step"], ck["qat"]) == ({"w": 1}, {"D": 128}, 5000, False)


def test_every_save_in_train_writes_a_full_checkpoint():
    saves = calls("torch.save")
    assert len(saves) >= 2
    for call in saves:
        assert isinstance(call.args[0], ast.Call) and ast.unparse(call.args[0].func) == "ckpt_dict", ast.unparse(call)
        assert "thr_p" in ast.unparse(call.args[0].args[4]), ast.unparse(call)


def test_a_resume_starts_from_the_saved_threshold():
    assert T.thr_init({"thr": 0.31}, 0.45) == 0.31


def test_a_fresh_or_old_checkpoint_starts_from_thr0():
    assert T.thr_init(None, 0.45) == 0.45
    assert T.thr_init({"thr": None}, 0.45) == 0.45
    assert T.thr_init({"state": {}}, 0.45) == 0.45


def test_train_takes_its_threshold_from_the_checkpoint():
    assert calls("thr_init")


def test_a_threshold_outside_its_range_gets_a_gradient_again():
    torch = pytest.importorskip("torch")
    thr_p = torch.nn.Parameter(torch.tensor(1.5))
    thr_p.clamp(0.0, 0.9).backward()
    assert thr_p.grad.item() == 0.0                 # a clamp in the forward pass: stuck once outside the range
    with torch.no_grad():                           # master's clamp (train.py, after every optimiser step)
        thr_p.clamp_(0.0, 0.9)
    assert thr_p.item() == pytest.approx(0.9)
    thr_p.grad = None
    (thr_p * 1.0).backward()
    assert thr_p.grad.item() == 1.0


def test_train_clamps_the_parameter_and_not_the_forward_value():
    assert len(calls("thr_p.clamp_")) >= 2, "Mythos clamps when the parameter is created and after every step"
    assert not calls("thr_p.clamp")


# ---- S2: the Moonshine tensor names whether or not --init is set (the review's own fix text)
def test_train_reads_the_pretrained_names_whether_or_not_init_is_set():
    loads = calls("load_moonshine")
    assert loads
    guards = [n for n in ast.walk(TREE) if isinstance(n, ast.If) and any(c in loads for b in n.body for c in ast.walk(b))]
    assert guards and all("a.init" not in ast.unparse(g.test) for g in guards), [ast.unparse(g.test) for g in guards]
