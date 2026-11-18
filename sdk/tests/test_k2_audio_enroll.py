"""Review K2 (Nov 17, 2026): audio enrollment is the SDK default. The prototypes and the reject threshold come from TTS
recordings of the config's phrases through the chip model, calibrated on audio scores with one voice held out at a
time; text enrollment stays available behind --enroll text, with a warning.
Run: .venv/Scripts/python.exe -m pytest -q sdk/tests/test_k2_audio_enroll.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sdk"))
import compile as C  # noqa: E402
import chip_models as CM  # noqa: E402

D = 16
FAN = ROOT / "sdk/configs/fan.json"


def text_blob():
    c = json.load(open(FAN, encoding="utf-8"))
    blob, _ = C.compile_config(c, "fan", C.StubEmbedder(), None, D=D, proj="random")
    return c, C.parse(blob)


def synthetic_clips(c, voices=("kokoro/a", "kokoro/b", "kokoro/c"), per=20, neg=30, seed=0):
    """Each action is a direction in q space; each voice a small shift of it; non-commands point anywhere."""
    rng = np.random.default_rng(seed)
    unit = lambda x: x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True))
    q = lambda x: np.clip(np.round(32 * unit(x)), -127, 127).astype(np.int64)
    dirs = {a["name"]: rng.standard_normal(D) for a in c["actions"]}
    clips, Q = [], []
    for v in voices:
        shift = 0.2 * rng.standard_normal(D)
        for name, u in dirs.items():
            for _ in range(per):
                clips.append(dict(action=name, voice=v)); Q.append(q(u + shift + 0.3 * rng.standard_normal(D)))
        for _ in range(neg):
            clips.append(dict(action=None, voice=v)); Q.append(q(rng.standard_normal(D)))
    qb = q(rng.standard_normal((100, D)))
    return clips, np.array(Q), qb


def test_audio_enrollment_calibrates_on_audio_with_a_voice_held_out():
    c, b0 = text_blob()
    clips, Q, qb = synthetic_clips(c)
    b, rep = C.enroll_audio(b0, c, clips, Q, qb, 0.90)
    held = rep["held_out"]
    assert rep["enroll"] == "audio" and [h["voice"] for h in held] == ["kokoro/a", "kokoro/b", "kokoro/c"]
    assert b["reject_thr"] == int(np.median([h["reject_thr"] for h in held]))
    assert b["margin"] == int(np.median([h["margin"] for h in held]))
    for h in held:                          # a new voice: its commands pass, its non-commands and the bank are rejected
        assert h["commands_right"] >= 0.9 and h["commands_right_rejection_off"] >= 0.9
        assert h["noncommands_rejected"] >= 0.8 and h["bank_rejected"] >= 0.8
    # the final blob on the enrolled audio itself: commands pass the audio threshold, non-commands do not
    cmd = np.array([r["action"] is not None for r in clips])
    pred = C.decide(b, Q, b["reject_thr"], b["margin"])[0]
    assert (pred[cmd] != 0).mean() >= 0.9 and (pred[~cmd] == 0).mean() >= 0.8
    blob = C.repack(b)                      # packs and passes the chip's checks; ids unchanged
    assert blob[0x1D] == b0["model_id"] and C.parse(blob)["N"] == len(b["vec_action"])
    assert set(np.unique(b["vec_action"])) == {0} | {a["id"] for a in c["actions"]}


def test_audio_enrollment_needs_two_voices_and_known_actions():
    c, b0 = text_blob()
    clips, Q, qb = synthetic_clips(c, voices=("kokoro/a",))
    with pytest.raises(ValueError, match="2 or more voices"):
        C.enroll_audio(b0, c, clips, Q, qb, 0.90)
    clips, Q, qb = synthetic_clips(c)
    clips[0] = dict(clips[0], action="defrost")
    with pytest.raises(ValueError, match="defrost"):
        C.enroll_audio(b0, c, clips, Q, qb, 0.90)


def run_compile(tmp_path, *extra):
    return subprocess.run([sys.executable, str(ROOT / "sdk/compile.py"), str(FAN), "-o", str(tmp_path / "fan.mgc"),
                           "--expansion", "none", *extra], capture_output=True, text=True)


def fake_teacher(tmp_path):
    d = tmp_path / CM.TEACHER[CM.SHIPPING]           # named like the shipping model's teacher, but nothing to load
    d.mkdir()
    return d


def test_cli_default_is_audio_and_its_inputs_are_checked_before_the_teacher_loads(tmp_path):
    r = run_compile(tmp_path, "--embedder", str(fake_teacher(tmp_path)), "--enroll-clips", str(tmp_path / "missing.jsonl"))
    assert r.returncode != 0 and "enrollment manifest not found" in r.stderr
    assert "compiling for model" not in r.stderr           # stopped before the embedder was built
    r = run_compile(tmp_path, "--embedder", str(tmp_path / CM.TEACHER[CM.SHIPPING]), "--golden", str(tmp_path / "none.npz"))
    assert r.returncode != 0 and "no chip model export" in r.stderr
    man = tmp_path / "enroll.jsonl"
    man.write_text(json.dumps(dict(product="microwave", action="start", voice="kokoro/a", out="x.wav")) + "\n")
    r = run_compile(tmp_path, "--embedder", str(tmp_path / CM.TEACHER[CM.SHIPPING]), "--enroll-clips", str(man))
    assert r.returncode != 0 and "no enrollment clips for 'fan'" in r.stderr


def test_cli_text_enrollment_warns_and_stub_blobs_stay_text(tmp_path):
    r = run_compile(tmp_path, "--embedder", "stub", "--enroll", "text")
    assert r.returncode == 0, r.stderr
    assert "most commands are rejected" in r.stderr and json.load(open(tmp_path / "fan.mgc.json"))["enroll"] == "text"
    r = run_compile(tmp_path, "--embedder", "stub")
    assert r.returncode == 0, r.stderr
    assert "stub embedder: text prototypes" in r.stderr


def test_chip_q_cache_survives_a_second_save(tmp_path):
    """Nov 17: the second compile on Windows failed replacing a cache file np.load still held open."""
    cache = tmp_path / "q-test.npz"
    C.cache_save(cache, {"a.wav": np.arange(4, dtype=np.int8)})
    have = C.cache_load(cache)
    have["b.wav"] = np.ones(4, np.int8)
    C.cache_save(cache, have)                      # replaces the file just read
    back = C.cache_load(cache)
    assert sorted(back) == ["a.wav", "b.wav"] and back["a.wav"].tolist() == [0, 1, 2, 3]


def test_default_rejection_rate_per_enrollment():
    """Audio aims at plan target 4 (95% rejected), chosen on the eval DEV split; text keeps its Nov 15 dev choice."""
    assert C.default_reject_rate("audio") == 0.95 and C.default_reject_rate("text") == 0.90


def test_evaluate_uses_the_sdk_matching_rule():
    src = (ROOT / "sw/intent/evaluate.py").read_text(encoding="utf-8")
    assert "def pick_thr_margin" not in src and "def scores" not in src
    assert "C.scores, C.pure_action, C.decide, C.pick_thr_margin" in src


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
