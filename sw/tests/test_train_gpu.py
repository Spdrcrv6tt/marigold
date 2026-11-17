"""Mythos's test (deoxylabs/marigold PR 1, Nov 17, 2026), adapted to master (marked ADAPTED): no --device (master trains on CUDA only); master keeps no tensor names in the checkpoint, so the Moonshine stage reads the 'moonshine init: N tensors' line instead.

sw/intent/train.py end to end on a GPU with a tiny synthetic dataset (review S1, S2). Run: pytest sw/tests

Runs on CUDA, or on an Apple GPU (--device mps); skipped when there is neither. It copies sw/intent into a
temporary tree and builds the data there, so the real data/ tree is never touched. The audio is synthetic tones and noise: only the wiring is under test (checkpoints, resume, QAT
start), not any accuracy.

Environment:
  MARIGOLD_MOONSHINE   folder holding snapshots/*/model.safetensors of UsefulSensors/moonshine-streaming-tiny, for
                       the --moonshine stage (default: model.py's own path; the stage is skipped when absent)
  MARIGOLD_AUTOCAST    "float16" on a GPU without bfloat16 (e.g. a V100); train.py itself asks for bfloat16
"""
import glob
import re
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sf = pytest.importorskip("soundfile")
DEVICE = "cuda" if torch.cuda.is_available() else None          # ADAPTED: master has no --device
pytestmark = pytest.mark.skipif(DEVICE is None, reason="needs a CUDA or Apple GPU")

ROOT = Path(__file__).resolve().parents[2]
D = 128
RUN = '''import os, sys
sys.path.insert(0, "sw/intent")
import torch
import model
if os.environ.get("MARIGOLD_MOONSHINE"):
    model.MOONSHINE = os.environ["MARIGOLD_MOONSHINE"]
if os.environ.get("MARIGOLD_AUTOCAST") == "float16":
    _autocast = torch.autocast
    torch.autocast = lambda device, dtype=None, **kw: _autocast(device, dtype=torch.float16, **kw)
import train
if __name__ == "__main__":        # data-loader workers are spawned on macOS and Windows and re-import this file
    train.main()
'''


def tone(rng, seconds):
    t = np.arange(int(seconds * 16000)) / 16000
    x = sum(rng.uniform(0.1, 0.4) * np.sin(2 * np.pi * rng.uniform(120, 3000) * t) for _ in range(4))
    return (x + 0.02 * rng.standard_normal(len(t))).astype(np.float32) * 0.3


def build_tree(root):
    """A copy of sw/intent plus the smallest data tree train.py can run on. Returns nothing; everything is under root."""
    rng = np.random.default_rng(0)
    (root / "sw").mkdir(parents=True)
    shutil.copytree(ROOT / "sw/intent", root / "sw/intent", ignore=shutil.ignore_patterns("__pycache__"))
    (root / "run.py").write_text(RUN)
    rows, texts = [], []

    def clip(name, seconds):
        p = root / "data/audio" / f"{name}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(p), tone(rng, seconds), 16000, subtype="PCM_16")
        return f"data/audio/{name}.wav"

    def add(text, src, n, **extra):
        texts.append(text)
        rows.extend(dict(text=text, src=src, path=clip(f"{src}{len(rows)}", rng.uniform(1.0, 1.8)), **extra) for _ in range(n))

    add("turn it on", "tts", 4, kind="cmd", product="p0", action="on")
    add("set it to level five", "tts", 4, kind="cmd", product="p0", action="level")
    add("what a nice day", "tts", 4, kind="none", product="p0", action=None)
    add("he walked along the river", "libri", 6)
    add("stop", "sc", 6)
    (root / "data/intent").mkdir(parents=True)
    with open(root / "data/intent/index_train.jsonl", "w", encoding="utf8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    Y = rng.standard_normal((len(texts), D))
    Y /= np.sqrt((Y * Y).mean(1, keepdims=True))                    # unit RMS, as the teacher targets are
    np.savez(root / "data/intent/targets-smoke.npz", texts=np.array(texts, dtype=object), Y=Y.astype(np.float32))
    np.savez(root / "data/intent/protos-smoke.npz", pa=np.array(["p0|on", "p0|level"], dtype=object),
             prods=np.array(["p0"], dtype=object), A=rng.standard_normal((2, 4, D)).astype(np.float32),
             R=rng.standard_normal((1, 4, D)).astype(np.float32), pa_prod=np.array([0, 0]))
    (root / "data/noise").mkdir(parents=True)
    with open(root / "data/noise/index.jsonl", "w") as f:
        for kind in ("fan", "microwave", "background"):
            p = root / f"data/noise/{kind}.wav"
            sf.write(str(p), (0.1 * rng.standard_normal(16000 * 4)).astype(np.float32), 16000, subtype="PCM_16")
            f.write(json.dumps(dict(split="train", type=kind, path=f"data/noise/{kind}.wav")) + "\n")
    for i in range(8):                                               # word tails for the pre-roll ([::7] of the list)
        p = root / f"data/speech_commands/go/{i}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(p), tone(rng, 0.5), 16000, subtype="PCM_16")


def train(root, tag, *args):
    """Six optimiser steps of train.py in the temporary tree. Returns (its output, the checkpoint it saved)."""
    out = subprocess.run([sys.executable, "run.py", "--tag", tag, "--targets", "data/intent/targets-smoke.npz",
                          "--protos", "data/intent/protos-smoke.npz", "--steps", "6", "--minutes", "3", "--bs", "4",
                          "--workers", "1", "--gpu-gb", "3", "--max-s", "2", *args],
                         cwd=root, capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, (out.stdout + out.stderr)[-1500:]
    ck = torch.load(root / f"data/intent/{tag}.pt", map_location="cpu", weights_only=False)
    return out.stdout, ck


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("marigold")
    build_tree(root)
    return root


@pytest.fixture(scope="module")
def fresh(tree):
    return train(tree, "s1", "--thr0", "0.30")


def test_the_final_checkpoint_has_the_threshold(fresh):
    log, ck = fresh
    assert "reject threshold starts at 0.3" in log
    assert ck["thr"] == pytest.approx(0.30, abs=0.02)
    assert ck["step"] == 6 and ck["qat"] is False


def test_a_resume_starts_from_the_checkpoints_threshold_not_thr0(tree, fresh):
    log, ck = train(tree, "s2", "--init", "data/intent/s1.pt")       # --thr0 left at its default 0.45
    assert "reject threshold starts at 0.3" in log
    assert ck["thr"] == pytest.approx(fresh[1]["thr"], abs=0.02)


def test_qat_starts_from_the_float_models_threshold(tree, fresh):
    log, ck = train(tree, "q1", "--qat", "--init", "data/intent/s1.pt")
    assert "reject threshold starts at 0.3" in log
    assert ck["qat"] is True and ck["thr"] == pytest.approx(fresh[1]["thr"], abs=0.02)


def test_a_resume_keeps_the_moonshine_tensors_in_their_group(tree):
    sys.path.insert(0, str(ROOT / "sw/intent"))
    import model
    where = os.environ.get("MARIGOLD_MOONSHINE", model.MOONSHINE)
    if not glob.glob(where + "/snapshots/*/model.safetensors"):
        pytest.skip("no Moonshine Streaming Tiny weights (set MARIGOLD_MOONSHINE)")
    log, ck = train(tree, "m1", "--moonshine", "--lr-pre", "0.3")
    n = int(re.search(r"moonshine init: (\d+) tensors", log).group(1))     # ADAPTED
    assert n > 10
    log2, ck2 = train(tree, "m2", "--moonshine", "--lr-pre", "0.3", "--init", "data/intent/m1.pt")
    assert f"moonshine init: {n} tensors" in log2                            # ADAPTED: same names on the resume
