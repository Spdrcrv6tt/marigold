"""Review S6 / S7 checks for sw/intent/evaluate.py (CPU only, no audio):
  * S7: the "pure" rejection-off mode picks the best action prototype even when a reject prototype scores higher,
    while the plain rejection-off mode (threshold and margin off) still lets the reject prototype win;
  * S6: the enrollment speech bank is a frozen list of 1,500 distinct LibriSpeech training utterances.
Run: python sw/intent/test_evaluate_rules.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sw/intent"))
import evaluate as E  # noqa: E402


def test_pure_rejection_off():
    V = np.array([[10, 0], [0, 10], [9, 9]], dtype=np.int8)          # action 1, action 2, a reject prototype
    b = {"vectors": V, "vec_action": np.array([1, 2, 0], np.int32), "bias": np.zeros(3, np.int32), "margin": 0}
    Q = np.array([[10, 8], [8, 10], [0, 10]], dtype=np.int8)
    plain = E.decide(b, Q, -10 ** 9, 0)[0]
    assert plain.tolist() == [0, 0, 2]                                # the reject prototype wins twice
    assert E.pure_action(b, Q).tolist() == [1, 2, 2]                  # masked: pure action confusion


def test_speech_bank_frozen():
    lines = [l.strip() for l in open(ROOT / "sw/intent/speech_bank.txt", encoding="utf8") if l.strip() and not l.startswith("#")]
    assert len(lines) == 1500 == len(set(lines))
    assert all(l.startswith("data/librispeech") and "train-" in l and l.endswith(".flac") for l in lines)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
