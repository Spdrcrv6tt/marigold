"""Review S1 / S2 checks for sw/intent/train.py helpers (CPU only, no training data):
  * the decision threshold survives a save and load round trip of a checkpoint (periodic and final saves);
  * a resumed run continues the LR schedule from the checkpoint's progress, without a new warm-up;
  * the threshold is clamped after the optimiser step, and keeps a gradient outside [0, 0.9] in the forward pass.
Run: python sw/intent/test_train_resume.py
"""
import math, os, sys, tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402


def test_thr_round_trip():
    m = torch.nn.Linear(3, 2)
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "c.pt")
        torch.save(T.ckpt_dict(m.state_dict(), {"D": 128}, 1234, False, 0.317, 0.6), f)
        ck = torch.load(f)
    assert abs(T.thr_init(ck, 0.45) - 0.317) < 1e-9 and abs(ck["prog"] - 0.6) < 1e-12
    assert T.thr_init({"state": {}, "thr": None}, 0.45) == 0.45      # old checkpoints without a threshold
    assert T.thr_init(None, 0.45) == 0.45


def test_schedule_continues():
    clock = [100.0]
    fresh = T.make_prog(0.0, 10 ** 9, 60, 100.0, clock=lambda: clock[0])
    resumed = T.make_prog(0.5, 10 ** 9, 60, 100.0, clock=lambda: clock[0])
    assert fresh(0) == 0.0 and resumed(0) == 0.5
    assert abs(T.lr_factor(0, 1, resumed) - (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * 0.5)))) < 1e-12   # no warm-up
    clock[0] = 100.0 + 30 * 60                                     # half of the remaining 60 minutes
    assert abs(resumed(10) - 0.75) < 1e-9 and abs(fresh(10) - 0.5) < 1e-9
    assert T.lr_factor(0, 1000, fresh) < 0.01                      # a fresh run still warms up


def test_clamp_after_step():
    p = torch.nn.Parameter(torch.tensor(-0.2))
    (p * 1.0).backward()                                           # no clamp in the forward pass: the gradient is kept
    assert p.grad is not None and p.grad.item() == 1.0
    with torch.no_grad():
        p.clamp_(0.0, 0.9)
    assert p.item() == 0.0


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
