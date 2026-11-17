"""A test export of the intent encoder in the current golden format (sw/golden/intent_np.py, sw/intent/model.py CFG),
for the firmware checks before the model agent's freeze. Not a trained model: Moonshine Streaming Tiny's frontend and
first NL layers (model.load_moonshine; --init random keeps PyTorch's default init), random proj and ctc heads, QAT
scales at their MSE start. Calibrated and exported as sw/golden/intent_export.py does (its functions, unchanged), but
written outside data/golden:

    .venv/Scripts/python.exe rtl/core/tools/mkrand_mg.py [--tag rand3] [--init moonshine|random] [--calib 40]
        -> data/core/mg-<tag>/mg-<tag>.npz, mg-<tag>.json
then  make -C rtl/top TAG=<tag> MGGOLD=<repo>/data/core/mg-<tag> B=$CORE_BUILD/r3 host-mg image-mg
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "sw" / "intent")); sys.path.insert(0, str(ROOT / "sw" / "golden"))
import model as MD  # noqa: E402
import intent_np as S  # noqa: E402
import intent_export as EX  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rand3")
    ap.add_argument("--init", choices=["moonshine", "random"], default="moonshine")
    ap.add_argument("--calib", type=int, default=40)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    cfg = dict(MD.CFG)
    assert (cfg["W"], cfg["NH"], cfg["HD"], cfg["NL"], cfg["FF"]) == (S.W, S.NH, S.HD, S.NL, S.FF), "model.py vs intent_np"
    m = MD.IntentEncoder(cfg)
    if a.init == "moonshine":
        MD.load_moonshine(m)
    MD.enable_qat(m); m.eval()
    Q, bits = EX.torch_to_Q(m)
    P = S.Q_to_P(Q)
    maxabs = {}

    def rec(site, x):
        maxabs[site] = max(maxabs.get(site, 0.0), float(np.abs(x).max()))
    fl = S.FloatIntent(P, rec)
    for x in EX.calib_clips(a.calib):
        e = fl.encode(x); fl.head(e); fl.logits(e)
    G = S.build_int(Q, maxabs)
    out = ROOT / "data" / "core" / f"mg-{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"mg-{a.tag}.npz", **G)
    nw = {k: int(np.prod(Q[k + ".q"].shape)) for k in bits}
    meta = {"format_version": S.FORMAT_VERSION, "ckpt": f"none: rtl/core/tools/mkrand_mg.py --init {a.init} --seed {a.seed}",
            "step": 0, "cfg": {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()}, "weight_bits": bits,
            "weights": int(sum(nw.values())), "weight_rom_bits": int(sum(nw[k] * bits[k] for k in bits)),
            "formats": {k[2:]: int(v) for k, v in G.items() if k.startswith("F:")}, "maxabs": maxabs, "fe.k": Q["fe.k"],
            "calibration": f"{a.calib} training clips (intent_export.calib_clips)"}
    json.dump(meta, open(out / f"mg-{a.tag}.json", "w"), indent=1)
    print(f"mg-{a.tag}: {meta['weights']} weights, {meta['weight_rom_bits'] / 1e6:.2f} Mbit of weights -> {out}")


if __name__ == "__main__":
    main()
