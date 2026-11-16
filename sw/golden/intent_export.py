"""Export a QAT checkpoint of the intent encoder to the integer golden model (docs/golden-format.md).

  python sw/golden/intent_export.py data/intent/q1.pt --tag v1 [--calib 400]

Writes data/golden/mg-<tag>.npz (integers), mg-<tag>.json (bits per site, formats, calibration ranges) and
mg-<tag>-float.npz (dequantised weights). Calibration: activation ranges of the float graph (dequantised weights)
on training clips only (data/intent/index_train.jsonl), half of them with training noise mixed in.
Then checks: torch QAT model vs FloatIntent (same graph), FloatIntent vs IntIntent (q cosine), FAST vs int64.
"""
import argparse, json, random, sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sw" / "intent")); sys.path.insert(0, str(ROOT / "sw" / "golden"))
from model import IntentEncoder, enable_qat, quantized  # noqa: E402
import intent_np as S  # noqa: E402


def torch_to_Q(m):
    names = {"fe_lin": "fe.lin", "fe_c1": "fe.c1", "fe_c2": "fe.c2", "proj": "proj", "ctc": "ctc"}
    for i in range(S.NL):
        for c in ("q", "k", "v", "o", "fc1", "fc2"):
            names[f"layers.{i}.{c}"] = f"enc{i}.{c}"
    Q, bits = {}, {}
    for tn, (q, s, b, nb) in quantized(m).items():
        g = names[tn]
        Q[g + ".q"] = q.astype(np.int8); Q[g + ".s"] = s; bits[g] = nb
        if b is not None:
            Q[g + ".b"] = b
    sd = m.state_dict()
    for i in range(S.NL):
        Q[f"enc{i}.ln1.g"] = sd[f"layers.{i}.ln1.g"].double().cpu().numpy()
        Q[f"enc{i}.ln2.g"] = sd[f"layers.{i}.ln2.g"].double().cpu().numpy()
    Q["enc.ln.g"] = sd["ln.g"].double().cpu().numpy()
    Q["fe.k"] = float(torch.exp(sd["log_k"]).item())
    return Q, bits


def calib_clips(n, seed=0):
    rows = [json.loads(l) for l in open(ROOT / "data/intent/index_train.jsonl", encoding="utf8")]
    rng = random.Random(seed)
    pick = rng.sample([r for r in rows if r["src"] in ("tts", "libri") and "start" not in r], n)
    noise = [json.loads(l) for l in open(ROOT / "data/noise/index.jsonl")]
    noise = [r for r in noise if r["split"] == "train" and r["type"] in ("fan", "microwave", "babble", "tv")]
    out = []
    for j, r in enumerate(pick):
        x, _ = sf.read(str(ROOT / r["path"]), dtype="float64")
        x = x / (np.abs(x).max() + 1e-9) * 10 ** (rng.uniform(-25, -2) / 20)
        if j % 2:
            nz, _ = sf.read(str(ROOT / rng.choice(noise)["path"]), dtype="float64", frames=len(x))
            nz = np.resize(nz, len(x))
            x = x + nz * np.sqrt(np.mean(x ** 2) / (np.mean(nz ** 2) + 1e-12) / 10 ** (rng.uniform(0, 20) / 10))
        out.append(np.clip(np.round(x * 32767), -32768, 32767).astype(np.int64))
    return out


def load_qat(path, dev="cpu"):
    ck = torch.load(path, map_location=dev)
    m = IntentEncoder(ck["cfg"]); enable_qat(m); m.load_state_dict(ck["state"]); m.eval()
    return m, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--calib", type=int, default=400)
    a = ap.parse_args()
    m, ck = load_qat(a.ckpt)
    Q, bits = torch_to_Q(m)
    P = S.Q_to_P(Q)
    maxabs = {}

    def rec(site, x):
        v = float(np.abs(x).max())
        maxabs[site] = max(maxabs.get(site, 0.0), v)
    fl = S.FloatIntent(P, rec)
    clips = calib_clips(a.calib)
    for x in clips:
        e = fl.encode(x); fl.head(e); fl.logits(e)
    G = S.build_int(Q, maxabs)
    out = ROOT / "data/golden"; out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"mg-{a.tag}.npz", **G)
    np.savez(out / f"mg-{a.tag}-float.npz", **{k: np.asarray(v) for k, v in P.items()})
    nw = {k: int(np.prod(Q[k + ".q"].shape)) for k in bits}
    rom_bits = sum(nw[k] * bits[k] for k in bits)
    meta = {"format_version": S.FORMAT_VERSION, "ckpt": a.ckpt, "step": ck.get("step"), "cfg": ck["cfg"],
            "weight_bits": bits, "weights": int(sum(nw.values())), "weight_rom_bits": int(rom_bits),
            "formats": {k[2:]: int(v) for k, v in G.items() if k.startswith("F:")}, "maxabs": maxabs, "fe.k": Q["fe.k"],
            "calibration": f"{a.calib} training clips (TTS + LibriSpeech train), half with train-split noise"}
    json.dump(meta, open(out / f"mg-{a.tag}.json", "w"), indent=1)
    print("weights", meta["weights"], "weight ROM Mbit", rom_bits / 1e6, flush=True)
    # checks
    I = S.IntIntent({k: np.asarray(v) for k, v in G.items()})
    cs, same = [], 0
    for x in clips[:12]:
        e = fl.encode(x); qf = fl.head(e)
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32)[None] / 32768
            qt, lt, _ = m(xt, torch.tensor([len(x)]))
        qi, ki = I.run(x)
        cs.append([float(qf @ qt[0].double().numpy() / np.linalg.norm(qf) / np.linalg.norm(qt[0].numpy())),
                   float(qf @ qi / np.linalg.norm(qf) / np.linalg.norm(qi))])
        same += int((fl.logits(e).argmax(1) == ki).mean() > 0.95)
    print("cos(torch QAT, float graph) min %.4f; cos(float graph, integer q) min %.4f mean %.4f; ctc argmax agree %d/12"
          % (min(c[0] for c in cs), min(c[1] for c in cs), np.mean([c[1] for c in cs]), same), flush=True)
    S.IntIntent.FAST = False
    I2 = S.IntIntent({k: np.asarray(v) for k, v in G.items()})
    q2, k2 = I2.run(clips[0][:16000])
    S.IntIntent.FAST = True
    q1, k1 = I.run(clips[0][:16000])
    print("FAST == int64:", bool((q1 == q2).all() and (k1 == k2).all()), flush=True)


if __name__ == "__main__":
    main()
