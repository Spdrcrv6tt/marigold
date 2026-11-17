"""Teacher targets for the intent encoder, and the fixed teacher-side projection the SDK uses.

  python sw/intent/targets.py data/models/teacher-v3 [--dims 256]

1. Fits the projection on training text only (data/gate0b/train_text_v3.json lines plus the training-audio
   transcripts in data/intent/index_train.jsonl): mu = mean teacher vector, P = top-D principal directions.
2. Writes <teacher>/marigold_proj.npz {mu [768|384], P [D, E]} and the rule (golden-format.md, 'config vectors'):
       v = P (t - mu);  v = v - mean(v);  v = v / rms(v)          (t = normalized teacher embedding)
   so a config vector and the chip's q live in the same unit-RMS, zero-mean D-space.
3. Writes data/intent/targets-<teacher name>.npz {texts, Y float16 [n, D]} for every transcript in the index.
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf"))
sys.path.insert(0, str(ROOT / "sw"))
from textfilter import EvalFilter  # noqa: E402


def project(T, mu, P):
    v = (T - mu) @ P.T
    v = v - v.mean(1, keepdims=True)
    return v / np.sqrt((v * v).mean(1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teacher")
    ap.add_argument("--dims", type=int, default=256)
    a = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    import torch
    tp = ROOT / a.teacher if not Path(a.teacher).is_absolute() else Path(a.teacher)
    big = sum(f.stat().st_size for f in tp.glob("*.safetensors")) > 2e9      # iteration 2: multi-GB teachers in bf16
    m = SentenceTransformer(str(tp), device="cuda", model_kwargs={"dtype": torch.bfloat16} if big else {})
    enc = lambda xs: m.encode(list(xs), normalize_embeddings=True, batch_size=256 if big else 512, show_progress_bar=False).astype(np.float32)
    flt = EvalFilter()
    idx = [json.loads(l) for l in open(ROOT / "data/intent/index_train.jsonl", encoding="utf8")]
    texts = sorted({r["text"] for r in idx})
    tt = json.load(open(ROOT / "data/gate0b/train_text_v3.json", encoding="utf8"))
    fit = sorted({t for r in tt.values() for v in list(r["actions"].values()) + [r["none"]] for t in v if flt.ok(t)} | set(texts))
    pj = Path(ROOT / a.teacher) / "marigold_proj.npz"
    if pj.exists():                     # frozen with the teacher: never refit (the SDK's config vectors use it)
        z = np.load(pj); mu, P = z["mu"], z["P"]
        print(f"projection {P.shape} reused from {pj}", flush=True)
    else:
        rng = np.random.default_rng(0)
        fit = [fit[i] for i in rng.permutation(len(fit))[:60000]]
        Tf = enc(fit)
        mu = Tf.mean(0)
        _, sv, vt = np.linalg.svd(Tf - mu, full_matrices=False)
        P = vt[:a.dims]
        ev = (sv[:a.dims] ** 2).sum() / (sv ** 2).sum()
        np.savez(pj, mu=mu, P=P)
        print(f"projection {P.shape}, variance kept {ev:.3f} (fit on {len(fit)} training texts)", flush=True)
    Y = project(enc(texts), mu, P).astype(np.float16)
    name = Path(a.teacher).name
    np.savez(ROOT / f"data/intent/targets-{name}.npz", texts=np.array(texts, dtype=object), Y=Y)
    print("targets", Y.shape, flush=True)


if __name__ == "__main__":
    main()
