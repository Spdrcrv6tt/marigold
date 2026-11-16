"""Decision prototypes for training the intent encoder (iteration 2): per training (product, action), K spherical
k-means centroids of the teacher's chip-space vectors (golden-format 5) of its training phrasings; per training
product, Kr centroids of its non-command lines. Training text only (data/gate0b/train_text_v4.json, eval leak filter
applied); never fan or microwave.

  python sw/intent/protos.py data/models/teacher-v4 [--K 8 --Kr 16]  ->  data/intent/protos-<teacher>.npz
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf"))
sys.path.insert(0, str(ROOT / "sw")); sys.path.insert(0, str(ROOT / "sdk"))
from textfilter import EvalFilter  # noqa: E402
import compile as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teacher")
    ap.add_argument("--data", default="data/gate0b/train_text_v4.json")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--Kr", type=int, default=16)
    a = ap.parse_args()
    import torch
    from sentence_transformers import SentenceTransformer
    tp = ROOT / a.teacher
    big = sum(f.stat().st_size for f in tp.glob("*.safetensors")) > 2e9
    m = SentenceTransformer(str(tp), device="cuda", model_kwargs={"dtype": torch.bfloat16} if big else {})
    z = np.load(tp / "marigold_proj.npz"); mu, P = z["mu"].astype(np.float64), z["P"].astype(np.float64)
    G = lambda xs: C.unit(C.golden_vec(m.encode(list(xs), batch_size=256, normalize_embeddings=True).astype(np.float64), mu, P))
    flt = EvalFilter()
    d = json.load(open(ROOT / a.data, encoding="utf8"))
    prods = sorted(p for p in d if p not in ("fan", "microwave"))
    pa, A, R = [], [], []
    for pi, p in enumerate(prods):
        for act, ph in d[p]["actions"].items():
            ph = flt.keep(ph)
            if len(ph) < 2:
                continue
            c = C.kmeans(G(ph), a.K)
            A.append(np.concatenate([c, c[np.arange(a.K - len(c)) % len(c)]]) if len(c) < a.K else c)
            pa.append((p, act))
        nn = flt.keep(d[p]["none"])
        c = C.kmeans(G(nn), a.Kr)
        R.append(np.concatenate([c, c[np.arange(a.Kr - len(c)) % len(c)]]) if len(c) < a.Kr else c)
    np.savez(ROOT / f"data/intent/protos-{tp.name}.npz", prods=np.array(prods, dtype=object),
             pa=np.array([f"{p}|{x}" for p, x in pa], dtype=object), pa_prod=np.array([prods.index(p) for p, _ in pa]),
             A=np.stack(A).astype(np.float32), R=np.stack(R).astype(np.float32))
    print("products", len(prods), "actions", len(pa), "A", np.stack(A).shape, "R", np.stack(R).shape, flush=True)


if __name__ == "__main__":
    main()
