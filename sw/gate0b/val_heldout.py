"""Teacher model selection on held-out TRAINING products (never eval/).

For each held-out product: 8 phrasings per action are the 'typed examples' (fixed seed), the rest are
test phrasings; its non-command lines are split in half (dev sets the threshold: the lowest that rejects
95% of dev non-commands; test measures). Scores: nearest typed example ('max'), in the full teacher space
and after the fixed D-dim projection (PCA, fitted on the training products only) the chip uses.
Accuracy counts a rejection as an error.

Usage: python sw/gate0b/val_heldout.py MODEL [MODEL ...] --holdout "a,b,..." [--dims 256]
"""
import argparse, json, os, random, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf"))
sys.path.insert(0, str(ROOT / "sw"))
from textfilter import EvalFilter  # noqa: E402


def fit_pca(X, d):
    mu = X.mean(0)
    _, _, vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, vt[:d]


def score_product(emb, rec, rng, proj=None):
    names = [a for a in rec["actions"] if len(rec["actions"][a]) >= 10]
    enrol, test = {}, []
    for i, a in enumerate(names):
        ph = sorted(rec["actions"][a]); rng.shuffle(ph)
        enrol[a] = ph[:8]; test += [(t, i) for t in ph[8:]]
    none = sorted(rec["none"]); rng.shuffle(none)
    ndev, ntest = none[::2], none[1::2]

    def E(xs):
        v = emb(xs)
        if proj is not None:
            v = (v - proj[0]) @ proj[1].T
            v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v
    ex = {a: E(enrol[a]) for a in names}

    def S(xs):
        v = E(xs)
        return np.stack([(v @ ex[a].T).max(1) for a in names], 1)
    St = S([t for t, _ in test]); y = np.array([i for _, i in test])
    nd = np.sort(S(ndev).max(1)); thr = nd[int(np.ceil(0.95 * len(nd))) - 1] + 1e-6
    nt = S(ntest).max(1)
    acc = float(((St.argmax(1) == y) & (St.max(1) >= thr)).mean())
    return acc, float((nt < thr).mean()), float((St.argmax(1) == y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--data", default="data/gate0b/train_text_v3.json")
    ap.add_argument("--dims", default="128,256", help="comma-separated projection sizes to test")
    ap.add_argument("--out")
    a = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    flt = EvalFilter()
    data = json.load(open(ROOT / a.data, encoding="utf8"))
    data = {p: {"actions": {k: flt.keep(v) for k, v in r["actions"].items()}, "none": flt.keep(r["none"])} for p, r in data.items()}
    hold = [h.strip() for h in a.holdout.split(",") if h.strip()]
    train_txt = [t for p, r in data.items() if p not in hold for v in list(r["actions"].values()) + [r["none"]] for t in v]
    random.Random(0).shuffle(train_txt)
    res = []
    for mn in a.models:
        m = SentenceTransformer(mn, device="cuda")
        emb = lambda xs: m.encode(list(xs), normalize_embeddings=True, batch_size=256)
        dims = [int(d) for d in a.dims.split(",") if d]
        Xfit = emb(train_txt[:20000])
        row = {"model": mn}
        for tag, pj in [("full", None)] + [(f"pca{d}", fit_pca(Xfit, d)) for d in dims]:
            accs, rejs, nr = [], [], []
            for p in hold:
                ac, rj, n0 = score_product(emb, data[p], random.Random(1), pj)
                accs.append(ac); rejs.append(rj); nr.append(n0)
            row[tag] = {"acc": float(np.mean(accs)), "rej": float(np.mean(rejs)), "acc_norej": float(np.mean(nr)),
                        "per_product": {p: [round(x, 3), round(y, 3)] for p, x, y in zip(hold, accs, rejs)}}
            print(f"{mn:<48} {tag:<7} held-out acc {row[tag]['acc']:.3f} rej {row[tag]['rej']:.3f} "
                  f"(no-reject acc {row[tag]['acc_norej']:.3f})", flush=True)
        res.append(row)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
