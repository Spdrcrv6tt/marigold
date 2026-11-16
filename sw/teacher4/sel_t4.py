"""Teacher v4 selection on held-out TRAINING products (never eval/): sw/gate0b/val_heldout.py's protocol (8 typed
examples per action, nearest example, threshold rejecting 95% of half the product's non-commands, rejection counted
as an error) in the chip space (golden-format 5 with the model's own marigold_proj.npz; PCA-D fitted on training text
when the model has none) and in the full embedding space.

Usage: python sw/teacher4/sel_t4.py MODEL [MODEL ...] --holdout "a,b,..." [--out JSON]
"""
import argparse, json, os, random, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf"))
sys.path.insert(0, str(ROOT / "sw"))
from textfilter import EvalFilter  # noqa: E402
from gate0b.val_heldout import score_product  # noqa: E402


def golden(T, mu, P):
    v = (T - mu) @ P.T
    v = v - v.mean(1, keepdims=True)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--data", default="data/gate0b/train_text_v4.json")
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--out")
    a = ap.parse_args()
    import torch
    from sentence_transformers import SentenceTransformer
    flt = EvalFilter()
    data = json.load(open(ROOT / a.data, encoding="utf8"))
    data = {p: {"actions": {k: flt.keep(v) for k, v in r["actions"].items()}, "none": flt.keep(r["none"])} for p, r in data.items()}
    hold = [h.strip() for h in a.holdout.split(",") if h.strip()]
    res = []
    for mn in a.models:
        m = SentenceTransformer(mn, device="cuda", model_kwargs={"dtype": torch.bfloat16} if "qwen" in mn.lower() or "t4-q" in mn.lower() else {})
        cache = {}

        def enc(xs):
            miss = [x for x in dict.fromkeys(xs) if x not in cache]
            if miss:
                for x, v in zip(miss, m.encode(miss, normalize_embeddings=True, batch_size=128, convert_to_numpy=True).astype(np.float64)):
                    cache[x] = v
            return np.stack([cache[x] for x in xs])
        pj = Path(mn) / "marigold_proj.npz"
        if pj.exists():
            z = np.load(pj); mu, P = z["mu"].astype(np.float64), z["P"].astype(np.float64); src = "own head"
        else:
            fit = [t for p, r in data.items() if p not in hold for v in list(r["actions"].values()) + [r["none"]] for t in v]
            random.Random(0).shuffle(fit)
            Tf = enc(fit[:12000]); mu = Tf.mean(0)
            _, _, vt = np.linalg.svd(Tf - mu, full_matrices=False); P = vt[:a.D]; src = "PCA on training text"
        row = {"model": mn, "proj": src}
        for tag, emb in (("full", enc), (f"chip{a.D}", lambda xs: golden(enc(xs), mu, P))):
            accs, rejs, nr = [], [], []
            for p in hold:
                ac, rj, n0 = score_product(emb, data[p], random.Random(1), None)
                accs.append(ac); rejs.append(rj); nr.append(n0)
            row[tag] = {"acc": float(np.mean(accs)), "rej": float(np.mean(rejs)), "acc_norej": float(np.mean(nr)),
                        "per_product": {p: [round(x, 3), round(y, 3)] for p, x, y in zip(hold, accs, rejs)}}
            print(f"{mn:<44} {tag:<8} held-out acc {row[tag]['acc']:.3f} rej {row[tag]['rej']:.3f} "
                  f"(no-reject acc {row[tag]['acc_norej']:.3f}) [{src}]", flush=True)
        res.append(row)
        del m
        torch.cuda.empty_cache()
    if a.out:
        json.dump(res, open(ROOT / a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
