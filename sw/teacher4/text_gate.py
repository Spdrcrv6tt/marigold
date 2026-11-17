"""Gate 0 for teacher v4: text-only matching in the chip space, the upper bound for audio.

The q of a test line is the teacher's own golden vector of its text (golden-format 5, int8 at 5 fractional bits),
matched against the SDK's config blob for the product, exactly as the chip matches the encoder's q:
  sdk      sdk/compile.py compile_config(proj="teacher"): k-means prototypes of the typed examples plus the SDK
           expansion (fill 32 KB), 16 reject prototypes from the expansion non-commands, optionally action-0 prototypes
           from the generic reject bank (--bank; iteration-2 reject fix); integer scores; the (threshold, margin) pair
           chosen on the calibration DEV set only (sw/intent/evaluate.py pick_thr_margin: the lowest threshold rejecting
           95% of the dev headline non-commands, the margin with the best dev accuracy);
  knn_neg  every example and expansion phrase is a prototype, every expansion non-command a reject prototype
           (float cosine), threshold from the same dev rule, no margin.
--cal v1: eval/text dev split (iteration-1 rule); --cal dev2: eval/text_dev2 (iteration-2 reject fix). With --grid the
bank settings (prototypes 0/32/64/96, keep 0.7/0.8/0.9) are chosen per product on the calibration set only.
Scored sets (--sets): v1 = eval/text test, v2 = eval/text_v2 (seen since Nov 16), v3 = eval/text_v3 (blind, once).
Accuracy counts a rejection as an error.

Usage: python sw/teacher4/text_gate.py MODEL [--cal v1|dev2] [--sets v1,v2,v3] [--bank NPZ --grid] [--out JSON]
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf"))
sys.path.insert(0, str(ROOT / "sdk")); sys.path.insert(0, str(ROOT / "eval/gate0"))
import compile as C  # noqa: E402
from text_bound import load_paraphrases, load_noncommands  # noqa: E402

HEAD = ("chat", "near", "tv")
DIRS = {"v2": "eval/text_v2", "v3": "eval/text_v3", "dev2": "eval/text_dev2"}


class Emb:
    """sdk/compile.py's deterministic STEmbedder with a cache (the text gate re-embeds the same lines often)."""
    def __init__(self, path):
        self.path = Path(path)
        self.m = C.STEmbedder(path)
        self.cache = {}

    def encode(self, texts):
        miss = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if miss:
            for t, v in zip(miss, self.m.encode(miss)):
                self.cache[t] = v.astype(np.float64)
        return np.stack([self.cache[t] for t in texts])


def load_set(name):
    """Rows of a v2-layout set: commands {product, action, text, values, split=name}, non-commands {cat, text, split=name}."""
    para, nonc = [], []
    for f in sorted((ROOT / DIRS[name]).glob("*.txt")):
        cur = None
        for line in open(f, encoding="utf8"):
            line = line.rstrip("\n")
            if line.startswith("## "):
                cur = line[3:].split()
                continue
            if not line or line.startswith("#") or cur is None:
                continue
            text, _, vals = line.partition(" | ")
            if "noncommand" in f.name:
                nonc.append(dict(cat=" ".join(cur), text=text.strip(), split=name))
            else:
                para.append(dict(product=cur[0], action=cur[1], text=text.strip(), values=vals.strip(), split=name))
    return para, nonc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--v2", action="store_true", help="(old flag) same as --sets v1,v2")
    ap.add_argument("--sets", default="v1")
    ap.add_argument("--cal", default="v1", choices=["v1", "dev2"])
    ap.add_argument("--bank", default="")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--expansion", default="data/expand/expanded_qwen25.json")
    ap.add_argument("--out")
    a = ap.parse_args()
    sets_wanted = ["v1", "v2"] if a.v2 else a.sets.split(",")
    sys.path.insert(0, str(ROOT / "sw/intent"))
    from evaluate import scores, pick_thr_margin  # noqa: E402
    emb = Emb(ROOT / a.model if not Path(a.model).is_absolute() else a.model)
    z = np.load(emb.path / "marigold_proj.npz"); mu, P = z["mu"].astype(np.float64), z["P"].astype(np.float64)
    G = lambda texts: C.unit(C.golden_vec(emb.encode(texts), mu, P))
    Qi = lambda texts: np.clip(np.round(C.golden_vec(emb.encode(texts), mu, P) * 32.0), -127, 127).astype(np.int64)
    exp = json.load(open(ROOT / a.expansion, encoding="utf8"))
    bank = np.load(ROOT / a.bank) if a.bank else None
    p1, n1 = load_paraphrases(), load_noncommands()
    sets = {}
    if "v1" in sets_wanted:
        sets["v1"] = ([r for r in p1 if r["split"] == "test"], [r for r in n1 if r["split"] == "test"])
    for s_ in ("v2", "v3"):
        if s_ in sets_wanted:
            sets[s_] = load_set(s_)
    if a.cal == "v1":
        calp, caln = [r for r in p1 if r["split"] == "dev"], [r for r in n1 if r["split"] == "dev" and r["cat"] in HEAD]
    else:
        calp, cn = load_set("dev2"); caln = [r for r in cn if r["cat"] in HEAD]
    grid = [(0, 0.8)] + ([(k, t) for k in (32, 64, 96) for t in (0.7, 0.8, 0.9)] if (bank is not None and a.grid) else
                         ([(64, 0.8)] if bank is not None else []))
    out = {"model": a.model, "cal": a.cal, "bank": a.bank or None, "products": {}}
    for prod in ("fan", "microwave"):
        cfg = json.load(open(ROOT / f"sdk/configs/{prod}.json", encoding="utf8"))
        name2id = {x["name"]: x["id"] for x in cfg["actions"]}
        dev_p = [r for r in calp if r["product"] == prod]
        yd = np.array([name2id[r["action"]] for r in dev_p])
        Qdp, Qdn = Qi([r["text"] for r in dev_p]), Qi([r["text"] for r in caln])
        tried = []
        for k, t in grid:
            blob, rep = C.compile_config(cfg, prod, emb, exp, proj="teacher", reject_bank=bank if k else None, bank_protos=k, bank_keep=t,
                                         model_tag=None)    # K1: candidate teachers, so a blob for no chip model (id 0)
            b = C.parse(blob)
            thr, mg, dacc = pick_thr_margin(b, Qdn, Qdp, yd)
            tried.append(dict(bank_protos=k, bank_keep=t, thr=int(thr), margin=int(mg), dev_acc=dacc, N=int(b["N"]),
                              protos_per_action=rep["protos_per_action"], b=b))
            print(f"{prod} cal {a.cal} bank {k:>3} keep {t:.1f}: dev acc {dacc:.3f} thr {thr} margin {mg} N {b['N']}", flush=True)
        best = max(tried, key=lambda x: x["dev_acc"])
        b, thr, mg = best["b"], best["thr"], best["margin"]
        res = {"chosen": {k_: v for k_, v in best.items() if k_ != "b"}, "grid": [{k_: v for k_, v in x.items() if k_ != "b"} for x in tried]}
        # knn_neg (float, all phrases), threshold on the same calibration set
        exa = exp["actions"][prod]
        ph, lab = [], []
        for x in cfg["actions"]:
            e = list(dict.fromkeys(x["examples"] + exa.get(x["name"], [])))
            ph += e; lab += [x["id"]] * len(e)
        Vp, lab = G(ph), np.array(lab)
        Vn = G(exp["negatives"][prod])

        def knn(texts):
            q = G(texts)
            S = q @ Vp.T; sn = (q @ Vn.T).max(1)
            bi = S.argmax(1); best_ = S[np.arange(len(S)), bi]
            return np.where(sn >= best_, 0, lab[bi]), np.where(sn >= best_, -1.0, best_)
        _, bn = knn([r["text"] for r in caln])
        kd = np.sort(bn); k_thr = float(kd[int(np.ceil(0.95 * len(kd))) - 1] + 1e-9)
        for sname, (P_, N_) in sets.items():
            cp = [r for r in P_ if r["product"] == prod]
            cn = list(N_)
            y = np.array([name2id[r["action"]] for r in cp])
            headm = np.array([r["cat"] in HEAD for r in cn])
            best_s, act, other = scores(b, Qi([r["text"] for r in cp]))
            ok = (act == y) & (best_s >= thr) & (best_s - other >= mg)
            bn_, an_, on_ = scores(b, Qi([r["text"] for r in cn]))
            rej = (an_ == 0) | (bn_ < thr) | (bn_ - on_ < mg)
            pa, pb = knn([r["text"] for r in cp])
            na, nb = knn([r["text"] for r in cn])
            res[sname] = {"n": len(cp), "n_non": int(headm.sum()),
                          "sdk_acc": float(ok.mean()), "sdk_acc_norej": float((act == y).mean()), "sdk_rej": float(rej[headm].mean()),
                          "sdk_rej_other": float(rej[~headm].mean()) if (~headm).any() else None,
                          "sdk_rej_by_cat": {c_: float(rej[np.array([r["cat"] == c_ for r in cn])].mean()) for c_ in sorted({r["cat"] for r in cn})},
                          "knn_acc": float(((pa == y) & (pb >= k_thr)).mean()), "knn_acc_norej": float((pa == y).mean()),
                          "knn_rej": float(((na == 0) | (nb < k_thr))[headm].mean()),
                          "per_action_sdk": {x: float(ok[y == name2id[x]].mean()) for x in name2id if (y == name2id[x]).any()},
                          "errors_sdk": [f"{r['action']} -> {act[i] if (act[i] != 0 and best_s[i] >= thr and best_s[i] - other[i] >= mg) else 'REJECT'}  {r['text']}"
                                         for i, r in enumerate(cp) if not ok[i]][:80],
                          "false_accepts_sdk": [r["cat"] + ": " + r["text"] for i, r in enumerate(cn) if not rej[i]][:60]}
            print(f"{a.model} {prod:<9} {sname:<3} [bank {best['bank_protos']} keep {best['bank_keep']}] sdk acc {res[sname]['sdk_acc']:.3f} "
                  f"(no-rej {res[sname]['sdk_acc_norej']:.3f}) rej {res[sname]['sdk_rej']:.3f} other {res[sname]['sdk_rej_other']} | "
                  f"knn_neg acc {res[sname]['knn_acc']:.3f} rej {res[sname]['knn_rej']:.3f}  n {len(cp)}/{int(headm.sum())}", flush=True)
        out["products"][prod] = res
    if a.out:
        json.dump(out, open(ROOT / a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
