"""Gate 0 for teacher v4: text-only matching in the chip space, the upper bound for audio.

The q of a test line is the teacher's own golden vector of its text (golden-format 5, int8 at 5 fractional bits),
matched against the SDK's config blob for the product, exactly as the chip matches the encoder's q:
  sdk      sdk/compile.py compile_config(proj="teacher"): k-means prototypes of the typed examples plus the SDK
           expansion (fill 32 KB), 16 reject prototypes from the expansion non-commands; integer scores; the
           (threshold, margin) pair chosen on eval/text DEV only (sw/intent/evaluate.py pick_thr_margin: the lowest
           threshold rejecting 95% of the dev headline non-commands, the margin with the best dev accuracy);
  knn_neg  every example and expansion phrase is a prototype, every expansion non-command a reject prototype
           (float cosine), threshold from the same dev rule, no margin.
eval/text: dev tunes, test scores (once). eval/text_v2: TEST-ONLY, scored with the v1-dev threshold and margin.
Accuracy counts a rejection as an error.

Usage: python sw/teacher4/text_gate.py MODEL [--v2] [--out JSON]
"""
import argparse, json, os, re, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf"))
sys.path.insert(0, str(ROOT / "sdk")); sys.path.insert(0, str(ROOT / "eval/gate0"))
import compile as C  # noqa: E402
from text_bound import load_paraphrases, load_noncommands  # noqa: E402

HEAD = ("chat", "near", "tv")


class Emb:
    def __init__(self, path):
        import torch
        from sentence_transformers import SentenceTransformer
        self.path = Path(path)
        big = sum(f.stat().st_size for f in self.path.glob("*.safetensors")) > 2e9
        self.m = SentenceTransformer(str(path), device="cuda", model_kwargs={"dtype": torch.bfloat16} if big else {})
        self.cache = {}

    def encode(self, texts):
        miss = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if miss:
            for t, v in zip(miss, self.m.encode(miss, batch_size=128, normalize_embeddings=True, convert_to_numpy=True)):
                self.cache[t] = v.astype(np.float64)
        return np.stack([self.cache[t] for t in texts])


def load_v2():
    """eval/text_v2 rows: commands {product, action, text, values} and non-commands {cat, text}. Layout is read from
    the files (v1-style '## product action' / '## category' headers)."""
    d = ROOT / "eval/text_v2"
    para, nonc = [], []
    for f in sorted(d.glob("*.txt")):
        cur = None
        for line in open(f, encoding="utf8"):
            line = line.rstrip("\n")
            if line.startswith("## "):
                cur = line[3:].split()
                continue
            if not line or line.startswith("#") or cur is None:
                continue
            text, _, vals = line.partition(" | ")
            if "noncommand" in f.name or "non_command" in f.name or "negatives" in f.name:
                nonc.append(dict(cat=" ".join(cur), text=text.strip(), split="v2"))
            else:
                para.append(dict(product=cur[0], action=cur[1], text=text.strip(), values=vals.strip(), split="v2"))
    return para, nonc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--v2", action="store_true")
    ap.add_argument("--expansion", default="data/expand/expanded_qwen25.json")
    ap.add_argument("--out")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "sw/intent"))
    from evaluate import scores, pick_thr_margin  # noqa: E402
    emb = Emb(ROOT / a.model if not Path(a.model).is_absolute() else a.model)
    z = np.load(emb.path / "marigold_proj.npz"); mu, P = z["mu"].astype(np.float64), z["P"].astype(np.float64)
    G = lambda texts: C.unit(C.golden_vec(emb.encode(texts), mu, P))
    Qi = lambda texts: np.clip(np.round(C.golden_vec(emb.encode(texts), mu, P) * 32.0), -127, 127).astype(np.int64)
    exp = json.load(open(ROOT / a.expansion, encoding="utf8"))
    para, nonc = load_paraphrases(), load_noncommands()
    sets = {"v1": (para, nonc)}
    if a.v2:
        sets["v2"] = load_v2()
    out = {"model": a.model, "products": {}}
    for prod in ("fan", "microwave"):
        cfg = json.load(open(ROOT / f"sdk/configs/{prod}.json", encoding="utf8"))
        name2id = {x["name"]: x["id"] for x in cfg["actions"]}
        blob, rep = C.compile_config(cfg, prod, emb, exp, proj="teacher")
        b = C.parse(blob)
        res = {"blob_N": int(b["N"]), "protos_per_action": rep["protos_per_action"]}
        pr = [r for r in para if r["product"] == prod]
        dev_p = [r for r in pr if r["split"] == "dev"]
        dev_n = [r for r in nonc if r["split"] == "dev" and r["cat"] in HEAD]
        yd = np.array([name2id[r["action"]] for r in dev_p])
        thr, mg, dacc = pick_thr_margin(b, Qi([r["text"] for r in dev_n]), Qi([r["text"] for r in dev_p]), yd)
        res["sdk_thr"], res["sdk_margin"], res["sdk_dev_acc"] = int(thr), int(mg), dacc
        # knn_neg (float, all phrases)
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
            bi = S.argmax(1); best = S[np.arange(len(S)), bi]
            return np.where(sn >= best, 0, lab[bi]), np.where(sn >= best, -1.0, best)
        an, bn = knn([r["text"] for r in dev_n])
        kd = np.sort(bn); k_thr = float(kd[int(np.ceil(0.95 * len(kd))) - 1] + 1e-9)
        res["knn_thr"] = k_thr
        for sname, (P_, N_) in sets.items():
            for sp in (("dev", "test") if sname == "v1" else ("v2",)):
                cp = [r for r in P_ if r["product"] == prod and r["split"] == sp]
                cn = [r for r in N_ if r["split"] == sp]
                y = np.array([name2id[r["action"]] for r in cp])
                headm = np.array([(r["cat"] in HEAD) if sname == "v1" else not r["cat"].startswith("other") for r in cn])
                # sdk (integer, the chip's rule)
                best, act, other = scores(b, Qi([r["text"] for r in cp]))
                ok = (act == y) & (best >= thr) & (best - other >= mg)
                bn_, an_, on_ = scores(b, Qi([r["text"] for r in cn]))
                rej = (an_ == 0) | (bn_ < thr) | (bn_ - on_ < mg)
                pa, pb = knn([r["text"] for r in cp])
                na, nb = knn([r["text"] for r in cn])
                key = f"{sname}_{sp}"
                res[key] = {"n": len(cp), "n_non": int(headm.sum()),
                            "sdk_acc": float(ok.mean()), "sdk_acc_norej": float((act == y).mean()), "sdk_rej": float(rej[headm].mean()),
                            "sdk_rej_other": float(rej[~headm].mean()) if (~headm).any() else None,
                            "knn_acc": float(((pa == y) & (pb >= k_thr)).mean()), "knn_acc_norej": float((pa == y).mean()),
                            "knn_rej": float(((na == 0) | (nb < k_thr))[headm].mean()),
                            "per_action_sdk": {x: float(ok[y == name2id[x]].mean()) for x in name2id if (y == name2id[x]).any()},
                            "errors_sdk": [f"{r['action']} -> {act[i] if (act[i] != 0 and best[i] >= thr and best[i] - other[i] >= mg) else 'REJECT'}  {r['text']}"
                                           for i, r in enumerate(cp) if not ok[i]][:60] if sp != "dev" else [],
                            "false_accepts_sdk": [r["text"] for i, r in enumerate(cn) if headm[i] and not rej[i]][:40] if sp != "dev" else []}
                print(f"{a.model} {prod:<9} {key:<8} sdk acc {res[key]['sdk_acc']:.3f} (no-rej {res[key]['sdk_acc_norej']:.3f}) rej {res[key]['sdk_rej']:.3f}"
                      f" | knn_neg acc {res[key]['knn_acc']:.3f} rej {res[key]['knn_rej']:.3f}  n {len(cp)}/{int(headm.sum())}", flush=True)
        out["products"][prod] = res
    if a.out:
        json.dump(out, open(ROOT / a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
