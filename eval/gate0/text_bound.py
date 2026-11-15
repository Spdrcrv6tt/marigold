"""Gate 0: text-only upper bound for text-enrolled intent matching.

Embeds each config's example phrases and the held-out paraphrase / non-command
sets with a text-embedding model, matches by cosine similarity, and reports
accuracy on unseen phrasings (target 2) and rejection of non-commands (target 4).

Threshold rule (fixed before looking at test numbers): per product, the reject
threshold is the lowest value that rejects >= 95% of the DEV non-commands
(headline categories chat+near+tv). Test numbers use that threshold unchanged.
Accuracy counts a rejected command as wrong.

Usage: python eval/gate0/text_bound.py MODEL [MODEL ...] [--extra FILE] [--score max|top2|centroid] [--out JSON]
"""
import argparse, json, os, re, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf"))
HEADLINE = ("chat", "near", "tv")

# Prefixes a model card asks for in symmetric similarity use.
PREFIX = {"intfloat/e5-small-v2": "query: ", "intfloat/e5-base-v2": "query: ",
          "nomic-ai/nomic-embed-text-v1.5": "clustering: "}


def load_paraphrases():
    rows, cur, i = [], None, 0
    for line in open(ROOT / "eval/text/paraphrases.txt", encoding="utf8"):
        line = line.rstrip("\n")
        if line.startswith("## "):
            cur, i = line[3:].split(), 0
            continue
        if not line or line.startswith("#"):
            continue
        text, _, vals = line.partition(" | ")
        rows.append(dict(product=cur[0], action=cur[1], text=text.strip(), values=vals.strip(),
                         split="dev" if i % 4 == 0 else "test"))
        i += 1
    return rows


def load_noncommands():
    rows, cur, i = [], None, 0
    for line in open(ROOT / "eval/text/noncommands.txt", encoding="utf8"):
        line = line.rstrip("\n")
        if line.startswith("## "):
            cur, i = line[3:].strip(), 0
            continue
        if not line or line.startswith("#"):
            continue
        rows.append(dict(cat=cur, text=line.strip(), split="dev" if i % 3 == 0 else "test"))
        i += 1
    return rows


def load_configs(extra=None):
    cfgs = {}
    for p in ("fan", "microwave"):
        c = json.load(open(ROOT / f"sdk/configs/{p}.json", encoding="utf8"))
        cfgs[p] = {a["name"]: list(a["examples"]) for a in c["actions"]}
    if extra:  # JSON {product: {action: [phrases]}} added to the enrolled examples
        for p, acts in json.load(open(extra, encoding="utf8")).items():
            for a, phr in acts.items():
                cfgs[p][a].extend(phr)
    return cfgs


def norm(s):
    return re.sub(r"[^a-z0-9' ]", "", s.lower()).strip()


def run(model_name, extra=None, score="max"):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name, device="cuda", trust_remote_code=False)
    pre = PREFIX.get(model_name, "")
    emb = lambda xs: m.encode([pre + x for x in xs], normalize_embeddings=True, batch_size=256)
    para, nonc, cfgs = load_paraphrases(), load_noncommands(), load_configs(None)
    negs = {}
    if extra:  # SDK expansion file from sdk/expand.py: {"actions": {p: {a: [...]}}, "negatives": {p: [...]}}
        X = json.load(open(extra, encoding="utf8"))
        for p, acts in X["actions"].items():
            for a, phr in acts.items():
                cfgs[p][a] = cfgs[p][a] + phr
        negs = X.get("negatives", {})
    # Guard: no test phrase may be a typed example (expansion overlaps are counted, not removed).
    overlap = 0
    for r in para:
        if r["split"] != "test":
            continue
        typed = json.load(open(ROOT / f"sdk/configs/{r['product']}.json", encoding="utf8"))
        if norm(r["text"]) in {norm(e) for a in typed["actions"] if a["name"] == r["action"] for e in a["examples"]}:
            sys.exit(f"leak: test phrase typed in config: {r}")
        overlap += norm(r["text"]) in {norm(e) for e in cfgs[r["product"]][r["action"]]}
    out = {"model": model_name, "score": score, "extra": extra, "test_phrases_also_in_expansion": overlap, "products": {}}
    for p, acts in cfgs.items():
        names = list(acts)
        ex_vecs = {a: emb(acts[a]) for a in names}
        neg_vecs = emb(negs[p]) if negs.get(p) else None
        if score == "linear":  # chip-feasible: one weight vector + bias per action, plus a 'none' class
            from sklearn.linear_model import LogisticRegression
            Xtr = np.concatenate([ex_vecs[a] for a in names] + ([neg_vecs] if neg_vecs is not None else []))
            ytr = np.concatenate([np.full(len(ex_vecs[a]), i) for i, a in enumerate(names)]
                                 + ([np.full(len(neg_vecs), len(names))] if neg_vecs is not None else []))
            clf = LogisticRegression(C=10.0, max_iter=5000).fit(Xtr, ytr)

        def scores(texts):
            v = emb(texts)
            if score == "linear":
                P = clf.predict_proba(v)
                S = P[:, :len(names)].copy()
                if neg_vecs is not None:  # a 'none' win zeroes the action scores
                    S[P[:, len(names)] >= P[:, :len(names)].max(1)] = 0.0
                return S
            cols = []
            for a in names:
                s = v @ ex_vecs[a].T
                if score in ("max", "knn_neg"):
                    cols.append(s.max(1))
                elif score == "top2":
                    k = min(2, s.shape[1]); cols.append(np.sort(s, 1)[:, -k:].mean(1))
                else:
                    c = ex_vecs[a].mean(0); c /= np.linalg.norm(c); cols.append(v @ c)
            S = np.stack(cols, 1)
            if score == "knn_neg" and neg_vecs is not None:  # nearest neighbour is a negative: reject
                S[(v @ neg_vecs.T).max(1) >= S.max(1)] = 0.0
            return S

        pr = [r for r in para if r["product"] == p]
        S = scores([r["text"] for r in pr])
        y = np.array([names.index(r["action"]) for r in pr])
        split = np.array([r["split"] for r in pr])
        NS = scores([r["text"] for r in nonc])
        nbest = NS.max(1)
        ncat = np.array([r["cat"] for r in nonc]); nsplit = np.array([r["split"] for r in nonc])
        head = np.isin(ncat, HEADLINE)
        dev_n = np.sort(nbest[head & (nsplit == "dev")])
        # lowest threshold rejecting >= 95% of dev headline non-commands (reject if best < thr)
        k = int(np.ceil(0.95 * len(dev_n)))
        thr = float(dev_n[k - 1] + 1e-6) if k <= len(dev_n) else float(dev_n[-1] + 1e-6)
        best, arg = S.max(1), S.argmax(1)
        res = {"threshold": thr}
        for sp in ("dev", "test"):
            mk = split == sp
            correct = (arg == y) & (best >= thr)
            res[f"{sp}_acc"] = float(correct[mk].mean())
            res[f"{sp}_acc_norej"] = float((arg == y)[mk].mean())
            res[f"{sp}_n"] = int(mk.sum())
            for cat in ("chat", "near", "tv", "other_device"):
                cm = (ncat == cat) & (nsplit == sp)
                res[f"{sp}_rej_{cat}"] = float((nbest[cm] < thr).mean())
            hm = head & (nsplit == sp)
            res[f"{sp}_rej_headline"] = float((nbest[hm] < thr).mean())
        # per-action test accuracy and the errors
        mt = split == "test"
        res["per_action_test"] = {a: float((((arg == y) & (best >= thr))[mt & (y == i)]).mean()) for i, a in enumerate(names)}
        errs = []
        for i in np.where(mt & ~((arg == y) & (best >= thr)))[0]:
            errs.append(f"{pr[i]['action']:>11} -> {'REJECT' if best[i] < thr else names[arg[i]]:>11} {best[i]:.3f}  {pr[i]['text']}")
        res["test_errors"] = errs
        fa = []
        for i in np.where(head & (nsplit == "test") & (nbest >= thr))[0]:
            fa.append(f"{ncat[i]:>5} -> {names[NS[i].argmax()]:>11} {nbest[i]:.3f}  {nonc[i]['text']}")
        res["test_false_accepts"] = fa
        # trade-off: test acc vs test headline rejection across thresholds
        curve = []
        for t in np.linspace(0.3, 0.95, 66):
            curve.append((round(float(t), 3), float(((arg == y) & (best >= t))[mt].mean()),
                          float((nbest[head & (nsplit == "test")] < t).mean())))
        res["curve_test"] = curve
        out["products"][p] = res
    a = [out["products"][p]["test_acc"] for p in out["products"]]
    r = [out["products"][p]["test_rej_headline"] for p in out["products"]]
    out["test_acc_mean"], out["test_rej_mean"] = float(np.mean(a)), float(np.mean(r))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--extra")
    ap.add_argument("--score", default="max", choices=["max", "top2", "centroid", "knn_neg", "linear"])
    ap.add_argument("--out")
    ap.add_argument("--errors", action="store_true")
    args = ap.parse_args()
    allres = []
    for mn in args.models:
        o = run(mn, args.extra, args.score)
        allres.append(o)
        for p, r in o["products"].items():
            print(f"{mn:<40} {p:<9} {args.score:<8} thr={r['threshold']:.3f}  unseen acc test={r['test_acc']:.3f} "
                  f"(no-reject {r['test_acc_norej']:.3f})  reject test chat/near/tv={r['test_rej_chat']:.2f}/"
                  f"{r['test_rej_near']:.2f}/{r['test_rej_tv']:.2f} headline={r['test_rej_headline']:.3f} "
                  f"other_dev={r['test_rej_other_device']:.2f}", flush=True)
            if args.errors:
                print("  per action:", {k: round(v, 2) for k, v in r["per_action_test"].items()})
                print("  errors:\n   " + "\n   ".join(r["test_errors"]))
                print("  false accepts:\n   " + "\n   ".join(r["test_false_accepts"]))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(allres, open(args.out, "w"), indent=1)
