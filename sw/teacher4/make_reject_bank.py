"""Generic reject bank for the SDK (iteration 2 reject fix): chip-space unit vectors (golden-format 5) of
  * every command of the 120 training appliances (data/gate0b/train_text_v4.json; never fan or microwave),
  * their non-command lines (chat, near misses, other-device commands written for those appliances),
  * 3,000 LibriSpeech training transcript snippets (2 to 10 words).
Lines identical to any eval/text, eval/text_v2, eval/text_dev2 or eval/text_v3 line are dropped. The SDK picks
action-0 prototypes from it per product (sdk/compile.py --reject-bank, --bank-protos, --bank-keep).

  python sw/teacher4/make_reject_bank.py data/models/teacher-v4  ->  data/sdk/reject_bank-teacher-v4.npz
"""
import json, os, random, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf"))
sys.path.insert(0, str(ROOT / "sw")); sys.path.insert(0, str(ROOT / "sdk"))
from textfilter import EvalFilter, norm  # noqa: E402
import compile as C  # noqa: E402


def main():
    tp = ROOT / sys.argv[1]
    flt = EvalFilter()
    for d in ("eval/text_dev2", "eval/text_v3"):
        for f in (ROOT / d).glob("*.txt"):
            for line in open(f, encoding="utf8"):
                if line.strip() and not line.startswith("#"):
                    flt.ev.add(norm(line.partition(" | ")[0]))
    tt = json.load(open(ROOT / "data/gate0b/train_text_v4.json", encoding="utf8"))
    rows = []
    for p, r in tt.items():
        if p in ("fan", "microwave"):
            continue
        rows += [(t, "cmd", p) for v in r["actions"].values() for t in v]
        rows += [(t, "none", p) for t in r["none"]]
    rng = random.Random(0)
    libri = []
    for f in sorted((ROOT / "data/librispeech/LibriSpeech/train-clean-100").rglob("*.trans.txt")):
        for line in open(f, encoding="utf8"):
            w = line.split()[1:]
            if len(w) >= 2:
                n = rng.randint(2, min(10, len(w))); s = rng.randint(0, len(w) - n)
                libri.append(" ".join(w[s:s + n]).lower())
    rows += [(t, "libri", "") for t in rng.sample(libri, 3000)]
    seen, keep = set(), []
    for t, k, p in rows:
        n_ = norm(t)
        if n_ and n_ not in seen and flt.ok(t):
            seen.add(n_); keep.append((t, k, p))
    emb = C.STEmbedder(tp)
    z = np.load(tp / "marigold_proj.npz")
    V = C.unit(C.golden_vec(emb.encode([t for t, _, _ in keep]), z["mu"].astype(np.float64), z["P"].astype(np.float64)))
    out = ROOT / f"data/sdk/reject_bank-{tp.name}.npz"
    np.savez(out, texts=np.array([t for t, _, _ in keep], dtype=object), kind=np.array([k for _, k, _ in keep]),
             product=np.array([p for _, _, p in keep]), V=V.astype(np.float32), teacher=str(tp.name))
    print(out, V.shape, {k: sum(1 for _, x, _ in keep if x == k) for k in ("cmd", "none", "libri")}, "eval-identical dropped", flt.dropped, flush=True)


if __name__ == "__main__":
    main()
