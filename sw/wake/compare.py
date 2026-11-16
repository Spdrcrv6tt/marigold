"""Compare result files at the dev-chosen operating point: pooled dev false wakes <= target per 24 h (mean over phrases)."""
import json, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0].rsplit("\\", 1)[0])
from summarize import pick, pooled_fa, CONDS
target = float(sys.argv[1]); files = sys.argv[2:]
for ph in (["hey breeze", "hey kitchen"], ["hey breeze", "hey kitchen", "hey marigold", "okay toaster", "hello lumen", "hey computer"]):
    print("phrases:", ", ".join(ph))
    for f in files:
        r = json.load(open(f)); op = pick(r, ph, target)
        if op is None:
            print(f"  {f}: no operating point"); continue
        k = r["sens"].index(op)
        det = {c: sum(r["pos"][c][p]["det"][k] for p in ph) / len(ph) for c in CONDS}
        fa = sum(pooled_fa(r, k, r["phrases"].index(p)) for p in ph) / len(ph)
        print(f"  {f.split('/')[-1]}: sens {op}, FA {fa:.2f}/24h, det " + ", ".join(f"{c} {100 * v:.1f}%" for c, v in det.items()))
