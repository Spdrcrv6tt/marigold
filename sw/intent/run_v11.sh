#!/usr/bin/env bash
# v1.1 accuracy run (coordinator, Nov 15, 10 PM): continue f4 with an LR restart, more command audio, stronger noise;
# then QAT, export, tests, vectors and the integer eval, and a dev-based comparison with v1. Unattended.
cd /d/marigold
PY=/d/marigold/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
bash sw/intent/run_resume.sh f5 data/intent/f4-final.pt $(date -d "tomorrow 04:15" +%s) --targets data/intent/targets-teacher-v3.npz \
     --ppb 2 --lr 2e-4 --ctc-w 1.0 --mix 0.55,0.15,0.20,0.10 --noise-p 0.85 --snr-hi 15 --steps 4000000 --workers 8 --gpu-gb 8 --bs 64
cp data/intent/f5.pt data/intent/f5-final.pt
bash sw/intent/run_final.sh data/intent/f5-final.pt $(date -d "tomorrow 04:55" +%s) v11
$PY - <<'PY' > reports/p0/v11-decision.txt 2>&1
import json
def dev(tag):
    r = json.load(open(f"reports/p0/eval-golden-{tag}.json"))
    return {p: (v["chosen"], v["variants"][v["chosen"]]["dev_unseen_acc"]) for p, v in r["products"].items()}
a, b = dev("v1"), dev("v11")
ma = sum(x[1] for x in a.values()) / len(a); mb = sum(x[1] for x in b.values()) / len(b)
print("dev unseen accuracy at the 95%-rejection operating point (chosen matching variant per product)")
print("v1 ", a, round(ma, 4)); print("v1.1", b, round(mb, 4))
print("DECISION:", "v1.1 beats v1 on dev -> tag v11" if mb > ma else "v1 stands")
PY
