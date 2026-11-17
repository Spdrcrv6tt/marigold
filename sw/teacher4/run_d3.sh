#!/usr/bin/env bash
# Review D3: the gate-0 results of Nov 16 re-run with the current sw/teacher4/text_gate.py so the committed JSON has
# its layout (same protocol: eval/text dev calibration, no reject bank; eval/text test and the now-seen eval/text_v2).
cd /d/marigold
PY=/d/marigold/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
for t in teacher-v4 teacher-v3; do
  $PY sw/teacher4/text_gate.py data/models/$t --cal v1 --sets v1,v2 --out reports/gate0/text_gate_${t/-/_}.json 2>&1 | grep "sdk acc" >> data/logs/d3_gate.log
done
echo "d3 finished $(date +%H:%M)" >> data/logs/d3_gate.log
