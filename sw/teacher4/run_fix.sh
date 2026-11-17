#!/usr/bin/env bash
# Reject fix (model agent, iteration 2): generic reject bank in teacher-v4's chip space, then the text gate with
# calibration on eval/text_dev2 only (bank settings chosen per product on dev2), scored on eval/text_v2 (seen) and,
# once, on the blind eval/text_v3. Logs data/logs/fix_*.log.
cd /d/marigold
PY=/d/marigold/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
$PY sw/teacher4/make_reject_bank.py data/models/teacher-v4 > data/logs/fix_bank.log 2>&1
$PY sw/teacher4/text_gate.py data/models/teacher-v4 --cal dev2 --sets v2,v3 --bank data/sdk/reject_bank-teacher-v4.npz --grid \
    --out reports/gate0/text_gate_teacher_v4_fix.json > data/logs/fix_gate.log 2>&1
echo "fix gate finished $(date)" >> data/logs/fix_gate.log
