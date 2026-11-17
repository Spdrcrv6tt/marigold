#!/usr/bin/env bash
# Post-merge reruns (review S6, S7; Nov 17): integer golden models v1, v2, v3 scored once each with the fixed code
# (frozen speech bank, pure rejection-off metric) under the reject-fix protocol (dev2 calibration, reject bank with the
# per-product settings chosen by the text gate on dev2; v1 uses a teacher-v3 bank with the same settings, no tuning).
cd /d/marigold
PY=/d/marigold/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
set -o pipefail
L=data/logs/postmerge.log
CONDS=clean,fan10db,microwave10db,v2,v3,dev2
CFG=reports/gate0/text_gate_teacher_v4_fix.json
[ -f data/sdk/reject_bank-teacher-v3.npz ] || $PY sw/teacher4/make_reject_bank.py data/models/teacher-v3 >> $L 2>&1 || { echo "=== failed: bank-v3" >> $L; exit 1; }
for m in "v1 teacher-v3" "v2 teacher-v4" "v3 teacher-v4"; do
  set -- $m
  echo "=== $1 $(date +%H:%M)" >> $L
  $PY sw/intent/evaluate.py --model data/intent/q$1.pt --golden data/golden/mg-$1.npz --tag pm-golden-$1 --teacher data/models/$2 \
      --conds $CONDS --enroll --cal dev2 --bank data/sdk/reject_bank-$2.npz --bank-cfg $CFG 2>&1 | grep -v "Warning\|Loading\|get_sentence" >> $L \
      || { echo "=== failed: $1" >> $L; exit 1; }
done
echo "=== done $(date +%H:%M)" >> $L
