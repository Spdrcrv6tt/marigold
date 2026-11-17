#!/usr/bin/env bash
# Milestone (c) ship check: model v2's integer golden under the reject-fix protocol (calibration on eval/text_dev2,
# reject bank, per-product bank settings from the text gate), the same protocol as qv3's golden-v3-fix evaluation.
cd /d/marigold
PY=/d/marigold/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
set -o pipefail
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY sw/intent/evaluate.py --model data/intent/qv2.pt --golden data/golden/mg-v2.npz \
    --tag golden-v2-fix --teacher data/models/teacher-v4 --conds clean,fan10db,microwave10db,v2,v3,dev2 --enroll \
    --cal dev2 --bank data/sdk/reject_bank-teacher-v4.npz --bank-cfg reports/gate0/text_gate_teacher_v4_fix.json 2>&1 \
    | grep -v "Warning\|Loading\|get_sentence" > data/logs/v2fix_eval.log && echo "v2fix finished $(date +%H:%M)" >> data/logs/v2fix_eval.log \
    || echo "v2fix failed $(date +%H:%M)" >> data/logs/v2fix_eval.log
