#!/usr/bin/env bash
# Nov 17: the v3 integer eval once more with the deterministic SDK embedding (sdk/compile.py STEmbedder), same
# reject-fix protocol and frozen speech bank as run_postmerge.sh, no tuning; settles v3's fan variant.
cd /d/marigold
PY=/d/marigold/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
set -o pipefail
$PY sw/intent/evaluate.py --model data/intent/qv3.pt --golden data/golden/mg-v3.npz --tag pm-golden-v3-det --teacher data/models/teacher-v4 \
    --conds clean,fan10db,microwave10db,v2,v3,dev2 --enroll --cal dev2 --bank data/sdk/reject_bank-teacher-v4.npz \
    --bank-cfg reports/gate0/text_gate_teacher_v4_fix.json 2>&1 | grep -v "Warning\|Loading\|get_sentence\|warn" > data/logs/v3det.log \
    && echo "=== done $(date +%H:%M)" >> data/logs/v3det.log || echo "=== failed $(date +%H:%M)" >> data/logs/v3det.log
