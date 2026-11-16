#!/usr/bin/env bash
# v1 freeze pipeline (model agent): QAT from the float model, integer export, exactness tests, budget, and the
# golden (integer) evaluation on synthetic eval audio. Usage: run_final.sh FLOAT_CKPT QAT_DEADLINE_EPOCH TAG
FLOAT=$1; DL=$2; TAG=$3
cd /d/marigold
PY=/d/marigold/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
L=data/logs/final_$TAG.log
echo "=== QAT from $FLOAT until $(date -d @$DL +%H:%M)" >> $L
bash sw/intent/run_resume.sh q$TAG $FLOAT $DL --qat --float-ref $FLOAT --targets data/intent/targets-teacher-v3.npz \
     --ppb 2 --lr 1e-4 --lr-scale 5e-4 --fw 0.5 --steps 400000 --workers 8 --gpu-gb 8 --bs 64
echo "=== export $(date +%H:%M)" >> $L
$PY sw/golden/intent_export.py data/intent/q$TAG.pt --tag $TAG --calib 400 >> $L 2>&1
echo "=== tests $(date +%H:%M)" >> $L
$PY sw/golden/intent_test.py --tag $TAG --n 6 >> $L 2>&1
$PY sw/intent/budget.py --tag $TAG >> $L 2>&1
echo "=== golden eval $(date +%H:%M)" >> $L
$PY sw/intent/evaluate.py --model data/intent/q$TAG.pt --golden data/golden/mg-$TAG.npz --tag golden-$TAG --enroll 2>&1 | grep -v "Warning\|Loading\|get_sentence" >> $L
echo "=== done $(date +%H:%M)" >> $L
