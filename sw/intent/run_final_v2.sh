#!/usr/bin/env bash
# Model v2 freeze pipeline (model agent, iteration 2): QAT from f6 with the decision loss (teacher-v4 space), integer
# export mg-v2, exactness tests, budget, integer golden evaluation (eval/text clean + 10 dB and eval/text_v2), test
# vectors with the dev-chosen microwave blob. Logs data/logs/final_v2.log.
cd /d/marigold
PY=/d/marigold/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
L=data/logs/final_v2.log
FLOAT=data/intent/f6-final.pt
echo "=== QAT from $FLOAT $(date +%H:%M)" >> $L
bash sw/intent/run_resume.sh qv2 $FLOAT $(( $(date +%s) + 40 * 60 )) --qat --float-ref $FLOAT --targets data/intent/targets-teacher-v4.npz \
     --protos data/intent/protos-teacher-v4.npz --ppb 2 --lr 1e-4 --lr-scale 5e-4 --fw 0.5 --ctc-w 1.0 --steps 400000 --workers 8 --gpu-gb 8 \
     --bs 64 --dec-w 1.0 --dec-tau 0.05 --dec-m 0.05 --mix 0.5,0.15,0.25,0.10 --noise-p 0.8
echo "=== export $(date +%H:%M)" >> $L
$PY sw/golden/intent_export.py data/intent/qv2.pt --tag v2 --calib 400 >> $L 2>&1
echo "=== tests $(date +%H:%M)" >> $L
$PY sw/golden/intent_test.py --tag v2 --n 6 >> $L 2>&1
$PY sw/intent/budget.py --tag v2 >> $L 2>&1
echo "=== golden eval $(date +%H:%M)" >> $L
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY sw/intent/evaluate.py --model data/intent/qv2.pt --golden data/golden/mg-v2.npz \
    --tag golden-v2 --teacher data/models/teacher-v4 --conds clean,fan10db,microwave10db,v2 --enroll 2>&1 | grep -v "Warning\|Loading\|get_sentence" >> $L
CH=$($PY -c "import json;print(json.load(open('reports/p0/eval-golden-v2.json'))['products']['microwave']['chosen'])")
echo "=== vectors (microwave blob: $CH) $(date +%H:%M)" >> $L
$PY sw/golden/intent_vectors.py --tag v2 --blob "data/sdk/eval-golden-v2/microwave-$CH.mgc" >> $L 2>&1
echo "=== done $(date +%H:%M)" >> $L
