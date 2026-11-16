#!/usr/bin/env bash
# Model v2, float stage (model agent, iteration 2): after the teacher-v4 chain (targets, prototypes), train f6 from
# v1.1's float model with the decision loss in teacher-v4's space, then the float evaluation on eval/text (clean and
# 10 dB) and eval/text_v2. In parallel (CPU): model v1's integer golden on the v2 audio, for the v1-vs-v2 table.
cd /d/marigold
PY=/d/marigold/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
for i in $(seq 1 540); do grep -q "t4 final chain finished" data/logs/t4_gate.log 2>/dev/null && break; sleep 10; done
grep -q "t4 final chain finished" data/logs/t4_gate.log || { echo "teacher chain not finished in 90 min" > data/logs/v2a.log; exit 1; }
[ -f data/intent/targets-teacher-v4.npz ] && [ -f data/intent/protos-teacher-v4.npz ] || { echo "targets/protos missing" > data/logs/v2a.log; exit 1; }
MIN=${MIN:-150}
( OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY sw/intent/evaluate.py --model data/intent/qv1.pt --golden data/golden/mg-v1.npz \
    --tag golden-v1-v2 --teacher data/models/teacher-v3 --conds clean,v2 --enroll 2>&1 | grep -v "Warning\|Loading\|get_sentence" > data/logs/v2_eval_v1golden.log ) &
bash sw/intent/run_resume.sh f6 data/intent/f5-final.pt $(( $(date +%s) + MIN * 60 )) --targets data/intent/targets-teacher-v4.npz \
     --protos data/intent/protos-teacher-v4.npz --ppb 2 --lr 4e-4 --ctc-w 1.0 --mix 0.5,0.15,0.25,0.10 --noise-p 0.8 --snr-hi 20 \
     --steps 4000000 --workers 8 --gpu-gb 10 --bs 64 --dec-w 1.0 --dec-tau 0.05 --dec-m 0.05 --aux-w 1.0
cp data/intent/f6.pt data/intent/f6-final.pt
$PY sw/intent/evaluate.py --model data/intent/f6-final.pt --tag f6 --teacher data/models/teacher-v4 --conds clean,fan10db,microwave10db,v2 --enroll 2>&1 \
    | grep -v "Warning\|Loading\|get_sentence" > data/logs/v2_eval_f6.log
wait
echo "v2a finished $(date)" >> data/logs/v2_eval_f6.log
