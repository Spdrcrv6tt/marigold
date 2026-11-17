#!/usr/bin/env bash
# Milestone (c) (model agent, iteration 2; John/coordinator approved): an audio teacher = the full 6-layer Moonshine
# Streaming Tiny encoder (MIT) + the same head, fine-tuned into teacher-v4's chip space with the decision loss (PC
# only, not a chip model); then the unchanged 2-layer chip student (f6) distilled from it (audio-to-audio KD) on
# LibriSpeech (all training subsets present, random crops, no text) plus the TTS and Speech Commands, keeping the
# text distillation, decision and CTC losses; QAT, export v3, tests, integer evaluation with the reject fix
# (calibration on eval/text_dev2, reject bank), vectors. Eval voices and eval text are held out as before.
# Waits for the v2 freeze pipeline so the GPU is never shared with it. Log markers: data/logs/c_chain.log.
cd /d/marigold
PY=/d/marigold/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
L=data/logs/c_chain.log
TV4="--targets data/intent/targets-teacher-v4.npz --protos data/intent/protos-teacher-v4.npz"
AUG="--ppb 2 --ctc-w 1.0 --noise-p 0.8 --snr-hi 20 --steps 4000000 --bs 64 --dec-w 1.0 --gpu-gb 18 --min-spm 100"
FIX="--cal dev2 --bank data/sdk/reject_bank-teacher-v4.npz --bank-cfg reports/gate0/text_gate_teacher_v4_fix.json"
waitfor() { for i in $(seq 1 $3); do grep -q "$2" $1 2>/dev/null && return 0; sleep 10; done; echo "=== gave up waiting for '$2' in $1 $(date +%H:%M)" >> $L; return 1; }
waitfor data/logs/final_v2.log "=== done" 1080 || exit 1
echo "=== at6 (6-layer audio teacher) start $(date +%H:%M)" >> $L
bash sw/intent/run_resume.sh at6 "" $(( $(date +%s) + 150 * 60 )) --moonshine --nl 6 --fut 4,4,0,0,4,4 $TV4 $AUG --lr 1.5e-3 --lr-pre 0.3 \
     --mix 0.5,0.15,0.25,0.10 --workers 8 || { echo "=== at6 failed rc=$? $(date +%H:%M)" >> $L; exit 1; }
cp data/intent/at6.pt data/intent/at6-final.pt
echo "=== at6 eval $(date +%H:%M)" >> $L
$PY sw/intent/evaluate.py --model data/intent/at6-final.pt --tag at6 --teacher data/models/teacher-v4 --conds clean,v2 --enroll 2>&1 \
    | grep -v "Warning\|Loading\|get_sentence" > data/logs/c_eval_at6.log
waitfor data/logs/fix_gate.log "fix gate finished" 360 && waitfor data/logs/tts_dev2v3.log "tts dev2v3 finished" 360 || exit 1
echo "=== f6 eval with the reject fix $(date +%H:%M)" >> $L
$PY sw/intent/evaluate.py --model data/intent/f6-final.pt --tag f6-fix --teacher data/models/teacher-v4 --conds clean,v2,v3,dev2 --enroll $FIX 2>&1 \
    | grep -v "Warning\|Loading\|get_sentence" > data/logs/c_eval_f6fix.log
echo "=== f7 (student distilled from at6) start $(date +%H:%M)" >> $L
bash sw/intent/run_resume.sh f7 data/intent/f6-final.pt $(( $(date +%s) + 180 * 60 )) --teacher-audio data/intent/at6-final.pt --akd-w 1.0 --lkd 0.25 \
     $TV4 $AUG --lr 3e-4 --mix 0.45,0.12,0.13,0.05 --workers 10 || { echo "=== f7 failed rc=$? $(date +%H:%M)" >> $L; exit 1; }
cp data/intent/f7.pt data/intent/f7-final.pt
echo "=== f7 eval with the reject fix $(date +%H:%M)" >> $L
$PY sw/intent/evaluate.py --model data/intent/f7-final.pt --tag f7-fix --teacher data/models/teacher-v4 --conds clean,fan10db,microwave10db,v2,v3,dev2 --enroll $FIX 2>&1 \
    | grep -v "Warning\|Loading\|get_sentence" > data/logs/c_eval_f7fix.log
F=data/logs/final_v3.log
echo "=== QAT from data/intent/f7-final.pt $(date +%H:%M)" >> $F
bash sw/intent/run_resume.sh qv3 data/intent/f7-final.pt $(( $(date +%s) + 40 * 60 )) --qat --float-ref data/intent/f7-final.pt $TV4 $AUG --lr 1e-4 \
     --lr-scale 5e-4 --fw 0.5 --teacher-audio data/intent/at6-final.pt --akd-w 1.0 --lkd 0.25 --mix 0.45,0.12,0.13,0.05 --workers 10
echo "=== export $(date +%H:%M)" >> $F
$PY sw/golden/intent_export.py data/intent/qv3.pt --tag v3 --calib 400 >> $F 2>&1
echo "=== tests $(date +%H:%M)" >> $F
$PY sw/golden/intent_test.py --tag v3 --n 6 >> $F 2>&1
$PY sw/intent/budget.py --tag v3 >> $F 2>&1
echo "=== golden eval $(date +%H:%M)" >> $F
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY sw/intent/evaluate.py --model data/intent/qv3.pt --golden data/golden/mg-v3.npz \
    --tag golden-v3-fix --teacher data/models/teacher-v4 --conds clean,fan10db,microwave10db,v2,v3,dev2 --enroll $FIX 2>&1 | grep -v "Warning\|Loading\|get_sentence" >> $F
CH=$($PY -c "import json;print(json.load(open('reports/p0/eval-golden-v3-fix.json'))['products']['microwave']['chosen'])")
echo "=== vectors (microwave blob: $CH) $(date +%H:%M)" >> $F
$PY sw/golden/intent_vectors.py --tag v3 --blob "data/sdk/eval-golden-v3-fix/microwave-$CH.mgc" >> $F 2>&1
echo "=== done $(date +%H:%M)" >> $F
echo "=== chain finished $(date +%H:%M)" >> $L
