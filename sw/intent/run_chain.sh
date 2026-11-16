#!/usr/bin/env bash
# After the float run (f4) finishes: float evaluation (milestone c) in parallel with QAT + export + tests + golden eval.
cd /d/marigold
until grep -q "=== finished" data/logs/train_f4.log 2>/dev/null; do sleep 20; done
cp data/intent/f4.pt data/intent/f4-final.pt
PYTHONIOENCODING=utf-8 /d/marigold/.venv/Scripts/python.exe sw/intent/evaluate.py --model data/intent/f4-final.pt --tag f4-final --enroll 2>&1 | grep -v "Warning\|Loading\|get_sentence" > data/logs/eval_f4_final.log &
bash sw/intent/run_final.sh data/intent/f4-final.pt $(date -d "today 23:00" +%s) v1
wait
