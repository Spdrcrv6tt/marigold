#!/usr/bin/env bash
# Teacher v3 selection (model agent, Nov 15): two candidates trained without the held-out TRAINING products,
# scored on them (sw/gate0b/val_heldout.py); never on eval/. Run from Git Bash.
set -u
cd /d/marigold
PY=/d/marigold/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
HOLD="humidifier,soundbar,treadmill,garage door,rice cooker,massage chair,security camera,thermostat,display turntable,mini split heat pump"
curl -s http://localhost:11434/api/generate -d '{"model":"qwen2.5:14b","keep_alive":0}' > /dev/null
curl -s http://localhost:11434/api/generate -d '{"model":"qwen2.5:32b-instruct-q4_K_M","keep_alive":0}' > /dev/null
$PY sw/gate0b/finetune_text_v3.py BAAI/bge-small-en-v1.5 data/gate0b/models/v3cand-small --steps 2500 --tau 0.1 --libri 32 --holdout "$HOLD" > data/logs/ft_v3cand_small.log 2>&1 &
P1=$!
$PY sw/gate0b/finetune_text_v3.py BAAI/bge-base-en-v1.5 data/gate0b/models/v3cand-base --steps 2500 --tau 0.1 --libri 32 --holdout "$HOLD" > data/logs/ft_v3cand_base.log 2>&1 &
P2=$!
wait $P1 $P2
$PY sw/gate0b/val_heldout.py data/gate0b/models/v3cand-small data/gate0b/models/v3cand-base BAAI/bge-small-en-v1.5 --holdout "$HOLD" --dims 128,256 --out reports/gate0/teacher_v3_selection.json > data/logs/val_v3.log 2>&1
grep -v "Batches\|Loading\|Warning" data/logs/val_v3.log
