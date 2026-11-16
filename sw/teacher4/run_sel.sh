#!/usr/bin/env bash
# Teacher v4 selection (model agent, Nov 16): candidates trained without the 10 held-out TRAINING products (same as
# v3), scored on them in the chip space (sw/teacher4/sel_t4.py); never on eval/. Logs data/logs/t4_*.log.
cd /d/marigold
PY=/d/marigold/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
HOLD="humidifier,soundbar,treadmill,garage door,rice cooker,massage chair,security camera,thermostat,display turntable,mini split heat pump"
M=data/gate0b/models
run() { n=$1; shift; $PY sw/teacher4/finetune_t4.py "$@" --holdout "$HOLD" > data/logs/t4_ft_$n.log 2>&1; $PY sw/teacher4/sel_t4.py $M/t4-$n --holdout "$HOLD" --out reports/gate0/t4_sel_$n.json 2>&1 | grep "held-out" >> data/logs/t4_sel.log; }
[ -f reports/gate0/t4_sel_v3cand-small.json ] || $PY sw/teacher4/sel_t4.py $M/v3cand-small --holdout "$HOLD" --out reports/gate0/t4_sel_v3cand-small.json 2>&1 | grep "held-out" >> data/logs/t4_sel.log
run q06 Qwen/Qwen3-Embedding-0.6B $M/t4-q06 --steps 3000 --lr 2e-5
run bgel BAAI/bge-large-en-v1.5 $M/t4-bgel --steps 3000 --lr 2e-5
run q4 Qwen/Qwen3-Embedding-4B $M/t4-q4 --steps 1500 --lora 16 --lr 1e-4
[ -f data/logs/t4_skip_q8 ] || run q8 Qwen/Qwen3-Embedding-8B $M/t4-q8 --steps 1200 --lora 16 --lr 1e-4
echo "selection chain finished $(date)" >> data/logs/t4_sel.log
