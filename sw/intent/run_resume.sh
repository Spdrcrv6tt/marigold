#!/usr/bin/env bash
# Keeps a training run going until a deadline: after a crash (the CUDA "illegal memory access" seen 3 times on this
# shared GPU) it restarts from the last checkpoint (saved every 1000 steps). Usage: run_resume.sh TAG INIT DEADLINE_EPOCH ARGS...
TAG=$1; INIT=$2; DL=$3; shift 3
cd /d/marigold
PY=/d/marigold/.venv/Scripts/python.exe
n=0
while [ $(date +%s) -lt $((DL - 120)) ]; do
  MIN=$(( (DL - $(date +%s)) / 60 ))
  [ -f data/intent/$TAG.pt ] && [ $n -gt 0 ] && INIT=data/intent/$TAG-resume.pt && cp data/intent/$TAG.pt $INIT
  echo "=== start $n at $(date +%H:%M) init $INIT minutes $MIN" >> data/logs/train_$TAG.log
  PYTHONIOENCODING=utf-8 $PY sw/intent/train.py --tag $TAG --init $INIT --minutes $MIN "$@" >> data/logs/train_$TAG.log 2>&1 && break
  n=$((n + 1))
done
echo "=== finished at $(date +%H:%M) after $n restarts" >> data/logs/train_$TAG.log
