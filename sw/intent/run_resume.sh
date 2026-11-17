#!/usr/bin/env bash
# Keeps a training run going until a deadline: after a crash (the CUDA "illegal memory access" seen 3 times on this
# shared GPU) it restarts from the last checkpoint (saved every 1000 steps) with --resume, so the LR schedule and the
# decision threshold continue (review S1, S2). Exit 0 only when an attempt finished; 1 when every attempt crashed
# before the deadline; 3 on the throughput abort (review S5).
# Usage: run_resume.sh TAG INIT DEADLINE_EPOCH ARGS...   (MG_ROOT, MG_PY override the tree and interpreter for tests)
TAG=$1; INIT=$2; DL=$3; shift 3
cd ${MG_ROOT:-/d/marigold}
PY=${MG_PY:-/d/marigold/.venv/Scripts/python.exe}
n=0; ok=0
while [ $(date +%s) -lt $((DL - 120)) ]; do
  MIN=$(( (DL - $(date +%s)) / 60 ))
  RES=""
  if [ -f data/intent/$TAG.pt ] && [ $n -gt 0 ]; then INIT=data/intent/$TAG-resume.pt; cp data/intent/$TAG.pt $INIT; RES="--resume"; fi
  echo "=== start $n at $(date +%H:%M) init $INIT minutes $MIN $RES" >> data/logs/train_$TAG.log
  PYTHONIOENCODING=utf-8 $PY sw/intent/train.py --tag $TAG --init "$INIT" --minutes $MIN $RES "$@" >> data/logs/train_$TAG.log 2>&1
  rc=$?
  [ $rc -eq 0 ] && { ok=1; break; }
  [ $rc -eq 3 ] && { echo "=== throughput abort at $(date +%H:%M)" >> data/logs/train_$TAG.log; exit 3; }
  n=$((n + 1))
done
echo "=== finished at $(date +%H:%M) after $n restarts, success $ok" >> data/logs/train_$TAG.log
[ $ok -eq 1 ] || exit 1
