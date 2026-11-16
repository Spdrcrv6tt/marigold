#!/bin/bash
# Resume a wake training run after sporadic CUDA faults (seen twice on the shared GPU): up to 6 attempts.
# usage: sw/wake/train_loop.sh OUT_DIR <train.py args without --resume>
out=$1; shift
for i in 1 2 3 4 5 6; do
  .venv/Scripts/python.exe -I sw/wake/train.py "$@" --resume >> $out/train.log 2>&1
  grep -q "done best PER" $out/train.log && break
  echo "== restart $i after a crash" >> $out/train.log
  sleep 5
done
