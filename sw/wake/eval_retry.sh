#!/bin/bash
# kws_eval.py with up to 3 attempts (sporadic CUDA faults on the shared GPU). usage: eval_retry.sh OUT.json <args>
out=$1; shift
for i in 1 2 3; do
  .venv/Scripts/python.exe -I sw/wake/kws_eval.py "$@" > ${out%.json}.log 2>&1 && [ -f $out ] && break
  sleep 5
done
