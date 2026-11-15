#!/bin/bash
# usage: run_synth.sh <jobs.jsonl> <piper_shards> <kokoro_cuda_shards> [kokoro_cpu_shards]
# Runs all shards in parallel, logs to data/logs/synth_<jobs>_<engine>_<i>.log, waits for all.
R=/d/marigold; J=$1; NP=$2; NKG=$3; NKC=${4:-0}; NK=$((NKG+NKC)); s=$(basename $J .jsonl)
cd $R
for i in $(seq 0 $((NP-1))); do PIPER_THREADS=3 .venv/Scripts/python.exe -I sw/data/tts_build.py synth $J piper $i $NP > data/logs/synth_${s}_piper_$i.log 2>&1 & done
for i in $(seq 0 $((NK-1))); do dev=cuda; [ $i -ge $NKG ] && dev=cpu; KOKORO_THREADS=4 .venv/Scripts/python.exe -I sw/data/tts_build.py synth $J kokoro $i $NK $dev > data/logs/synth_${s}_kokoro_$i.log 2>&1 & done
wait
grep -h ' done:' data/logs/synth_${s}_*.log
