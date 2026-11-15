#!/bin/bash
# Rebuild the eval speech set end to end: voices, jobs, synthesis, manifest, ASR QA, noisy copies.
set -e
R=/d/marigold; cd $R; PY=".venv/Scripts/python.exe -I"
rm -rf data/tts/eval data/tts/eval_noisy data/tts/done/jobs_eval.*
$PY sw/data/tts_build.py voices && $PY sw/data/tts_build.py jobs
sw/data/run_synth.sh data/tts/jobs_eval.jsonl 4 2 0
$PY sw/data/tts_build.py merge
HF_HOME=$R/data/hf $PY sw/data/tts_asr_check.py data/tts/eval/manifest.jsonl data/tts/eval/asr_check.json cuda | head -5
for n in fan microwave; do $PY sw/data/mix.py data/tts/eval/manifest.jsonl $n 10 data/tts/eval_noisy/${n}10db; done
