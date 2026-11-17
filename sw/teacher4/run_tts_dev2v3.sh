#!/usr/bin/env bash
# TTS of eval/text_dev2 (calibration DEV) and eval/text_v3 (blind TEST) with the EVAL voices, CPU only.
cd /d/marigold
PY=/d/marigold/.venv/Scripts/python.exe
for n in dev2 v3; do $PY -I sw/data/tts_build.py jobs_eval $n >> data/logs/tts_dev2v3.log 2>&1; bash sw/data/run_synth.sh data/tts/jobs_eval_$n.jsonl 6 0 3 >> data/logs/tts_dev2v3.log 2>&1; done
$PY -I sw/data/tts_build.py merge eval_dev2 eval_v3 >> data/logs/tts_dev2v3.log 2>&1
echo "tts dev2v3 finished $(date)" >> data/logs/tts_dev2v3.log
