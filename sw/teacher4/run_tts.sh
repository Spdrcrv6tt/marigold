#!/usr/bin/env bash
# Iteration-2 TTS (model agent): eval/text_v2 (eval voices) and the new training products (train voices), Kokoro on
# CPU so the GPU stays free for the teacher runs. Started at below-normal priority. Log data/logs/t4_tts.log.
cd /d/marigold
PY=/d/marigold/.venv/Scripts/python.exe
bash sw/data/run_synth.sh data/tts/jobs_eval_v2.jsonl 0 0 3 >> data/logs/t4_tts.log 2>&1
bash sw/data/run_synth.sh data/tts/jobs_train_v4.jsonl 6 0 3 >> data/logs/t4_tts.log 2>&1
$PY -I sw/data/tts_build.py merge eval_v2 train_v4 >> data/logs/t4_tts.log 2>&1
echo "tts finished $(date)" >> data/logs/t4_tts.log
