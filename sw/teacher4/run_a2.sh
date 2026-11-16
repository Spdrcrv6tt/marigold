#!/usr/bin/env bash
# Teacher v4 step 2 (model agent, Nov 16): after the text generation, the teacher selection chain (GPU) and, in
# parallel, TTS of eval/text_v2 (eval voices, TEST-ONLY) and of the new training products (train voices).
cd /d/marigold
PY=/d/marigold/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
for i in $(seq 1 360); do grep -q "gen finished" data/logs/t4_gen.log && break; sleep 10; done
grep -q "gen finished" data/logs/t4_gen.log || { echo "gen did not finish in 60 min" >> data/logs/t4_sel.log; exit 1; }
( $PY -I sw/data/tts_build.py jobs2 > data/logs/t4_tts.log 2>&1
  bash sw/data/run_synth.sh data/tts/jobs_eval_v2.jsonl 4 1 >> data/logs/t4_tts.log 2>&1
  bash sw/data/run_synth.sh data/tts/jobs_train_v4.jsonl 8 1 >> data/logs/t4_tts.log 2>&1
  $PY -I sw/data/tts_build.py merge eval_v2 train_v4 >> data/logs/t4_tts.log 2>&1
  echo "tts finished $(date)" >> data/logs/t4_tts.log ) &
bash sw/teacher4/run_sel.sh
wait
