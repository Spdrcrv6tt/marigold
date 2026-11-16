#!/usr/bin/env bash
# Teacher v4, step 1 (model agent, Nov 16): text for the remaining training products (qwen2.5:14b) and the
# candidate base models, in parallel. Logs in data/logs/t4_*.log.
cd /d/marigold
PY=/d/marigold/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
$PY sw/teacher4/fetch_models.py Qwen/Qwen3-Embedding-0.6B Qwen/Qwen3-Embedding-4B Qwen/Qwen3-Embedding-8B > data/logs/t4_fetch.log 2>&1 &
$PY sw/teacher4/gen_train_v4.py > data/logs/t4_gen.log 2>&1
curl -s http://localhost:11434/api/generate -d '{"model":"qwen2.5:14b","keep_alive":0}' > /dev/null
echo "gen finished $(date)" >> data/logs/t4_gen.log
wait
