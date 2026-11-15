#!/bin/bash
# Build the train speech set: job list from sdk/configs + data/gate0b/train_text.json (train voices only),
# synthesis, manifest, Whisper QA fields. Resumable: finished clips are skipped.
set -e
R=/d/marigold; cd $R; PY=".venv/Scripts/python.exe -I"
$PY sw/data/tts_build.py jobs
sw/data/run_synth.sh data/tts/jobs_train.jsonl 8 2 0
$PY sw/data/tts_build.py merge
HF_HOME=$R/data/hf $PY sw/data/tts_asr_check.py data/tts/train/manifest.jsonl data/tts/train/asr_check.json cuda | head -5
