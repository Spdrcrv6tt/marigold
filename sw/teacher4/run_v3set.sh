#!/usr/bin/env bash
cd /d/marigold
PY=/d/marigold/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
$PY sw/teacher4/gen_evalsets.py v3 >> data/logs/evalsets.log 2>&1
echo "evalsets finished $(date)" >> data/logs/evalsets.log
