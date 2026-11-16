#!/usr/bin/env bash
# Teacher v4 final (model agent, Nov 16): Qwen3-Embedding-4B + LoRA 16, v4 recipe, all 120 training products (selection
# winner on the held-out training products, reports/gate0/t4_sel_*.json); gate 0 scored once on eval/text test and
# eval/text_v2 (teacher-v3 scored the same way as the reference); then the audio-training data index, targets and
# decision prototypes in teacher-v4's space. Logs data/logs/t4_*.log, v2_*.log.
cd /d/marigold
PY=/d/marigold/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
$PY sw/intent/data_index.py > data/logs/v2_index.log 2>&1 &
$PY sw/teacher4/finetune_t4.py Qwen/Qwen3-Embedding-4B data/models/teacher-v4 --steps 1500 --lora 16 --lr 1e-4 > data/logs/t4_ft_final.log 2>&1
$PY sw/teacher4/text_gate.py data/models/teacher-v4 --v2 --out reports/gate0/text_gate_teacher_v4.json > data/logs/t4_gate.log 2>&1
$PY sw/teacher4/text_gate.py data/models/teacher-v3 --v2 --out reports/gate0/text_gate_teacher_v3.json > data/logs/t4_gate_v3.log 2>&1
wait
$PY sw/intent/targets.py data/models/teacher-v4 > data/logs/v2_targets.log 2>&1
$PY sw/intent/protos.py data/models/teacher-v4 > data/logs/v2_protos.log 2>&1
echo "t4 final chain finished $(date)" >> data/logs/t4_gate.log
