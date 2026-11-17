#!/bin/bash
# Runs the whole-chip regression for one model tag inside WSL (model agent); log in data/logs/regress_<tag>.log.
cd /mnt/d/marigold
bash rtl/top/regress_mg.sh $1 > /mnt/d/marigold/data/logs/regress_$1.log 2>&1
echo "regress finished rc=$? $(date)" >> /mnt/d/marigold/data/logs/regress_$1.log
