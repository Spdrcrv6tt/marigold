#!/usr/bin/env bash
# Waits for the freeze pipeline of TAG (data/logs/<LOGNAME> "=== done", up to WAIT x 10 s), then starts the WSL
# regression through PowerShell Start-Process (a wsl.exe started from a hidden Git Bash did not run).
# Usage: run_regress.sh TAG LOGNAME [WAIT]
cd /d/marigold
for i in $(seq 1 ${3:-1080}); do grep -q "=== done" data/logs/$2 2>/dev/null && break; sleep 10; done
grep -q "=== done" data/logs/$2 || { echo "freeze pipeline not done in time" > data/logs/regress_$1.log; exit 1; }
powershell.exe -NoProfile -Command "Start-Process -WindowStyle Hidden -FilePath wsl.exe -ArgumentList '-d','Ubuntu','--','bash','/mnt/d/marigold/sw/intent/regress_wsl.sh','$1'"
