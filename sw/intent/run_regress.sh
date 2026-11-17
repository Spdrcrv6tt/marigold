#!/usr/bin/env bash
# Waits for the freeze pipeline of TAG to write "=== done" to data/logs/<LOGNAME> AFTER this waiter started (a marker
# from an earlier attempt does not count; review S5), then starts the WSL regression through PowerShell Start-Process
# (a wsl.exe started from a hidden Git Bash did not run). Stops on "=== failed". Usage: run_regress.sh TAG LOGNAME [WAIT]
cd ${MG_ROOT:-/d/marigold}
LOG=data/logs/$2
START=$( [ -f $LOG ] && wc -l < $LOG || echo 0 )
for i in $(seq 1 ${3:-1080}); do
  NEW=$( [ -f $LOG ] && tail -n +$((START + 1)) $LOG )
  echo "$NEW" | grep -q "=== failed" && { echo "freeze pipeline failed: $(echo "$NEW" | grep '=== failed' | tail -1)" > data/logs/regress_$1.log; exit 1; }
  echo "$NEW" | grep -q "=== done" && break
  sleep 10
done
echo "$NEW" | grep -q "=== done" || { echo "freeze pipeline not done in time" > data/logs/regress_$1.log; exit 1; }
[ -n "$MG_NO_WSL" ] && { echo "would start the regression for $1"; exit 0; }
powershell.exe -NoProfile -Command "Start-Process -WindowStyle Hidden -FilePath wsl.exe -ArgumentList '-d','Ubuntu','--','bash','/mnt/d/marigold/sw/intent/regress_wsl.sh','$1'"
