#!/usr/bin/env bash
# Review S5 / S2 check for run_resume.sh with a stub trainer (no GPU, no main tree): exit codes, success marker,
# --resume on a restart. Run from anywhere: bash sw/intent/test_resume.sh
set -u
HERE=$(cd "$(dirname "$0")/../.." && pwd)
T=$(mktemp -d)
mkdir -p $T/data/logs $T/data/intent $T/sw/intent
cat > $T/stub.sh <<'EOF'
#!/usr/bin/env bash
n=$(cat "$STUB_DIR/count" 2>/dev/null || echo 0); echo $((n + 1)) > "$STUB_DIR/count"
echo "$@" >> "$STUB_DIR/args"
rcs=($STUB_RCS); rc=${rcs[$n]:-${rcs[-1]}}
[ "$rc" = 0 ] && touch "$STUB_DIR/data/intent/$4.pt"
exit $rc
EOF
chmod +x $T/stub.sh
run() { rm -f $T/count $T/args; STUB_DIR=$T STUB_RCS="$1" MG_ROOT=$T MG_PY=$T/stub.sh bash $HERE/sw/intent/run_resume.sh t init.pt $2 --x 1; echo $?; }
fails=0
check() { if [ "$1" = "$2" ]; then echo "ok   $3"; else echo "FAIL $3 (got $1, want $2)"; fails=$((fails + 1)); fi; }
check "$(run "0" $(( $(date +%s) + 600 )))" 0 "success exits 0"
check "$(grep -c 'success 1' $T/data/logs/train_t.log)" 1 "success marker"
check "$(run "1" $(( $(date +%s) + 123 )))" 1 "every attempt crashed: exit 1"
check "$(tail -1 $T/data/logs/train_t.log | grep -c 'success 0')" 1 "failure marker"
check "$(run "3" $(( $(date +%s) + 600 )))" 3 "throughput abort: exit 3"
touch $T/data/intent/t.pt
check "$(run "1 0" $(( $(date +%s) + 600 )))" 0 "crash then success"
check "$(sed -n 2p $T/args | grep -c -- '--resume')" 1 "restart passes --resume"
check "$(sed -n 1p $T/args | grep -c -- '--resume')" 0 "first attempt has no --resume"
# run_regress.sh: a "=== done" from an earlier attempt does not count; "=== failed" stops the wait
mkdir -p $T/r/data/logs; echo "=== done 01:00" > $T/r/data/logs/f.log
( sleep 3; echo "=== done 02:00" >> $T/r/data/logs/f.log ) &
check "$(MG_ROOT=$T/r MG_NO_WSL=1 bash $HERE/sw/intent/run_regress.sh x f.log 3)" "would start the regression for x" "regress waiter: new marker only"
wait
MG_ROOT=$T/r MG_NO_WSL=1 bash $HERE/sw/intent/run_regress.sh y f.log 1 > /dev/null; check "$?" 1 "regress waiter: old marker alone times out"
( sleep 3; echo "=== failed: qat 03:00" >> $T/r/data/logs/f.log ) &
MG_ROOT=$T/r MG_NO_WSL=1 bash $HERE/sw/intent/run_regress.sh z f.log 3 > /dev/null; check "$?" 1 "regress waiter: stops on failure"
check "$(grep -c 'failed: qat' $T/r/data/logs/regress_z.log)" 1 "failure reason recorded"
wait
rm -rf $T
[ $fails -eq 0 ] && echo "test_resume: all ok" || { echo "test_resume: $fails failed"; exit 1; }
