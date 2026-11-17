#!/bin/bash
# From Mythos's review fixes (deoxylabs/marigold PR 1, Nov 17, 2026), kept as written.
# The checks of run.sh without WSL: any Linux or macOS shell with verilator, cocotb-config, make and python3 on PATH.
#   bash rtl/wake/tb/run_native.sh [variant ...]     default: plain stall v2plain v2stall
# Every variant runs on every case of its vector set; then, unless WAKE_NEG=0, the two negative controls of each
# format that was run (a corrupted input sample and a flipped ROM bit must both be detected).
# Exit status: 0 only when every run reports FAIL=0. Logs in rtl/wake/tb/results/.
cd "$(dirname "$0")" || exit 1
mkdir -p results
VARIANTS="${*:-plain stall v2plain v2stall}"
bad=0
# a flipped weight bit (one ROM word) for the ROM negative control
for m in mem mem2; do
  [ -d ../$m ] || continue
  mkdir -p results/${m}_flip && cp ../$m/*.mem results/${m}_flip/
  python3 - results/${m}_flip/wake_rom.mem <<'EOP'
import sys
p = sys.argv[1]; w = open(p).read().split()
w[5000] = "%0*x" % (len(w[5000]), int(w[5000], 16) ^ 0x0010); open(p, "w").write(chr(10).join(w) + chr(10))
EOP
done
run() {  # variant case corrupt expect_fail
  make VARIANT=$1 WAKE_CASE=$2 WAKE_CORRUPT=$3 WAKE_EXPECT_FAIL=$4 > results/wake_$1_$2_$3.log 2>&1
  local line
  line=$(grep -E 'TESTS=' results/wake_$1_$2_$3.log | tail -1 | tr -s ' ')
  echo "$1 $2 corrupt=$3 expect_fail=$4: ${line:-no result (see results/wake_$1_$2_$3.log)}"
  case "$line" in *"FAIL=0"*) ;; *) bad=1 ;; esac
}
cases() { python3 -c "import json; print(' '.join(json.load(open('$1/cases.json'))))"; }
for v in $VARIANTS; do
  vd=vectors; case $v in v2*) vd=vectors2;; esac
  for c in $(cases $vd); do run $v $c 0 0; done
done
if [ "${WAKE_NEG:-1}" = "1" ]; then
  for pre in "" v2; do
    vd=vectors; [ -n "$pre" ] && vd=vectors2
    case " $VARIANTS " in *" ${pre}plain "*) ;; *) continue;; esac
    first=$(cases $vd | cut -d' ' -f1)
    run ${pre}plain $first 1 1     # corrupted input sample must mismatch
    run ${pre}flip $first 0 1      # flipped ROM bit must mismatch
  done
fi
[ $bad -eq 0 ] && echo "wake tests: PASS" || echo "wake tests: FAIL"
exit $bad
