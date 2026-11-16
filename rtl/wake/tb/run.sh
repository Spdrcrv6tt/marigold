#!/bin/bash
# From Git Bash on Windows: bash rtl/wake/tb/run.sh [variant ...]  (default: plain stall). Logs in rtl/wake/tb/results/.
cd "$(dirname "$0")"
mkdir -p results
cat > results/_run_wsl.sh <<'EOS'
. /mnt/d/marigold/rtl/tools/env.sh
cd /mnt/d/marigold/rtl/wake/tb
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
  echo "$1 $2 corrupt=$3 expect_fail=$4: $(grep -E 'TESTS=' results/wake_$1_$2_$3.log | tail -1 | tr -s ' ')"
}
cases() { python3 -c "import json; print(' '.join(json.load(open('$1/cases.json'))))"; }
for v in $WAKE_VARIANTS; do
  vd=vectors; case $v in v2*) vd=vectors2;; esac
  for c in $(cases $vd); do run $v $c 0 0; done
done
if [ "$WAKE_NEG" = "1" ]; then
  for pre in "" v2; do
    vd=vectors; [ -n "$pre" ] && vd=vectors2
    case " $WAKE_VARIANTS " in *" ${pre}plain "*) ;; *) continue;; esac
    first=$(cases $vd | cut -d' ' -f1)
    run ${pre}plain $first 1 1     # corrupted input sample must mismatch
    run ${pre}flip $first 0 1      # flipped ROM bit must mismatch
  done
fi
EOS
MSYS_NO_PATHCONV=1 WSLENV=WAKE_VARIANTS:WAKE_NEG WAKE_NEG="${WAKE_NEG:-1}" WAKE_VARIANTS="${*:-plain stall}" wsl.exe -d Ubuntu -- bash /mnt/d/marigold/rtl/wake/tb/results/_run_wsl.sh
