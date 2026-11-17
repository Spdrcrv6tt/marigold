#!/bin/bash
# Whole-chip regression for a model export (run in WSL):   bash rtl/top/regress_mg.sh <tag> [--install-rom]
#   1. LUT ROM: generate mg-<tag>'s rtl/core/imh_rom.svh and compare it with the tracked file; a difference fails
#      the run unless --install-rom is given, which installs it (and the image step then checks it again).
#   2. ROM image (mkimage_mg.py data + image-mg), whole-chip simulators (spotter and stand-in builds), each step checked.
#   3. Checks (rtl/top/tb/mg_e2e.py, at most $JOBS simulators at a time; 1 while another regression runs):
#      positives (must be bit-exact): LibriSpeech with stand-in wakes, "Hey Breeze..." x2 with the spotter, both
#      also with the trace strap off (+trace=0, the product path), and a second wake 150 samples after a segment
#      closes (review R2); negatives (must differ): ROM bit, mic bits, CRC, a blob for another model (K1: error 7).
#   4. Unit tests: core (test-fan, test-mac on mg-<tag>'s tables) and the island blocks (cfg ctl cap).
# Exit status 0 only if every step succeeded, every positive was identical and every negative failed.
# The repo root comes from this file's location; builds go to $MARIGOLD_BUILD and $CORE_BUILD (rtl/core/tools/env.sh).
# Inputs: the export from $GOLDEN_DIR (default weights/, then data/golden), LibriSpeech under data/librispeech, the
# test blob rtl/top/tb/fan-i2c.mgc (data/sdk/fan-v3.mgc, compiled for model v3 with teacher-v4, with host = I2C and
# event flags 0xF; regenerate with rtl/top/tb/mkblob_i2c.py). The chip refuses a blob for another model (K1); chip.mk
# takes the chip's model and teacher ids from the LUT ROM.
# Test hooks: REGRESS_DRY=1 skips builds and replaces every check with `true`; REGRESS_FAKE_FAIL=<check> makes that
# check fail (used to prove that a failing child fails the run).
# (no set -u: bash treats an empty associative array as unbound under it)
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
. "$ROOT/rtl/core/tools/env.sh"
cd "$ROOT"
T=${1:-v2}
INSTALL=0; [ "${2:-}" = "--install-rom" ] && INSTALL=1
DRY=${REGRESS_DRY:-0}
GD=${GOLDEN_DIR:-}
if [ -z "$GD" ]; then for d in "$ROOT/weights" "$ROOT/data/golden"; do [ -f "$d/mg-$T.npz" ] && { GD=$d; break; }; done; fi
[ -n "$GD" ] && [ -f "$GD/mg-$T.npz" ] || { [ "$DRY" = 1 ] || { echo "REGRESS FAIL: mg-$T.npz not found"; exit 1; }; }
B=$CORE_BUILD/$T; L=$MARIGOLD_BUILD/regress-$T
mkdir -p "$B" "$L"
FAILED=0
fail() { echo "REGRESS FAIL: $*"; FAILED=1; }
step() {   # name, command...: run it, log it, stop the run on failure
    local n=$1; shift
    if [ "$DRY" = 1 ]; then echo "step $n (dry)"; return 0; fi
    if "$@" > "$L/$n.log" 2>&1; then echo "step $n ok"; else echo "REGRESS FAIL: step $n (log $L/$n.log)"; tail -15 "$L/$n.log"; exit 1; fi
}

# ---- 1. LUT ROM of this export, compared with (or installed as) the tracked file
step rom python3 rtl/core/tools/mkimage_mg.py rom --tag "$T" --golden-dir "$GD" --out "$B/rom"
if [ "$DRY" != 1 ]; then
    # the tracked ROM is LF: a generator that wrote CRLF must neither fail the compare nor install CRs (Mythos's check)
    tr -d '\r' < "$B/rom/imh_rom.svh" > "$B/rom/imh_rom.lf" && mv "$B/rom/imh_rom.lf" "$B/rom/imh_rom.svh"
    if ! cmp -s "$B/rom/imh_rom.svh" rtl/core/imh_rom.svh; then
        if [ $INSTALL = 1 ]; then cp "$B/rom/imh_rom.svh" rtl/core/imh_rom.svh; echo "installed mg-$T's LUT ROM as rtl/core/imh_rom.svh"
        else echo "REGRESS FAIL: rtl/core/imh_rom.svh is not mg-$T's LUT ROM (rerun with --install-rom, then commit it)"; exit 1; fi
    fi
    cmp -s "$B/rom/imh_rom.svh" rtl/core/imh_rom.svh || { echo "REGRESS FAIL: LUT ROM install"; exit 1; }
fi
# ---- 2. image (mkimage_mg.py data refuses a mismatched LUT ROM) and simulators (old obj/ removed by chip.mk)
rm -f "$B/flash.hex" "$ROOT/data/core/mg-$T/layout.h"
step image make -C rtl/top TAG="$T" MGGOLD="$GD" B="$B" image-mg
[ "$DRY" = 1 ] || [ -s "$B/flash.hex" ] || { echo "REGRESS FAIL: no image"; exit 1; }
IMG=$B/flash.hex
step chip make -f rtl/top/chip.mk sim BSUF="$T" STUB=0 IMG="$IMG"
step chips make -f rtl/top/chip.mk sim BSUF="${T}s" STUB=1 IMG="$IMG"

# ---- 3. whole-chip checks, a bounded number of simulators at a time
JOBS=${JOBS:-2}
X=rtl/wake/tb/vectors2/carrier_op/x.txt
E="python3 rtl/top/tb/mg_e2e.py --img $IMG --golden $GD/mg-$T.npz"
SB="--audio $X,$X --gaps 0.6,1.5,1.5 --blob rtl/top/tb/fan-i2c.mgc --wakes golden --sim $MARIGOLD_BUILD/chip$T/obj/sim"
SL="--sim $MARIGOLD_BUILD/chip${T}s/obj/sim"
declare -A PID2NAME
run_check() {   # name, args...
    local n=$1; shift
    while :; do
        local lim=$JOBS
        [ "$T" != v3 ] && pgrep -f 'regress_mg.sh v3' > /dev/null && lim=1   # the model chain's regression is running
        [ "${#PID2NAME[@]}" -lt "$lim" ] && break
        reap_one
    done
    if [ "$DRY" = 1 ]; then cmd="true"; else cmd="timeout 9000 $E $*"; fi
    [ "${REGRESS_FAKE_FAIL:-}" = "$n" ] && cmd="false"
    bash -c "$cmd" > "$L/e2e-$n.txt" 2>&1 &
    PID2NAME[$!]=$n
}
reap_one() {
    local p st
    wait -n -p p; st=$?
    [ -n "${p:-}" ] || return 0
    if [ $st -eq 0 ]; then echo "check ${PID2NAME[$p]} ok"; else fail "check ${PID2NAME[$p]} (exit $st, $L/e2e-${PID2NAME[$p]}.txt)"; fi
    tail -1 "$L/e2e-${PID2NAME[$p]}.txt" 2>/dev/null | cut -c1-400
    unset "PID2NAME[$p]"
}
run_check libri-pos $SL --out "$L/libri-pos"
run_check breeze-pos $SB --out "$L/breeze-pos"
run_check libri-pos-t0 $SL --trace 0 --out "$L/libri-pos-t0"
run_check libri-wake2 $SL --wake2 150 --out "$L/libri-wake2"       # review R2: a wake 9 ms after a close
run_check breeze-pos-t0 $SB --trace 0 --out "$L/breeze-pos-t0"
for n in rom mic crc model; do run_check breeze-$n $SB --neg $n --out "$L/breeze-$n"; done
while [ "${#PID2NAME[@]}" -gt 0 ]; do reap_one; done       # every child's status is collected, finished or not

# ---- 4. unit tests (one simulator at a time)
U="$ROOT/data/core/mg-$T"
if [ "$DRY" = 1 ]; then echo "units (dry)"
else
    if make -C rtl/top test-fan test-mac UIMG="$U" UTSIMG="$U" > "$L/unit-core.log" 2>&1 && ! grep -q -E "FAIL|failures [1-9]" "$L/unit-core.log"
    then echo "unit core ok: $(grep -c PASS "$L/unit-core.log") test-mac runs, $(grep -c '^fan:' "$L/unit-core.log") test-fan runs"
    else fail "unit core ($L/unit-core.log)"; fi
    if make -C rtl test BLOCKS="cfg ctl cap" > "$L/unit-island.log" 2>&1; then echo "unit island ok: $(tail -1 "$L/unit-island.log")"
    else fail "unit island ($L/unit-island.log)"; fi
fi
if [ $FAILED = 0 ]; then echo "REGRESS PASS ($T)"; else echo "REGRESS FAILED ($T)"; fi
exit $FAILED
