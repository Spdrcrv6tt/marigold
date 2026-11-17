#!/bin/bash
# Run a command line in WSL with the core environment of this tree: bash rtl/core/tools/wsl.sh 'make -C rtl/top test-fan'
# The command is written to a file first (in this tree's data/tmp) so that $VARS survive wsl.exe's own shell.
W=$(cd "$(dirname "$0")/../../.." && pwd -W)
M="/mnt/$(echo "${W:0:1}" | tr 'A-Z' 'a-z')${W:2}"
D="$(cd "$(dirname "$0")/../../.." && pwd)/data/tmp"
mkdir -p "$D"
T="$D/core-cmd-$$.sh"
printf '%s\n' ". $M/rtl/core/tools/env.sh" "cd $M" "$*" > "$T"
MSYS_NO_PATHCONV=1 WSLENV=MARIGOLD_BUILD:CORE_BUILD wsl.exe -d Ubuntu -- bash "$M/data/tmp/core-cmd-$$.sh"
rc=$?; rm -f "$T"; exit $rc
