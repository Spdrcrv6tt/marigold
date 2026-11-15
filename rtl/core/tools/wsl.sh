#!/bin/bash
# Run a command line in WSL with the core environment: bash rtl/core/tools/wsl.sh 'make -C rtl/core fw'
# The command is written to a file first so that $VARS survive wsl.exe's own shell.
T=/d/marigold/data/tmp/core-cmd-$$.sh
mkdir -p /d/marigold/data/tmp
printf '%s\n' ". /mnt/d/marigold/rtl/core/tools/env.sh" "cd /mnt/d/marigold" "$*" > $T
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -- bash /mnt/d/marigold/data/tmp/core-cmd-$$.sh
rc=$?; rm -f $T; exit $rc
