#!/bin/bash
# Run a command in the WSL toolchain from Git Bash on Windows, in this tree:   bash rtl/tools/wsl.sh make -C rtl test
# (MSYS path conversion is turned off so /mnt paths survive; MARIGOLD_BUILD and CORE_BUILD pass through if set.)
W=$(cd "$(dirname "$0")/../.." && pwd -W)
M="/mnt/$(echo "${W:0:1}" | tr 'A-Z' 'a-z')${W:2}"
MSYS_NO_PATHCONV=1 WSLENV=MARIGOLD_BUILD:CORE_BUILD wsl.exe -d Ubuntu -- bash -c ". $M/rtl/tools/env.sh && cd $M && $*"
