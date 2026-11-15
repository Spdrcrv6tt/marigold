#!/bin/bash
# Run a command in the Imhotep WSL toolchain from Git Bash on Windows:
#   bash rtl/tools/wsl.sh make -C rtl test
# (MSYS path conversion is turned off so /mnt/d paths survive.)
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -- bash -c ". /mnt/d/marigold/rtl/tools/env.sh && cd /mnt/d/marigold && $*"
