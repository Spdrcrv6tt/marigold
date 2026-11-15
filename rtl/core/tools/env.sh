# Core/top build environment (source inside WSL after rtl/tools/env.sh).
# clang + lld for the RV32IM firmware live in their own conda prefix: tools/setup_llvm.sh
. /mnt/d/marigold/rtl/tools/env.sh
export LLVM=$HOME/.imhotep/core-llvm/bin   # Imhotep clang + lld: read-only reuse
export CORE_BUILD=$HOME/.marigold/build/core
mkdir -p $CORE_BUILD
