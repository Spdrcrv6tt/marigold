# Core/top build environment (source inside WSL; it sources rtl/tools/env.sh of the same tree).
# clang + lld for the RV32IM firmware live in their own conda prefix (Imhotep's, read-only).
. "$(dirname "${BASH_SOURCE[0]}")/../../tools/env.sh"
export LLVM=$HOME/.imhotep/core-llvm/bin
export CORE_BUILD=${CORE_BUILD:-$MARIGOLD_BUILD/core}
mkdir -p $CORE_BUILD
