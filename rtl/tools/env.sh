# Source inside WSL: . /mnt/d/marigold/rtl/tools/env.sh
export MARIGOLD=/mnt/d/marigold
export IMHOTEP_TOOLS=$HOME/.imhotep               # Imhotep WSL toolchain: read-only reuse
export PATH=$IMHOTEP_TOOLS/conda/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1
export MARIGOLD_BUILD=$HOME/.marigold/build     # Verilator / synthesis scratch, off the Windows drive
