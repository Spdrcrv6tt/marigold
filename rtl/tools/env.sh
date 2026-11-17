# Source inside WSL: . <repo>/rtl/tools/env.sh   (the repo root is taken from this file's own location, so a worktree
# or a fresh clone builds from itself). Build scratch: $MARIGOLD_BUILD (default ~/.marigold/build; set it first to
# keep a tree's builds apart).
export MARIGOLD=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export IMHOTEP_TOOLS=$HOME/.imhotep               # Imhotep WSL toolchain: read-only reuse
export PATH=$IMHOTEP_TOOLS/conda/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1
export MARIGOLD_BUILD=${MARIGOLD_BUILD:-$HOME/.marigold/build}
