r"""Where the scripts find their inputs. Golden-model exports: $GOLDEN_DIR, else data/golden (this repo's own,
written by the model agent), else data/imhotep/golden (Imhotep's exports, a read-only junction to D:\imhotep\data)."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
ROM_TAG = "w4c8r"   # the chip's ROM image (docs/decisions.md, Nov 14): default export of the RTL flows and harnesses;
                    # rtl/core/imh_rom.svh holds its asinh table. w4c8 before it (to 1779ad7).


def golden(name):
    for d in (os.environ.get("GOLDEN_DIR", ""), ROOT / "data" / "golden", ROOT / "data" / "imhotep" / "golden"):
        if d and (Path(d) / name).exists():
            return Path(d) / name
    raise FileNotFoundError(f"{name} in $GOLDEN_DIR, data/golden or data/imhotep/golden")
