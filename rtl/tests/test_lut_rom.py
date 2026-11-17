"""The LUT ROM in the tree belongs to the shipping model (review R3). Run: pytest rtl/tests
Written by Mythos (deoxylabs/marigold PR 1, Nov 17, 2026); only SHIPPING changed (v3 replaced v2 that morning).

rtl/core/imh_rom.svh holds the asinh table of one export (it depends on that export's frontend scale), so RTL built
from a fresh clone is bit-exact only with the model the committed file was generated from.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHIPPING = "v3"            # docs/decisions.md, Nov 17, 2026: model v3 ships (weights/mg-v3.npz)


def test_committed_lut_rom_is_the_shipping_models(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "rtl/core/tools/mkimage_mg.py"), "rom", "--tag", SHIPPING,
                    "--golden-dir", str(ROOT / "weights"), "--out", str(tmp_path)], check=True, capture_output=True)
    generated = (tmp_path / "imh_rom.svh").read_text(encoding="utf-8").replace("\r\n", "\n")
    committed = (ROOT / "rtl/core/imh_rom.svh").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert committed == generated
