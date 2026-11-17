"""The LUT ROM in the tree belongs to the shipping model (review R3). Run: pytest rtl/tests
Written by Mythos (deoxylabs/marigold PR 1, Nov 17, 2026); only SHIPPING changed (v3 replaced v2 that morning).
test_chip_model_ids_are_the_tracked_roms was added for K1 (John's ruling of Nov 17, 2026).

rtl/core/imh_rom.svh holds the asinh table of one export (it depends on that export's frontend scale), so RTL built
from a fresh clone is bit-exact only with the model the committed file was generated from.
"""
import re
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


def test_chip_model_ids_are_the_tracked_roms():
    """K1: the chip refuses a blob built for another model or teacher (STATUS cfg error 7), so its MODEL_ID and
    TEACHER_ID must be those of the model the tracked LUT ROM came from: marigold_top.sv's defaults, and rtl/top/chip.mk,
    which reads them from that ROM through sdk/chip_models.py."""
    rom = ROOT / "rtl/core/imh_rom.svh"
    out = subprocess.run([sys.executable, str(ROOT / "sdk/chip_models.py"), "--rom", str(rom)], capture_output=True,
                         text=True, check=True)
    want = tuple(int(x) for x in out.stdout.split())
    sys.path.insert(0, str(ROOT / "sdk"))
    import chip_models as CM
    assert CM.rom_tag(rom) == SHIPPING and want == CM.ids(SHIPPING) == (3, 4)
    top = (ROOT / "rtl/top/marigold_top.sv").read_text(encoding="utf-8")
    got = tuple(int(re.search(r"parameter \[7:0\]\s+%s = 8'd(\d+)" % n, top).group(1)) for n in ("MODEL_ID", "TEACHER_ID"))
    assert got == want, f"marigold_top.sv MODEL_ID, TEACHER_ID {got}; the tracked LUT ROM's model has {want}"
    mk = (ROOT / "rtl/top/chip.mk").read_text(encoding="utf-8")
    assert "sdk/chip_models.py --rom $(CORE)/imh_rom.svh" in mk and "-GMODEL_ID=$(MODEL_ID) -GTEACHER_ID=$(TEACHER_ID)" in mk
