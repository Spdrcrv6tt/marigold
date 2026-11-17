"""Mythos's test (deoxylabs/marigold PR 1, Nov 17, 2026), adapted to master (marked ADAPTED): the stand-ins follow master's calls (mkimage_mg.py rom --out DIR; image-mg must leave B/flash.hex; the export must exist under weights/; LibriSpeech checks are named libri-*); master installs the ROM only with --install-rom, which Mythos's script always does; master runs 9 checks, not 5 (8
until K1 added the neg-model check, Nov 17, 2026); master writes no blob (its checks read the committed rtl/top/tb/fan-i2c.mgc), so Mythos's blob test is left out.

rtl/top/regress_mg.sh reports failure when any step fails (review R9, R10, R11). Run: pytest rtl/tests

The real script runs in a copy of the tree where every heavy step is a stand-in that can be told to fail:
`make` and `timeout` come from a stub directory on PATH, and the two Python tools the script calls are small
fakes at their usual paths. Only the script's own control flow is under test.
"""
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TAG = "t9"

MAKE = r'''#!/bin/bash
# stand-in for make: logs its arguments and fails when $FAIL names the step
echo "make $*" >> "$STUB_LOG"
case "$*" in
  *image-mg*) [[ " $FAIL " == *" image "* ]] && { echo "image failed"; exit 2; }; for a in "$@"; do case "$a" in B=*) b=${a#B=} ;; esac; done; mkdir -p "$b"; echo img > "$b/flash.hex"; echo "image 123 bytes" ;;
  *"chip.mk sim"*)
    for a in "$@"; do case "$a" in BSUF=*) suf=${a#BSUF=} ;; esac; done
    [[ " $FAIL " == *" build-$suf "* ]] && { echo "build failed"; exit 2; }
    mkdir -p "$MARIGOLD_BUILD/chip$suf/obj"; echo fresh > "$MARIGOLD_BUILD/chip$suf/obj/sim" ;;
  *test-fan*) echo "fan: 467 operations, 0 failures"; [[ " $FAIL " == *" unit "* ]] && { echo "FAIL lane 3"; exit 2; } ;;
esac
exit 0
'''
TIMEOUT = '#!/bin/bash\nshift\nexec "$@"\n'
ENV = 'export MARIGOLD_BUILD="$MARIGOLD_ROOT/build"\nexport CORE_BUILD="$MARIGOLD_ROOT/build/core"\nmkdir -p "$CORE_BUILD"\n'
MKIMAGE = r'''import os, sys
from pathlib import Path
tag = sys.argv[sys.argv.index("--tag") + 1]
if "rom-tool" in os.environ.get("FAIL", "").split():
    sys.exit(1)
d = Path(sys.argv[sys.argv.index("--out") + 1]); d.mkdir(parents=True, exist_ok=True)
(d / "imh_rom.svh").write_bytes(b"// rom " + tag.encode() + b"\r\n")
'''
E2E = r'''import os, sys
a = sys.argv[1:]
open(os.environ["STUB_LOG"], "a").write("e2e " + " ".join(a) + "\n")
sim = open(a[a.index("--sim") + 1]).read().strip()
name = ("neg-" + a[a.index("--neg") + 1]) if "--neg" in a else "libri" if "libri" in os.path.basename(a[a.index("--out") + 1]) else "breeze"
print("stale binary" if sim != "fresh" else "ok")
sys.exit(1 if sim != "fresh" or name in os.environ.get("FAIL", "").split() else 0)
'''


def write(path, text, executable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "marigold"
    write(root / "rtl/top/regress_mg.sh", (ROOT / "rtl/top/regress_mg.sh").read_text())
    write(root / "rtl/core/tools/env.sh", ENV)
    write(root / "rtl/core/tools/mkimage_mg.py", MKIMAGE)
    write(root / "rtl/top/tb/mg_e2e.py", E2E)
    write(root / "rtl/core/imh_rom.svh", "// rom of the previous model\n")
    write(root / "weights/mg-t9.npz", "")                       # ADAPTED: master needs the export to exist
    (root / "data/sdk").mkdir(parents=True)
    (root / "data/sdk/fan.mgc").write_bytes(bytes(64))
    write(tmp_path / "bin/make", MAKE, executable=True)
    write(tmp_path / "bin/timeout", TIMEOUT, executable=True)
    return root


def run(root, fail=""):
    env = dict(os.environ, MARIGOLD_ROOT=str(root), FAIL=fail, STUB_LOG=str(root / "stub.log"),
               PATH=f"{root.parent / 'bin'}{os.pathsep}{os.environ['PATH']}")
    (root / "stub.log").write_text("")
    if not shutil.which("python3", path=env["PATH"]):
        pytest.skip("no python3 on PATH")
    return subprocess.run(["bash", str(root / "rtl/top/regress_mg.sh"), TAG, "--install-rom"], env=env, capture_output=True, text=True,
                          cwd=root.parent, timeout=120)


def e2e_runs(root):
    return [l for l in (root / "stub.log").read_text().splitlines() if l.startswith("e2e ")]


def test_passes_when_every_step_passes(tree):
    out = run(tree)
    assert out.returncode == 0, out.stdout[-600:] + out.stderr[-600:]
    assert len(e2e_runs(tree)) == 9                     # ADAPTED: master has 5 positives and 4 negatives (neg-model: K1)


def test_installs_the_models_lut_rom_without_carriage_returns(tree):
    run(tree)
    assert (tree / "rtl/core/imh_rom.svh").read_bytes() == b"// rom " + TAG.encode() + b"\n"


@pytest.mark.parametrize("step", ["libri", "breeze", "neg-rom", "neg-mic", "neg-crc", "neg-model", "unit"])   # ADAPTED: neg-model (K1)
def test_fails_when_a_check_fails(tree, step):
    assert run(tree, fail=step).returncode != 0


def test_fails_when_the_image_build_fails(tree):
    assert run(tree, fail="image").returncode != 0


def test_a_failed_rom_step_stops_the_run_and_leaves_the_tree_alone(tree):
    out = run(tree, fail="rom-tool")
    assert out.returncode != 0
    assert (tree / "rtl/core/imh_rom.svh").read_text() == "// rom of the previous model\n"
    assert not e2e_runs(tree)


@pytest.mark.parametrize("suffix", [TAG, TAG + "s"])
def test_a_failed_chip_build_never_runs_the_checks_on_the_old_binary(tree, suffix):
    write(tree / f"build/chip{suffix}/obj/sim", "stale\n")
    out = run(tree, fail=f"build-{suffix}")
    assert out.returncode != 0
    assert not e2e_runs(tree)
