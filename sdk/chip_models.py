"""The chip's intent models and the teacher each was distilled from (weights/README.md): the one table that
sdk/compile.py, sw/intent/evaluate.py, the whole-chip build (rtl/top/chip.mk) and its checks read (K1, John's ruling
of Nov 17, 2026). A config blob records the model id and the teacher id it was built for (docs/config-blob.md, bytes
0x1D and 0x1E); the chip refuses a blob made for another model or teacher (STATUS cfg error 7).

    python sdk/chip_models.py TAG        prints "MODEL_ID TEACHER_ID" for a model tag (e.g. "3 4" for v3)
    python sdk/chip_models.py --rom      the same for the model of the tracked LUT ROM (rtl/core/imh_rom.svh)

Ids: a model's id is its tag number (v3 -> 3); a teacher's id is its version (teacher-v4 -> 4). 0 means none: an
export outside the table (randtest, qsmoke, v11) or an embedder that is not a teacher (the test stub). A new model
gets its line here before any blob is compiled for it.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHIPPING = "v3"                                                      # docs/decisions.md, Nov 17, 2026
TEACHER = {"v1": "teacher-v3", "v2": "teacher-v4", "v3": "teacher-v4"}


def model_id(tag):
    """'v3' -> 3; a tag outside the table -> 0."""
    return int(tag[1:]) if tag in TEACHER else 0


def teacher_id(teacher):
    """'teacher-v4' (or a directory path ending in it) -> 4; anything else (the stub, a gate 0b model) -> 0."""
    m = re.fullmatch(r"teacher-v(\d+)", Path(str(teacher).replace("\\", "/")).name)
    return int(m.group(1)) if m else 0


def ids(tag):
    """(model id, teacher id) of a model tag."""
    return model_id(tag), teacher_id(TEACHER.get(tag, ""))


def rom_tag(path=ROOT / "rtl/core/imh_rom.svh"):
    """The model export the LUT ROM was generated from (its first line names mg-<tag>.npz)."""
    with open(path, encoding="utf-8") as f:
        m = re.search(r"from mg-([\w.]+?)\.npz", f.readline())
    if not m:
        raise ValueError(f"{path}: no 'from mg-<tag>.npz' in the first line")
    return m.group(1)


def rom_ids(path=ROOT / "rtl/core/imh_rom.svh"):
    return ids(rom_tag(path))


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    print("%d %d" % (rom_ids(*a[1:2]) if a[0] == "--rom" else ids(a[0])))
