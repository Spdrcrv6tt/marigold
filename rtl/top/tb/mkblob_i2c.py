"""Test blob for the whole-chip regression: an SDK blob with host = I2C at 0x2A and event flags 0xF (irq, WAKE,
NOMATCH, TIMEOUT frames), in blob version 1 with the model and teacher it was built for (K1, John's ruling of Nov 17,
2026), CRC recomputed, so every event reaches the harness as a frame. Vectors and tables are copied unchanged.
    python3 rtl/top/tb/mkblob_i2c.py [SRC [OUT]] [--model-tag TAG]                                  (OUT - : stdout)
SRC defaults to data/sdk/fan-v3.mgc: sdk/configs/fan.json compiled for model v3 with teacher-v4, the default expansion
and the teacher's projection, as text prototypes (--enroll text; the SDK default became audio on Nov 17, review K2,
and this stays the regression's test blob), and OUT to rtl/top/tb/fan-i2c.mgc. A
version-0 source records no model: --model-tag names it, else the teacher in the source's report (SRC.json,
"embedder") gives the one model paired with that teacher (sdk/chip_models.py)."""
import argparse, json, struct, sys, zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "sdk"))
import chip_models as CM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("src", nargs="?", default=str(ROOT / "data/sdk/fan-v3.mgc"))
ap.add_argument("out", nargs="?", default=str(ROOT / "rtl/top/tb/fan-i2c.mgc"))
ap.add_argument("--model-tag", default=None)
a = ap.parse_args()
b = bytearray(open(a.src, "rb").read())
ver = struct.unpack_from("<H", b, 4)[0]
if a.model_tag:
    mid, tid = CM.ids(a.model_tag)
elif ver == 0:
    teacher = Path(json.load(open(a.src + ".json", encoding="utf-8"))["embedder"].replace("\\", "/")).name
    tags = [t for t, x in CM.TEACHER.items() if x == teacher]
    if len(tags) != 1:
        sys.exit(f"{a.src} was built with {teacher}, which pairs with {tags or 'no model'}: pass --model-tag")
    mid, tid = CM.ids(tags[0])
else:
    mid, tid = b[0x1D], b[0x1E]
b[4:6] = struct.pack("<H", 1)
b[0x1D], b[0x1E], b[0x1F] = mid, tid, 0
b[14] = 1; b[15] = 0x2A; b[26:28] = struct.pack("<H", 0xF)
b[-4:] = struct.pack("<I", zlib.crc32(bytes(b[:-4])) & 0xFFFFFFFF)
if a.out == "-":
    sys.stdout.buffer.write(bytes(b))
else:
    open(a.out, "wb").write(b)
print(a.out, len(b), "bytes, model id", mid, "teacher id", tid, file=sys.stderr)
