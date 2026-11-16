"""Join firmware and data into the ROM/flash image (same layout as mkimage.py flash; numpy-free for WSL).
    python3 catimage.py fw.bin data.bin flash.bin [data_base]
data_base: where data.bin starts (layout.h DATA_BASE): 65536 (Imhotep, Poinsetta; the default) or 0x8040 (Marigold,
rtl/core/tools/mkimage_mg.py). From Poinsetta commit 1719528, with the data_base argument added."""
import sys
fw = open(sys.argv[1], "rb").read(); d = open(sys.argv[2], "rb").read()
base = int(sys.argv[4], 0) if len(sys.argv) > 4 else 65536
img = b"IMH1" + len(fw).to_bytes(4, "little") + fw
assert len(img) <= base, "firmware does not fit below DATA_BASE"
img += b"\xff" * (base - len(img)) + d
open(sys.argv[3], "wb").write(img)
print("image", len(img), "bytes, firmware", len(fw), "data at", hex(base))
