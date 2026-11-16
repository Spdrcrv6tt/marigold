#!/bin/bash
# Whole-chip regression for a model export (run in WSL): image, LUT ROM install, chip builds (spotter and stand-in),
# five checks (LibriSpeech stand-in; "Hey Breeze" x2 with the spotter; ROM, mic and CRC negative controls) and the
# core unit tests.   bash rtl/top/regress_mg.sh <tag>   (results under $MARIGOLD_BUILD/e2e*)
. /mnt/d/marigold/rtl/core/tools/env.sh
cd /mnt/d/marigold
T=${1:-v1}
mkdir -p $CORE_BUILD/$T
timeout 1200 make -C rtl/top TAG=$T B=$CORE_BUILD/$T image-mg > $CORE_BUILD/$T-image.log 2>&1 || { tail -20 $CORE_BUILD/$T-image.log; exit 1; }
grep -E "image .* bytes" $CORE_BUILD/$T-image.log
timeout 600 python3 rtl/core/tools/mkimage_mg.py rom --tag $T > $CORE_BUILD/$T-rom.log 2>&1 && cp data/core/mg-$T/imh_rom.svh rtl/core/imh_rom.svh && sed -i 's/\r$//' rtl/core/imh_rom.svh
IMG=$CORE_BUILD/$T/flash.hex
timeout 1500 make -f rtl/top/chip.mk sim BSUF=$T STUB=0 IMG=$IMG > $MARIGOLD_BUILD/chip$T-build.log 2>&1 &
timeout 1500 make -f rtl/top/chip.mk sim BSUF=${T}s STUB=1 IMG=$IMG > $MARIGOLD_BUILD/chip${T}s-build.log 2>&1 &
wait
python3 - <<'PY'
import struct, zlib
b = bytearray(open("data/sdk/fan.mgc", "rb").read())
b[14] = 1; b[15] = 0x2A; b[26:28] = struct.pack("<H", 0xF)
b[-4:] = struct.pack("<I", zlib.crc32(bytes(b[:-4])) & 0xFFFFFFFF)
open("/home/johnmc/.marigold/build/fan-i2c.mgc", "wb").write(b)
PY
X=rtl/wake/tb/vectors2/carrier_op/x.txt
E="python3 rtl/top/tb/mg_e2e.py --img $IMG --golden data/golden/mg-$T.npz"
B="--audio $X,$X --gaps 0.6,1.5,1.5 --blob $MARIGOLD_BUILD/fan-i2c.mgc --wakes golden --sim $MARIGOLD_BUILD/chip$T/obj/sim"
timeout 9000 $E --sim $MARIGOLD_BUILD/chip${T}s/obj/sim --out $MARIGOLD_BUILD/e2e/libri/$T-pos > $MARIGOLD_BUILD/e2e-$T-libri-pos.txt 2>&1 &
timeout 9000 $E $B --out $MARIGOLD_BUILD/e2e/breeze/$T-pos > $MARIGOLD_BUILD/e2e-$T-breeze-pos.txt 2>&1 &
for n in rom mic crc; do timeout 9000 $E $B --neg $n --out $MARIGOLD_BUILD/e2e/breeze/$T-$n > $MARIGOLD_BUILD/e2e-$T-breeze-$n.txt 2>&1 & done
timeout 2400 make -C rtl/top test-fan test-mac UIMG=/mnt/d/marigold/data/core/mg-$T UTSIMG=/mnt/d/marigold/data/core/mg-$T 2>&1 | grep -E "^fan:|^LANES=|FAIL" > $MARIGOLD_BUILD/unit-$T.txt &
wait
cat $MARIGOLD_BUILD/unit-$T.txt
for f in libri-pos breeze-pos breeze-rom breeze-mic breeze-crc; do echo "== $f"; tail -1 $MARIGOLD_BUILD/e2e-$T-$f.txt | cut -c1-900; done
