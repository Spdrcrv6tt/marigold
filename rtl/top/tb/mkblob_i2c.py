"""Test blob for the whole-chip regression: an SDK blob with host = I2C at 0x2A and event flags 0xF (irq, WAKE,
NOMATCH, TIMEOUT frames), CRC recomputed, so every event reaches the harness as a frame.
    python3 rtl/top/tb/mkblob_i2c.py data/sdk/fan.mgc rtl/top/tb/fan-i2c.mgc"""
import struct, sys, zlib
b = bytearray(open(sys.argv[1], "rb").read())
b[14] = 1; b[15] = 0x2A; b[26:28] = struct.pack("<H", 0xF)
b[-4:] = struct.pack("<I", zlib.crc32(bytes(b[:-4])) & 0xFFFFFFFF)
open(sys.argv[2], "wb").write(b)
print(sys.argv[2], len(b), "bytes")
