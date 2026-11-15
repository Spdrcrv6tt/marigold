"""Shared fixed-point helpers. Every RTL block follows these exactly."""


def sat(v, bits):
    """Saturate a Python int to a signed two's-complement range of `bits`."""
    hi = (1 << (bits - 1)) - 1
    lo = -(1 << (bits - 1))
    return hi if v > hi else lo if v < lo else v


def wrap(v, bits):
    """Wrap a Python int to signed two's complement of `bits` (RTL truncation)."""
    m = 1 << bits
    v &= m - 1
    return v - m if v >= (m >> 1) else v


def rshift_round(v, sh):
    """Arithmetic right shift, round half up (add half, floor). sh <= 0 shifts left."""
    if sh <= 0:
        return v << (-sh)
    return (v + (1 << (sh - 1))) >> sh


def read_mem(path):
    """Read a $readmemh file of signed 16-bit words."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.split("//")[0].strip()
            if line:
                v = int(line, 16)
                out.append(v - 65536 if v >= 32768 else v)
    return out
