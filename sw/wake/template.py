"""Wake template: typed phrase -> template bytes and wake_sens (docs/golden-wake.md, section Template).

Bytes: [0] format 1, [1] n_var (1..4), [2] clip (per-frame deficit cap, 1/16 nat), [3] refractory
(network frames of 20 ms), then per variant: K (1..30) followed by K phone ids (1..39, lexicon.PHONES).
Total phones over all variants <= 60, total length <= 255.
wake_sens (u16): allowed mean deficit per phone, 1/16 nat. The SDK maps the config's sensitivity
s in [0, 1] linearly between SENS_LO (s = 0, strict) and SENS_HI (s = 1, permissive).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lexicon import phrase_prons, PHONES, PID

FORMAT, MAX_VAR, MAX_K, MAX_PHONES = 1, 4, 30, 60
CLIP, REFRACTORY = 96, 50          # calibrated values, see docs/golden-wake.md
SENS_LO, SENS_HI = 8, 40           # wake_sens at sensitivity 0 and 1 (1/16 nat per phone)


def sens_to_u16(s):
    s = min(max(float(s), 0.0), 1.0)
    return int(round(SENS_LO + s * (SENS_HI - SENS_LO)))


def build(phrase, sensitivity=0.5, phones=None, max_var=MAX_VAR):
    """Returns (template bytes, wake_sens, report dict). `phones` (list of space-separated ARPAbet
    strings) overrides the lexicon, for OEM respellings."""
    if phones:
        seqs, notes = [tuple(PID[p.upper()] for p in s.split()) for s in phones], ["phones given by the OEM"]
    else:
        seqs, notes = phrase_prons(phrase, max_var)
    seqs = [s for s in seqs if s][:max_var]
    if not seqs:
        raise ValueError(f"wake phrase {phrase!r} has no pronounceable words")
    out, total = [], 0
    for s in seqs:
        if len(s) > MAX_K:
            raise ValueError(f"wake phrase too long: {len(s)} phones (max {MAX_K})")
        if total + len(s) > MAX_PHONES:
            break
        out.append(s); total += len(s)
    if min(len(s) for s in out) < 4:
        notes.append("short wake phrase (< 4 phones): expect more false wakes")
    b = bytearray([FORMAT, len(out), CLIP, REFRACTORY])
    for s in out:
        b.append(len(s)); b += bytes(s)
    assert len(b) <= 255
    rep = dict(phrase=phrase, variants=[" ".join(PHONES[p - 1] for p in s) for s in out], notes=notes,
               sensitivity=sensitivity, wake_sens=sens_to_u16(sensitivity), clip=CLIP, refractory=REFRACTORY)
    return bytes(b), sens_to_u16(sensitivity), rep


def parse(b):
    assert b[0] == FORMAT, "unknown wake template format"
    nv, clip, refr = b[1], b[2], b[3]
    o, var = 4, []
    for _ in range(nv):
        k = b[o]; var.append(list(b[o + 1:o + 1 + k])); o += 1 + k
    assert o == len(b), "trailing bytes in wake template"
    return dict(variants=var, clip=clip, refractory=refr)


if __name__ == "__main__":
    for p in sys.argv[1:] or ["hey breeze", "hey kitchen"]:
        t, s, r = build(p)
        print(t.hex(), s, r)
