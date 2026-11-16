"""Golden test vectors for rtl/wake (cocotb, rtl/wake/tb/test_wake.py).

  vectors.py PARAMS.npz OUT_DIR
Each case directory holds: x.txt (int16 samples, one per line), case.json (en segments, sens), cfg.mem (config
SRAM words with the template at byte 0x123) and expected.txt: the trace the RTL must write, per processed frame
64 feature lines "0 <b<<14 | band<<8 | v>" (band-major, frame a then b), 40 logit lines "1 <idx<<16 | z & 0xffff>",
one score line "2 <best>", and "W <n>" for every wake (n = sample index within the en segment).
"""
import json, sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import golden
import template as TPL
from lexicon import R

TPL_ADDR = 0x123


def cfg_mem(tpl):
    mem = bytearray(8192 * 4)
    mem[TPL_ADDR:TPL_ADDR + len(tpl)] = tpl
    return "\n".join(f"{int.from_bytes(mem[4 * i:4 * i + 4], 'little'):08x}" for i in range(8192)) + "\n"


def vad_flags(x):
    sys.path.insert(0, str(R / "rtl/model"))
    from vad_model import Vad
    return Vad().run(np.asarray(x, np.int64))["vad"].astype(np.int64)


def expected(x, segs, tpl, sens, P, fmt=1):
    lines = []
    for a, b in segs:
        if fmt == 2:
            return expected2(x, a, b, tpl, sens, P, lines)
        r = golden.run(x[a:b], tpl, sens, P)
        U, z, best = r["u"], r["z"], r["best"]
        fires = set(r["fires"])
        for t in range(len(U)):
            v = U[t].astype(int)
            for band in range(32):
                lines.append(f"0 {(0 << 14) | (band << 8) | v[band]}")
                lines.append(f"0 {(1 << 14) | (band << 8) | v[32 + band]}")
            for c in range(40):
                lines.append(f"1 {(c << 16) | (int(z[t, c]) & 0xFFFF)}")
            lines.append(f"2 {int(best[t])}")
            k = 320 * t + 879
            if t in fires and k < b - a:
                lines.append(f"W {k}")
    return lines


def expected2(x, a, b, tpl, sens, P, lines):
    """Format 2 trace: per real-time frame t' the 64 features, then (frame t = t' - PREROLL active) its logits and
    score, then 'W k' for a fire of frame t (k = 320 (t + PREROLL) + 879)."""
    D = golden.PREROLL
    vad = vad_flags(x[a:b])
    r = golden.run2(x[a:b], vad, tpl, sens, P)
    U, z, best, act = r["u"], r["z"], r["best"], r["act"]
    fires = set(r["fires"])
    for tp in range(len(U)):
        v = U[tp].astype(int)
        for band in range(32):
            lines.append(f"0 {(0 << 14) | (band << 8) | v[band]}")
            lines.append(f"0 {(1 << 14) | (band << 8) | v[32 + band]}")
        t = tp - D
        if t >= 0 and act[t]:
            for c in range(40):
                lines.append(f"1 {(c << 16) | (int(z[t, c]) & 0xFFFF)}")
            lines.append(f"2 {int(best[t])}")
            k = 320 * (t + D) + 879
            if t in fires and k < b - a:
                lines.append(f"W {k}")
    return lines, vad


def main(params, out, fmt=1):
    P = golden.load_params(params)
    from export import rom_words                      # the RTL ROM image must hold exactly these parameters
    memdir = "rtl/wake/mem" if fmt == 1 else "rtl/wake/mem2"
    rom = [int(l, 16) for l in open(R / memdir / "wake_rom.mem")]
    lanes = int(np.load(params)["lanes"]) if "lanes" in np.load(params).files else 4
    assert rom == rom_words(P, lanes), "rtl/wake/mem/wake_rom.mem does not match the golden parameters"
    print("ROM image matches the golden parameters:", len(rom), "words")
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    clip = lambda p: sf.read(str(R / p), dtype="int16")[0].astype(np.int64)
    man = [json.loads(l) for l in open(R / "data/wake/tts/manifest_kokoro.jsonl", encoding="utf-8")]
    hb = [m for m in man if m["phrase"] == "hey breeze" and m["kind"] == "isolated"][0]
    hl = [m for m in man if m["phrase"] == "hello lumen" and m["kind"] == "isolated"][1]
    def stream(m, lead=9000, tail=9000, gain=0.3):
        c = clip(m["path"]) * gain
        n = rng.standard_normal(lead + len(c) + tail) * 60
        x = n.copy(); x[lead:lead + len(c)] += c
        return np.clip(np.round(x), -32768, 32767).astype(np.int64)
    cases = []
    def add(name, x, segs, phrase, sens, refr):
        tpl = bytearray(TPL.build(phrase)[0]); tpl[3] = refr
        tpl = bytes(tpl)
        d = out / name; d.mkdir(exist_ok=True)
        np.savetxt(d / "x.txt", x, fmt="%d")
        (d / "cfg.mem").write_text(cfg_mem(tpl))
        json.dump(dict(segs=segs, sens=int(sens), tpl=tpl.hex(), phrase=phrase), open(d / "case.json", "w"))
        if fmt == 2:
            e, vads = [], []
            for a, b in segs:
                l2, v2 = expected2(x, a, b, tpl, sens, P, [])
                e += l2; vads.append([int(v) for v in v2])
            json.dump(vads, open(d / "vad.json", "w"))
        else:
            e = expected(x, segs, tpl, sens, P)
        (d / "expected.txt").write_text("\n".join(e) + "\n", newline="\n")
        nw = sum(l.startswith("W") for l in e)
        cases.append(name)
        print(name, len(x), "samples", segs, "lines", len(e), "wakes", nw, flush=True)
    x1 = stream(hb)
    # sens chosen so the golden model fires: the case checks scores, resets and the refractory count
    if fmt == 2:
        r = golden.run2(x1, vad_flags(x1), TPL.build("hey breeze")[0], 0, P)
        need = int(-r["best"][r["act"]].max() // 6 + 1)
    else:
        r = golden.run(x1, TPL.build("hey breeze")[0], 0, P)
        need = int(-r["best"].max() // 6 + 1)
    add("breeze", x1, [[0, len(x1)]], "hey breeze", max(need, 1), 10)
    x2 = stream(hl, 4000, 6000)
    add("lumen_segments", x2, [[0, 20000], [20500, len(x2)]], "hello lumen", 400, 5)
    # the SDK's own template and sensitivity 0.5 (the operating point) on a wake phrase followed by a command
    hc = [m for m in man if m["phrase"] == "hey breeze" and m["kind"] == "carrier"][0]
    x3 = stream(hc, 8000, 12000)
    t3, s3, _ = TPL.build("hey breeze", 0.5)
    add("carrier_op", x3, [[0, len(x3)]], "hey breeze", s3, t3[3])
    json.dump(cases, open(out / "cases.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)
