"""ROM image data of the Marigold core from a Marigold golden export (docs/golden-format.md), for rtl/core/fw/marigold.c.

    python3 rtl/core/tools/mkimage_mg.py data [--tag randtest]  -> data/core/mg-<tag>/{data.bin, layout.h}
    python3 rtl/core/tools/mkimage_mg.py rom  [--tag randtest]  -> data/core/mg-<tag>/imh_rom.svh (LUT + exp2 ROM)

Image (catimage.py fw.bin data.bin flash.bin DATA_BASE): "IMH1", firmware length (u32 LE), firmware (copied to PMEM
at boot), 0xFF up to DATA_BASE, data.bin. The firmware is at most 32 KB (PMEM), so DATA_BASE = 0x8040, the next
64-byte boundary after the largest header + firmware (Poinsetta: 0x10000).

Weight sites are laid out for the fan-out array (rtl/core/imh_fan.sv) with FAN_L = 16: per group of 16 output rows,
16 header records of 8 bytes {bias[39:0], M[15:0], sh[7:0]}, then for each input the 16 rows' weights (int8: 16 bytes;
int4: 8 bytes, low nibble first); rows past the last are zero. Order: fe.lin, fe.c1, fe.c2, enc<i>.{q,k,v,o,fc1,fc2},
proj, ctc. The frontend convolutions' columns are reordered from channel-major (c * 5 + tap) to tap-major
(tap * W + c), so that a conv input is 5 consecutive rows of the previous layer, contiguous in RAM. Norms (fe.cmvn,
enc<i>.ln1, enc<i>.ln2, enc.ln, out): 6-byte records {G int16, B int32 = 0} per channel. layout.h: dimensions and
windows (from the export's .json cfg, checked against the arrays), formats, the site and norm tables, and the LUT and
exp2 tables for the PC build. The chip's LUT ROM (rtl/core/imh_rom.svh) must hold this export's asinh table; `data`
warns when it does not, `rom` writes the matching file next to the image for the core agent to install.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import golden, ROOT

DATA_BASE = 0x8040
L = 16                       # fan-out rows per group (mg_core.sv FAN_L)
ENC = ["q", "k", "v", "o", "fc1", "fc2"]
ROM_SVH = ROOT / "rtl" / "core" / "imh_rom.svh"


def dims(G, meta):
    cfg = meta.get("cfg", {})
    W, FF = G["fe.lin.q"].shape[0], G["enc0.fc1.q"].shape[0]
    NL = sum(1 for k in G if k.startswith("enc") and k.endswith(".q.q"))
    D, V = G["proj.q"].shape[0], G["ctc.q"].shape[0]
    NH, HD = int(cfg.get("NH", 6)), int(cfg.get("HD", 40))
    PAST, FUT = int(cfg.get("PAST", 16)), [int(f) for f in cfg.get("FUT", [4] + [0] * (NL - 1))]
    for k, v in (("W", W), ("FF", FF), ("NL", NL), ("D", D), ("V", V)):
        assert k not in cfg or int(cfg[k]) == v, f"cfg {k} {cfg[k]} != arrays {v}"
    assert NH * HD == W and len(FUT) == NL and G["fe.lin.q"].shape[1] == 80
    assert G["fe.c1.q"].shape == (W, 5 * W) and G["fe.c2.q"].shape == (W, 5 * W)
    return dict(W=W, NH=NH, HD=HD, NL=NL, FF=FF, D=D, V=V, PAST=PAST, FUT=FUT)


def sites(NL):
    s = ["fe.lin", "fe.c1", "fe.c2"]
    for i in range(NL): s += [f"enc{i}.{k}" for k in ENC]
    return s + ["proj", "ctc"]


def norms(NL):
    n = ["fe.cmvn"]
    for i in range(NL): n += [f"enc{i}.ln1", f"enc{i}.ln2"]
    return n + ["enc.ln", "out"]


def tapmajor(q, C):
    """[out, C*5] channel-major (c * 5 + tap) -> tap-major (tap * C + c)."""
    return q.reshape(q.shape[0], C, 5).transpose(0, 2, 1).reshape(q.shape[0], 5 * C)


def pack_fan(q, M, sh, bi, wbits):
    out, nin = q.shape
    assert nin % 16 == 0 and M.min() >= 0 and M.max() < 65536 and sh.min() >= 1 and sh.max() < 62
    assert np.abs(bi).max() < 2 ** 39
    ng = (out + L - 1) // L
    pad = ng * L - out
    q = np.concatenate([q, np.zeros((pad, nin), q.dtype)]); M = np.concatenate([M, np.zeros(pad, np.int64)])
    sh = np.concatenate([sh, np.ones(pad, np.int64)]); bi = np.concatenate([bi, np.zeros(pad, np.int64)])
    hdr = np.zeros((ng * L, 8), np.uint8)
    b = bi.astype(np.int64) & ((1 << 40) - 1)
    for j in range(5): hdr[:, j] = (b >> (8 * j)) & 255
    hdr[:, 5] = M & 255; hdr[:, 6] = M >> 8; hdr[:, 7] = sh
    out_b = bytearray()
    for g in range(ng):
        out_b += hdr[g * L:(g + 1) * L].tobytes()
        w = q[g * L:(g + 1) * L].T                                      # [nin, L]
        if wbits == 8:
            assert w.min() >= -128 and w.max() <= 127
            out_b += w.astype(np.int8).tobytes()
        else:
            assert wbits == 4 and w.min() >= -8 and w.max() <= 7
            n = w.astype(np.int64) & 15
            out_b += (n[:, 0::2] | (n[:, 1::2] << 4)).astype(np.uint8).tobytes()
    return bytes(out_b)


def c_array(name, a, ctype):
    v = ", ".join(str(int(x)) for x in np.asarray(a).ravel())
    return f"static const {ctype} {name}[{np.asarray(a).size}] = {{{v}}};\n"


def rom_lines(G):
    """LUT ROM lines of tables 1 GELU, 2 SiLU, 3 asinh (R = 4) as rtl/core/imh_rom.svh writes them."""
    return ["        12'd%d: lut_rom = %s20'sd%d;" % (k * 1024 + j, "-" if int(v) < 0 else "", abs(int(v)))
            for k, t in ((1, "gelu"), (2, "silu"), (3, "asinh")) for j, v in enumerate(G["lut." + t])]


def rom_svh(G, tag):
    """The per-element unit's ROMs: table 0 kept from the current rtl/core/imh_rom.svh (Imhotep's tanh, unused here),
    1 GELU, 2 SiLU, 3 asinh of this export; exp2."""
    t0 = [l for l in ROM_SVH.read_text().splitlines() if "lut_rom = " in l and "12'd" in l
          and int(l.split("12'd")[1].split(":")[0]) < 1024]
    assert len(t0) == 513, "table 0 not found in " + str(ROM_SVH)
    for v in list(G["lut.gelu"]) + list(G["lut.silu"]) + list(G["lut.asinh"]): assert abs(int(v)) < 2 ** 19
    h = f"// generated by rtl/core/tools/mkimage_mg.py rom from mg-{tag}.npz: lut.tanh (Imhotep, unused), lut.gelu,\n"
    h += "// lut.silu, lut.asinh (this export's frontend scale), sm.exp2. Do not edit.\n"
    h += "function automatic signed [19:0] lut_rom(input [1:0] sel, input [9:0] idx);\n    case ({sel, idx})\n"
    h += "\n".join(t0 + rom_lines(G)) + "\n"
    h += "        default: lut_rom = 20'sd0;\n    endcase\nendfunction\n"
    h += "function automatic [16:0] exp2_rom(input [7:0] idx);\n    case (idx)\n"
    for j, v in enumerate(G["sm.exp2"]): h += "        8'd%d: exp2_rom = 17'd%d;\n" % (j, int(v))
    h += "        default: exp2_rom = 17'd0;\n    endcase\nendfunction\n"
    return h


def build(tag, outdir):
    G = dict(np.load(golden(f"mg-{tag}.npz")))
    assert int(G["version"]) == 1, "not a Marigold format-1 export"
    meta = json.load(open(golden(f"mg-{tag}.json")))
    dm = dims(G, meta)
    NL, W = dm["NL"], dm["W"]
    wb = meta.get("weight_bits", {})
    F = {k[2:]: int(v) for k, v in G.items() if k.startswith("F:")}
    for k in ("fe.cmvn", "fe.asinh", "fe.lin", "fe.c1") + tuple(f"enc{i}.fc1" for i in range(NL)):
        assert 0 <= F[k] <= 15, f"F[{k}] = {F[k]} does not fit the per-element unit's 4-bit LUT format field"
    if ROM_SVH.exists():
        have = ROM_SVH.read_text()
        if not all(l in have for l in rom_lines(G)):
            print(f"WARNING: {ROM_SVH.relative_to(ROOT)} is not the LUT ROM of mg-{tag}.npz (asinh / GELU / SiLU "
                  f"tables differ): the RTL cannot be bit-exact until it is replaced by `mkimage_mg.py rom --tag {tag}`",
                  file=sys.stderr)
    blob = bytearray()

    def put(b):
        while len(blob) % 64: blob.append(0)
        off = DATA_BASE + len(blob); blob.extend(b); return off

    st = []
    for w in sites(NL):
        q = G[w + ".q"].astype(np.int64)
        if w in ("fe.c1", "fe.c2"): q = tapmajor(q, W)
        bi = G[w + ".bi"] if w + ".bi" in G else np.zeros(q.shape[0], np.int64)
        bits = int(wb.get(w, 4 if np.abs(q).max() <= 7 else 8))
        assert bits in (4, 8), f"{w}: {bits}-bit weights"
        st.append((w, put(pack_fan(q, G[w + ".M"], G[w + ".sh"], bi, bits)), q.shape[1], q.shape[0], bits))
    nm = []
    for n in norms(NL):
        g, fg = G[n + ".G"].astype(np.int64), int(G[n + ".fg"])
        while np.abs(g).max() > 32767 and not (g & 1).any() and 12 + fg - F[n] > 1:
            g, fg = g >> 1, fg - 1         # e.g. gains of exactly 1.0 at fg 15: rshift_round(y * 2G, s) == rshift_round(y * G, s - 1)
            print(f"note: {n} gain halved to fit int16 (fg {fg}), bit-exact", file=sys.stderr)
        assert np.abs(g).max() <= 32767, f"{n}: gain over int16"
        eps = int(G[n + ".eps"]); assert 1 <= eps < 2 ** 32
        rec = np.zeros(len(g), dtype=[("g", "<i2"), ("b", "<i4")]); rec["g"] = g
        nm.append((n, put(rec.tobytes()), eps, fg, F[n]))
        assert 0 <= 12 + fg - F[n] < 64

    h = f"// generated by rtl/core/tools/mkimage_mg.py from mg-{tag}.npz; do not edit\n"
    h += f"#define DATA_BASE {DATA_BASE}u\n#define DATA_BYTES {len(blob)}u\n#define SM_C {int(G['sm.C'])}\n"
    for k in ("W", "NH", "HD", "NL", "FF", "D", "V", "PAST"):
        h += f"#define MG_{k} {dm[k]}\n"
    h += f"#define MG_FUT {', '.join(str(f) for f in dm['FUT'])}\n#define MG_FUTMAX {max(dm['FUT'])}\n"
    for k, v in sorted(F.items()):
        h += "#define F_%s %d\n" % (k.replace(".", "_").upper(), v)
    for a, s in (("FQ_E", "q"), ("FK_E", "k"), ("FF_E", "fc1")):
        h += f"static const signed char {a}[{NL}] = {{{', '.join(str(F[f'enc{i}.{s}']) for i in range(NL))}}};\n"
    h += f"#define S_PROJ {3 + 6 * NL}\n#define S_CTC {4 + 6 * NL}\n#define N_ENCLN {1 + 2 * NL}\n#define N_OUT {2 + 2 * NL}\n"
    h += "typedef struct { unsigned off; unsigned short nin, nout; unsigned char wbits; } site_t;\n"
    h += "static const site_t SITES[] = {\n" + "".join(
        f"  {{{o}u, {ni}, {no}, {b}}}, // {i} {w}\n" for i, (w, o, ni, no, b) in enumerate(st)) + "};\n"
    h += "typedef struct { unsigned off; unsigned eps; signed char fg, fout; } norm_t;\n"
    h += "static const norm_t NORMS[] = {\n" + "".join(
        f"  {{{o}u, {e}u, {fg}, {fo}}}, // {i} {n}\n" for i, (n, o, e, fg, fo) in enumerate(nm)) + "};\n"
    for t in ("gelu", "silu", "asinh"): h += c_array("LUT_" + t.upper(), G["lut." + t], "int")
    h += "#define LUT_TANH LUT_GELU      /* table 0 is Imhotep's tanh; this model does not use it */\n#define LUT_ASINH_N 513\n"
    h += c_array("EXP2", G["sm.exp2"], "unsigned")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "layout.h").write_text(h, newline="\n")
    (outdir / "data.bin").write_bytes(bytes(blob))
    nw = sum(ni * no for _, _, ni, no, _ in st)
    print(f"data.bin {len(blob)} bytes ({8 * (DATA_BASE + len(blob)) / 1e6:.2f} Mbit image with a 32 KB firmware slot), "
          f"{len(st)} sites, {nw} weights -> {outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["data", "rom"])
    ap.add_argument("--tag", default="randtest")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = Path(a.out) if a.out else ROOT / "data" / "core" / f"mg-{a.tag}"
    if a.what == "rom":
        out.mkdir(parents=True, exist_ok=True)
        (out / "imh_rom.svh").write_text(rom_svh(dict(np.load(golden(f"mg-{a.tag}.npz"))), a.tag), newline="\n")
        print("wrote", out / "imh_rom.svh", f"(install as {ROM_SVH.relative_to(ROOT)} for a bit-exact RTL)")
    else:
        build(a.tag, out)
