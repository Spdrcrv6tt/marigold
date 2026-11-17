"""Check the Marigold firmware's PC build (rtl/core/fw/marigold.c + hal_host.c, `make host-mg`) against the integer
golden model (sw/golden/intent_np.py IntIntent, sw/intent/values.py) bit for bit.

    .venv/Scripts/python.exe rtl/core/tools/check_mg.py [--tag randtest] [--n 6] [--exact] [--fuzz 20000] [--no-build]
        [--golden-dir data/core/mg-rand3 --bdir '$CORE_BUILD/r3']    (a mkrand_mg.py export, in its own build dir)
intent_np's module dimensions are set from the export, so draft-0.1 (randtest) and 0.3 exports both check.

1. Builds the image data and the PC firmware in WSL (make -C rtl/top TAG=<tag> host-mg).
2. A test config blob with the model's D (random vectors, one slot of each type, level words on one slot).
3. Runs the firmware on --n LibriSpeech test-clean utterances of 1 to 4 s (the segment as delivered by the PCM port,
   then an end marker), with the chip's RAM (256 KB less the 4 KB stack and 4 KB for .bss), and compares every trace
   with the golden model: per-row hashes of fe.cmvn, fe.asinh, fe.lin, fe.c1, fe.feat, enc<i>.out, enc.ln, then pool.m,
   proj, q, ctc.k (per frame), the tokens, the raw slot values, and the island register writes (Q words, VAL[0..7],
   MATCH_GO). One more run with the TIMEOUT end marker must write nothing to the island. Then segments cut at the
   streaming boundaries (EDGES: 1 sample to the 6 s segment cap; --no-edges skips them).
4. Value-parse fuzz: the firmware's parse (rtl/core/fw/mg_values.h, compiled in a small driver) against raw_values on
   --fuzz token sequences (phrases through values.targets, plus random sequences).
Prints identical / first difference per utterance and a summary; exit status 1 on any difference.
"""
import argparse, json, random, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
WROOT = str(ROOT) if str(ROOT).startswith("/") else "/mnt/" + str(ROOT)[0].lower() + str(ROOT)[2:].replace("\\", "/")
sys.path.insert(0, str(ROOT / "sw" / "golden")); sys.path.insert(0, str(ROOT / "sw" / "intent"))
sys.path.insert(0, str(ROOT / "rtl" / "model")); sys.path.insert(0, str(ROOT / "sdk"))
import intent_np as S  # noqa: E402
import values as VAL  # noqa: E402
import mg_model as MG  # noqa: E402

HMUL = 16777619
K = dict(CMVN=1, ASINH=2, LIN=3, C1=4, FEAT=5, ENC=10, ELN=20, CTCK=21, POOL=22, PROJ=23, Q=24, TOK=25, VAL=26,
         SEG=40, END=41, T=42, NTOK=43, MATCH=44, DONE=45, CFG=46)
ISL = 0x7F0
RAM_HOST = 262144 - 4096 - 4096


def set_dims(G, meta):
    """intent_np's module dimensions for this export (draft 0.1 randtest: W 240, 6 layers; 0.3: W 320, 2 layers)."""
    cfg = meta.get("cfg", {})
    S.W, S.FF = G["fe.lin.q"].shape[0], G["enc0.fc1.q"].shape[0]
    S.NL = sum(1 for k in G if k.startswith("enc") and k.endswith(".q.q"))
    S.HD = int(cfg.get("HD", 40)); S.NH = S.W // S.HD
    S.WIN = [(int(cfg.get("PAST", 16)), int(f)) for f in cfg.get("FUT", [4] + [0] * (S.NL - 1))]


def wsl(script, cwd_tmp):
    """Run a bash script file in WSL (wsl.exe mangles $ and quotes on its command line)."""
    p = cwd_tmp / f"run-{time.time_ns()}.sh"
    p.write_text(". " + WROOT + "/rtl/core/tools/env.sh\n" + script + "\n", newline="\n")
    r = subprocess.run(["wsl.exe", "-d", "Ubuntu", "--", "bash", WROOT + "/" + p.relative_to(ROOT).as_posix()],
                       capture_output=True, text=True, timeout=1800)
    p.unlink()
    return r


def wpath(p):
    return WROOT + "/" + Path(p).resolve().relative_to(ROOT).as_posix()


def row_hash(a):
    a = np.asarray(a, np.int64)
    if a.ndim == 1: a = a[None]
    n = a.shape[1]
    P = np.ones(n, np.uint64)
    for i in range(n - 2, -1, -1): P[i] = (int(P[i + 1]) * HMUL) & 0xFFFFFFFF
    v = a.astype(np.uint64) & np.uint64(0xFFFFFFFF)
    return [int(x) & 0xFFFFFFFF for x in (v * P[None]).sum(1, dtype=np.uint64)]


def test_blob(D, seed=7):
    rng = np.random.default_rng(seed)
    acts = [(1, 0b0001, 0, 0), (2, 0b0010, 0, 0), (3, 0b0100, 0, 0), (4, 0b1000, 0, 0), (5, 0b1111, 0, 0), (6, 0, 0, 0)]
    vec = []
    for i in range(24):
        a = 0 if i % 6 == 5 else acts[i % 6][0]
        vec.append((a, 0, int(rng.integers(-500, 500)), [int(x) for x in np.clip(rng.normal(0, 32, D).round(), -127, 127)]))
    slots = [(0, 0, 500), (1, 1, 5999), (2, 1, 10, (3, 0, 10)), (2, 1, 5)]
    cfg = dict(D=D, vectors=vec, actions=acts, slots=slots, gpio=[], wake=b"\x01\x02", host=1, i2c_addr=0x2A,
               reject_thr=-(1 << 30), margin=0, wake_sens=0, flags=0, conf_sh=9 if D <= 128 else 10)
    blob = MG.build_blob(cfg)
    err, c = MG.parse_blob(blob)
    assert err == MG.ERR_NONE, err
    specs = [dict(type=t, min=mn, max=mx) for t, mn, mx, _ in c["slots"]]
    return blob, specs


def utterances(n, lo=16000, hi=64000):
    import soundfile as sf
    base = ROOT / "data" / "librispeech" / "LibriSpeech" / "test-clean"
    out = []
    for f in sorted(base.rglob("*.flac")):
        if lo <= sf.info(str(f)).frames <= hi:
            x, sr = sf.read(str(f), dtype="int16"); assert sr == 16000
            out.append((f.stem, x.astype(np.int64)))
            if len(out) == n: break
    return out


# segment lengths around the streaming boundaries (frame 80, chunk 1280, the look-ahead) and the 6 s segment cap
EDGES = [1, 79, 80, 1279, 1280, 1281, 2560, 3839, 6400, 96000]


def edge_segments():
    import soundfile as sf
    base = ROOT / "data" / "librispeech" / "LibriSpeech" / "test-clean"
    f = next(f for f in sorted(base.rglob("*.flac")) if sf.info(str(f)).frames >= max(EDGES) + 16000)
    x = sf.read(str(f), dtype="int16")[0].astype(np.int64)[8000:]
    return [(f"{f.stem}[:{L}]", x[:L]) for L in EDGES]


def read_events(p):
    ev, isl = {}, []
    for line in open(p):
        t, v = (int(x, 16) for x in line.split())
        if t >> 20 == ISL: isl.append((t & 0xFFFF, v))
        else: ev.setdefault(t >> 20, {})[t & 0xFFFFF] = v
    return ev, isl


def s32(v): return v - (1 << 32) if v >= 1 << 31 else v


def compare(ev, isl, tr, q, toks, raw, D):
    """-> (number of items compared, first difference or None)"""
    n = 0
    rows = [("fe.cmvn", K["CMVN"]), ("fe.asinh", K["ASINH"]), ("fe.lin", K["LIN"]), ("fe.c1", K["C1"]),
            ("fe.feat", K["FEAT"])] + [(f"enc{i}.out", K["ENC"] + i) for i in range(S.NL)] + [("enc.ln", K["ELN"])]
    for name, k in rows:
        want = row_hash(tr[name]); got = ev.get(k, {})
        if len(got) != len(want): return n, f"{name}: {len(got)} rows, golden {len(want)}"
        for r, h in enumerate(want):
            if got.get(r) != h: return n, f"{name} row {r} differs (first)"
        n += len(want)
    for name, k, want in (("pool.m", K["POOL"], tr["pool.m"]), ("proj", K["PROJ"], tr["proj"]), ("q", K["Q"], q),
                          ("ctc.k", K["CTCK"], tr["ctc.k"]), ("tokens", K["TOK"], toks)):
        got = ev.get(k, {})
        want = [int(x) for x in np.asarray(want).ravel()]
        if len(got) != len(want): return n, f"{name}: {len(got)} values, golden {len(want)}"
        for j, w in enumerate(want):
            if s32(got[j]) != w: return n, f"{name}[{j}] = {s32(got[j])}, golden {w}"
        n += len(want)
    got = [s32(ev.get(K["VAL"], {}).get(j, 0)) for j in range(8)]
    want = [min(v, 2 ** 31 - 1) for v in raw] + [0] * (8 - len(raw))
    if got != want: return n, f"raw values {got}, golden {want}"
    n += 8
    words = [sum((int(q[4 * w + j]) & 255) << (8 * j) for j in range(4) if 4 * w + j < D) for w in range((D + 3) // 4)]
    exp = [(0x100 + 4 * w, x) for w, x in enumerate(words)] + [(0x40 + 4 * j, v & 0xFFFFFFFF) for j, v in enumerate(want)]
    exp += [(0x08, 1), (0xFFFC, 8)]
    if isl != exp:
        bad = next((j for j, (a, b) in enumerate(zip(isl, exp)) if a != b), min(len(isl), len(exp)))
        return n, f"island writes differ at #{bad}: {isl[bad] if bad < len(isl) else None} vs {exp[bad] if bad < len(exp) else None}"
    return n + len(exp), None


VTEST = r"""
#include <stdio.h>
#include "mg_values.h"
static u8 scratch[MG_VALUES_SCRATCH(4096)] __attribute__((aligned(8)));
int main(void) {
    unsigned ns, nt, t; u8 types[8], tok[4096]; i32 raw[8];
    while (scanf("%u", &ns) == 1) {
        for (unsigned s = 0; s < ns; s++) { scanf("%u", &t); types[s] = (u8)t; }
        scanf("%u", &nt);
        for (unsigned j = 0; j < nt; j++) { scanf("%u", &t); tok[j] = (u8)t; }
        mg_raw_values(tok, nt, types, ns, raw, scratch);
        for (unsigned s = 0; s < ns; s++) printf("%d%c", raw[s], s + 1 < ns ? ' ' : '\n');
    }
    return 0;
}
"""


def fuzz_cases(n, seed=11):
    rng = random.Random(seed)
    num = lambda: rng.choice([rng.randint(0, 20), rng.randint(20, 99), rng.randint(100, 999), rng.randint(1000, 99999)])
    tmpl = ["set it to {n}", "{n} degrees", "{n} percent", "for {n} minutes", "{n} seconds", "{n} hours and {n} minutes",
            "{n} and a half minutes", "a minute and a half", "half a minute", "an hour", "a quarter hour",
            "one minute {n}", "{n}:{s}", "level {n}", "speed {n}", "go to {w}", "on {w}", "half power", "{n} {n}",
            "heat it for {n} minutes on {w}", "{n} minutes at power level {n}", "three fifty", "two five", "{n} thousand",
            "a half", "{w} {n} percent", "{n} a half minute", "level level {n}", "{n} quarter hour", "min", "max {n}"]
    words = ["low", "medium", "high", "max", "min", "middle", "full", "lowest"]
    cases = []
    for k in range(n):
        if k % 3 == 0:
            toks = [rng.randint(1, 44) for _ in range(rng.randint(0, 14))]
        elif k % 3 == 1:
            pool = [rng.randint(1, 30) for _ in range(6)] + list(range(29, 45))
            toks = [rng.choice(pool) for _ in range(rng.randint(0, 10))]
        else:
            t = " ".join(rng.choice(tmpl) for _ in range(rng.randint(1, 3)))
            t = t.replace("{s}", f"{rng.randint(0, 59):02d}")
            while "{n}" in t: t = t.replace("{n}", str(num()), 1)
            while "{w}" in t: t = t.replace("{w}", rng.choice(words), 1)
            toks = VAL.targets(t)
        types = [rng.choice([0, 1, 2, 3]) for _ in range(rng.randint(1, 8))]
        cases.append((types, toks))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="randtest")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--exact", action="store_true", help="golden with FAST = False (int64 only)")
    ap.add_argument("--fuzz", type=int, default=20000)
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--no-edges", dest="edges", action="store_false", help="skip the boundary-length segments")
    ap.add_argument("--golden-dir", default="", help="directory of mg-<tag>.npz/.json (default data/golden; "
                    "mkrand_mg.py exports: data/core/mg-<tag>)")
    ap.add_argument("--bdir", default="$CORE_BUILD", help="WSL build directory of fw_host_mg (make B=...)")
    a = ap.parse_args()
    gdir = Path(a.golden_dir) if a.golden_dir else ROOT / "data" / "golden"
    tmp = ROOT / "data" / "tmp" / f"mg-check-{a.tag}"
    tmp.mkdir(parents=True, exist_ok=True)
    if not a.no_build:
        r = wsl(f"make -C {WROOT}/rtl/top TAG={a.tag} MGGOLD={wpath(gdir)} B={a.bdir} host-mg 2>&1 | grep -v '^note:'", tmp)
        print(r.stdout[-1500:], r.stderr[-500:])
    G = dict(np.load(gdir / f"mg-{a.tag}.npz"))
    D = int(G["proj.q"].shape[0])
    set_dims(G, json.load(open(gdir / f"mg-{a.tag}.json")))
    S.IntIntent.FAST = not a.exact
    blob, specs = test_blob(D)
    (tmp / "blob.bin").write_bytes(blob)
    utts = utterances(a.n)
    img = f"{WROOT}/data/core/mg-{a.tag}/data.bin"
    lines = []
    runs = [(name, x, 0) for name, x in utts] + [(utts[0][0] + "-timeout", utts[0][1], 2)]
    if a.edges: runs += [(name, x, 0) for name, x in edge_segments()]
    for k, (name, x, endf) in enumerate(runs):
        x.astype("<i2").tofile(tmp / f"pcm-{k}.raw")
        lines.append(f"{a.bdir}/fw_host_mg {img} {wpath(tmp / f'pcm-{k}.raw')} {wpath(tmp / f'ev-{k}.txt')} 1 {RAM_HOST} "
                     f"{wpath(tmp / 'blob.bin')} {endf} > /dev/null 2> {wpath(tmp / f'err-{k}.txt')}; echo rc$?")
    t0 = time.time()
    r = wsl("\n".join(lines), tmp)
    rcs = [l for l in r.stdout.split() if l.startswith("rc")]
    print(f"firmware: {len(rcs)} runs in {time.time() - t0:.1f} s; exit codes {' '.join(rcs)}")
    ok = 0; peak = 0
    for k, (name, x, endf) in enumerate(runs):
        err = (tmp / f"err-{k}.txt").read_text().strip()
        if rcs[k] != "rc0":
            print(f"{name}: firmware failed ({rcs[k]}): {err[-300:]}"); continue
        ev, isl = read_events(tmp / f"ev-{k}.txt")
        peak = max(peak, ev[K["DONE"]][0])
        if endf & 2:
            good = isl == [(0xFFFC, 8)] and K["T"] not in ev
            print(f"{name}: TIMEOUT run {'writes nothing to the island, powers off' if good else 'WRONG: ' + str(isl[:4])}")
            ok += good; continue
        tr = {}
        I = S.IntIntent(G, tr)
        q, kk = I.run(x)
        toks = VAL.ctc_greedy(kk)
        raw = VAL.raw_values(toks, specs)
        n, diff = compare(ev, isl, tr, q, toks, raw, D)
        T = int(tr["enc.ln"].shape[0])
        print(f"{name}: {len(x) / 16000:.2f} s, T {T}, {len(toks)} tokens, raw {raw}: "
              + ("identical" if diff is None else "FIRST DIFFERENCE " + diff) + f" ({n} items compared)")
        ok += diff is None
    print(f"runs identical (utterances, TIMEOUT, boundary lengths): {ok} / {len(runs)}; firmware heap peak {peak} bytes "
          f"(+ .bss + 4 KB stack vs 262144)")
    # value-parse fuzz
    fz = 0
    if a.fuzz:
        (tmp / "vtest.c").write_text(VTEST, newline="\n")
        cases = fuzz_cases(a.fuzz)
        (tmp / "vtest-in.txt").write_text("".join(f"{len(t)} {' '.join(map(str, t))} {len(s)} {' '.join(map(str, s))}\n"
                                                  for t, s in cases), newline="\n")
        r = wsl(f"gcc -O2 -I{WROOT}/rtl/core/fw -o $CORE_BUILD/mg_vtest {wpath(tmp / 'vtest.c')} && "
                f"$CORE_BUILD/mg_vtest < {wpath(tmp / 'vtest-in.txt')} > {wpath(tmp / 'vtest-out.txt')}; echo rc$?", tmp)
        got = (tmp / "vtest-out.txt").read_text().splitlines()
        bad = None
        for j, (types, toks) in enumerate(cases):
            want = [min(v, 2 ** 31 - 1) for v in VAL.raw_values(toks, [dict(type=t, min=0, max=0) for t in types])]
            g = [int(v) for v in got[j].split()] if j < len(got) else None
            if g == want: fz += 1
            elif bad is None: bad = (types, [VAL.VOCAB[t] for t in toks], g, want)
        print(f"value parse fuzz: {fz} / {len(cases)} identical" + ("" if bad is None else f"; first difference {bad}"))
    sys.exit(0 if ok == len(runs) and fz == a.fuzz else 1)


if __name__ == "__main__":
    main()
