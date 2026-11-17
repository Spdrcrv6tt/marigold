"""Whole-chip end-to-end check against the golden model: PDM microphone pin in; config over I2C; event frames (read
over I2C by the harness) and the core's island writes (q and raw slot values per segment) compared with
sw/golden/chip_golden.py run_stream on the chip's own input (the bit-exact front-end and VAD models of the bits).
Wakes: the stand-in spotter at chosen samples (--stub, default) or the wake spotter itself (--wakes from a file the
wake golden model wrote). Negative controls must fail: --neg rom (one weight bit flipped in the ROM image), mic (PDM
bits flipped), crc (config CRC corrupted), model (the blob stamped for another model: the chip must refuse it with
STATUS cfg error 7; K1). Every blob is stamped with the model and teacher ids of the chip under test (--model-tag,
default the tracked LUT ROM's model, as rtl/top/chip.mk builds it): these checks are about bit-exactness, and the test
blob rtl/top/tb/fan-i2c.mgc is a model-v1 blob. Run in WSL after rtl/core/tools/env.sh:

    python3 rtl/top/tb/mg_e2e.py --sim <sim> --img <flash.hex> --golden data/golden/mg-<tag>.npz --out <dir> [--neg ...]
"""
import argparse, glob, json, random, struct, subprocess, sys, zlib
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[3]
for p in ("rtl/model", "rtl/tb", "sw/golden", "sw/wake/golden", "sw/wake", "sdk"):
    sys.path.insert(0, str(ROOT / p))
import pdm_model, mg_model as M
from vad_model import Vad
from mg_util import rand_cfg
import chip_models as CM

ap = argparse.ArgumentParser()
ap.add_argument("--sim", required=True)
ap.add_argument("--img", required=True)
ap.add_argument("--golden", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--neg", default="")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--trace", type=int, default=1, help="the trace strap (+trace=); 0 = the product path")
ap.add_argument("--wake2", type=int, default=-1, help="R2: an extra stand-in wake this many samples after the "
                "first segment closes (the golden model is rerun with it)")
ap.add_argument("--blob", default="", help="config blob (default: a random valid one, D from the model)")
ap.add_argument("--model-tag", default="", help="the chip's model (sdk/chip_models.py; default the tracked LUT ROM's)")
ap.add_argument("--wakes", default="", help="'golden': wakes from the wake golden model (spotter build); a file of "
                "samples; default: chosen samples for the stand-in spotter build")
ap.add_argument("--audio", default="", help="comma-separated audio (.txt int16 per line, or sound files) instead of the "
                "LibriSpeech stimulus; --gaps seconds of noise before each and after the last")
ap.add_argument("--gaps", default="0.6,1.2,5.5,0.8")
ap.add_argument("--silent", type=int, default=-1, help="with --audio: index of a gap that gets a wake phrase "
                "copy (the first audio's first 1.2 s) and then silence, for a TIMEOUT case (-1: none)")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
MID, TID = CM.ids(a.model_tag) if a.model_tag else CM.rom_ids()


def stamp(b, mid, tid):
    """The blob with model and teacher ids mid, tid (bytes 0x1D, 0x1E) and its CRC recomputed."""
    b = bytearray(b)
    b[0x1D], b[0x1E] = mid, tid
    b[-4:] = struct.pack("<I", zlib.crc32(bytes(b[:-4])) & 0xFFFFFFFF)
    return bytes(b)

rng = random.Random(a.seed); nrng = np.random.default_rng(a.seed)

# ---- stimulus (as mg_smoke.py): noise, two utterances, a long gap with a wake and no speech (TIMEOUT), noise
LS = ROOT / "data" / "librispeech" / "LibriSpeech" / "test-clean"
files = sorted(glob.glob(str(LS / "*" / "*" / "*.flac")))
utts = [f for f in files if 2.0 < sf.info(f).duration < 3.5][:2]
parts, marks, pos = [], [], 0
def load_audio(f):
    if f.endswith(".txt"):
        return np.loadtxt(f, dtype=np.float64)
    return sf.read(f, dtype="float64")[0]


plan = (0.6, "u", 1.2, "u", 5.5, 0.8)
if a.audio:
    srcs = a.audio.split(",")
    gs = [float(g) for g in a.gaps.split(",")]
    plan = [x for i, f in enumerate(srcs) for x in (gs[i], "u")] + gs[len(srcs):]
    utts = srcs
for seg in plan:
    if isinstance(seg, str):
        x = load_audio(utts[len(marks)])
        x = x / np.abs(x).max() * 0.5
        marks.append(pos); parts.append(x); pos += len(x)
    else:
        parts.append(nrng.standard_normal(int(seg * 16000)) * 0.004); pos += int(seg * 16000)
gap = marks[1] + len(parts[3]) + 16000 if len(marks) > 1 else 0
import hashlib, os                       # the cache file names its stimulus: checks of different stimuli share a dir
key = hashlib.sha1(("%s|%s|%d" % (a.audio, a.gaps if a.audio else "libri", a.seed)).encode()).hexdigest()[:12]
cache = out.parent / ("stim-%s.npz" % key)
if cache.exists():
    z = np.load(cache); bits, ref = z["bits"].copy(), z["ref"]
else:
    bits = pdm_model.sdm(np.clip(pdm_model.upsample_to_pdm(np.concatenate(parts), 192), -0.9, 0.9)).astype(np.uint8)
    ref = np.array(pdm_model.pdm_decimate(bits), dtype=np.int64)
    tmp = out / "stim-tmp.npz"
    np.savez(tmp, bits=bits, ref=ref)
    os.replace(tmp, cache)                 # atomic: a parallel check never reads a half-written file
raw = np.asarray(Vad().run(ref)["raw"])
if a.wakes == "golden":
    import golden as WG                    # sw/wake/golden (frozen format 1)
    _, c0 = M.parse_blob(open(a.blob, "rb").read())
    vflag = np.asarray(Vad().run(ref)["vad"])           # format 2: the spotter is gated by vad_o
    wakes = WG.run2(ref, vflag, bytes(c0["wake"]), c0["wake_sens"],
                    WG.load_params(str(ROOT / "sw/wake/golden/wake_int_v2.npz")))["wake_k"]
elif a.wakes:
    wakes = [int(w) for w in open(a.wakes).read().split()]
else:
    wakes = [marks[0] + 6400, marks[1] + 6400, gap]
if a.wake2 >= 0:
    _sg = M.Segmenter()
    for _w in range(len(ref)):
        _sg.step(_w, _w in wakes, int(raw[_w // 256 - 1]) if (_w > 0 and _w % 256 == 0) else 0)
    wakes = sorted(set(wakes) | {_sg.segs[0][1] + a.wake2})
    print("wake2 at", _sg.segs[0][1] + a.wake2, "after the first close at", _sg.segs[0][1])
open(out / "wakes.txt", "w").write("\n".join(map(str, wakes)) + "\n")

# ---- config and the golden run (on the clean input; the negative controls change only the chip's side)
G = dict(np.load(a.golden))
D = int(G["proj.q"].shape[0])
if a.blob:
    blob = open(a.blob, "rb").read()
    if blob[0x1D:0x1F] != bytes([MID, TID]):
        print("blob built for model id %d, teacher id %d: stamped for the chip under test (%d, %d)" % (blob[0x1D], blob[0x1E], MID, TID))
        blob = stamp(blob, MID, TID)
else:
    cfg = rand_cfg(rng, D=D, N=12, n_actions=4, n_slots=3, n_gpio=6)
    cfg.update(host=1, flags=0xF, conf_sh=9, i2c_addr=0x2A, reject_thr=-(1 << 31), margin=0, model_id=MID, teacher_id=TID)
    blob = M.build_blob(cfg)
import chip_golden as CG
ev = CG.run_stream(G, blob, ref, raw, wakes)
want = [e["frame"].hex() for e in ev if e["to_host"]]
segs = [e for e in ev if "q" in e]
seg_all = [e for e in ev if e["ev"] == M.EV_WAKE]                    # every segment opens with a WAKE
want_q = [[int(v) for v in e["q"]] for e in segs]
want_v = [[int(v) for v in e["raw_values"]] for e in segs]

img = a.img
if a.neg == "rom":                        # one weight bit flipped: byte 0x9000 of the image (first encoder weights)
    lines = open(a.img).read().split("\n")
    i = 0x9000
    lines[i] = "%02x" % (int(lines[i], 16) ^ 0x10)
    img = str(out / "flash_neg.hex"); open(img, "w").write("\n".join(lines))
if not segs:                             # nothing for the core to run: the check would pass vacuously
    print(json.dumps({"neg": a.neg, "error": "the golden model found no segment in this stimulus"}))
    sys.exit(1)
if a.neg == "mic":                       # 4,000 samples of mic bits flipped inside the first segment the core runs
    s0, e0 = segs[0]["segment"]
    i = (s0 + (e0 - s0) // 3) * 192
    bits[i:i + 192 * 4000] ^= 1
if a.neg == "crc":
    b = bytearray(blob); b[-1] ^= 0x5A; blob = bytes(b)
if a.neg == "model":                     # K1: a blob for another model (the chip's id + 1) must be refused, error 7
    blob = stamp(blob, MID % 255 + 1, TID)
bits.tofile(out / "mic.pdm")
open(out / "cfg.blob", "wb").write(blob)

# ---- run the chip (the image path is a simulator parameter: the simulator reads it at start)
cmd = [a.sim, "+pdm=%s" % (out / "mic.pdm"), "+blob=%s" % (out / "cfg.blob"), "+wakes=%s" % (out / "wakes.txt"),
       "+out=%s" % (out / "chip.log"), "+pcmlog=%s" % (out / "fe.pcm"), "+rawlog=%s" % (out / "raw.txt"), "+tail=0.5", "+trace=%d" % a.trace,
       "+img=%s" % img]
p = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
log = open(out / "chip.log").read().split("\n")
got = ["".join(l.split()[3:]) for l in log if l.startswith("F ")]
qs, vs, cq, cv = [], [], [0] * 256, [0] * 8
for l in log:
    if l.startswith("W "):
        _, _, _, ad, da = l.split()
        ad, da = int(ad, 16), int(da, 16)
        if 0x8100 <= ad < 0x8200:
            for j in range(4):
                cq[ad - 0x8100 + j] = (((da >> (8 * j)) & 0xFF) ^ 0x80) - 0x80
        elif 0x8040 <= ad < 0x8060:
            cv[(ad - 0x8040) // 4] = da - (1 << 32) if da >> 31 else da
        elif ad == 0x8008 and da & 1:
            qs.append(cq[:D]); vs.append(cv[:])
fe = np.fromfile(out / "fe.pcm", dtype="<i2").astype(np.int64)
n = min(len(fe), len(ref))
# Frames are compared as two streams (docs/config-blob.md, pairing rule): WAKE frames, and decision frames
# (COMMAND, NOMATCH, TIMEOUT: one per WAKE, in segment order). The golden model dates a decision at its segment's
# end; the chip sends COMMAND / NOMATCH after the core's work, so the next segment's WAKE may come first. Each stream
# must be identical, in order, and pairing first-in first-out must hold: never more decisions than WAKEs so far,
# none outstanding at the end, as many decisions as segments (checked when every event goes to the host).
isl = lambda fs: [f for f in fs if int(f[2:4], 16) == 1]
dec = lambda fs: [f for f in fs if int(f[2:4], 16) in (0, 2, 3)]


def pairing(fs):
    out = 0
    for f in fs:
        out += 1 if int(f[2:4], 16) == 1 else -1
        if out < 0:
            return False
    return out == 0
res = dict(neg=a.neg, golden=a.golden, segments=[e["segment"] for e in segs], events=len(ev),
           frames_identical=(isl(got) == isl(want) and dec(got) == dec(want) and len(got) == len(want)),
           frame_order_identical=(got == want), pairing_ok=None, q_identical=(qs == want_q), values_identical=(vs == want_v),
           n_frames=len(want), n_segments_run=len(segs),
           frontend_identical=bool(np.array_equal(fe[:n], ref[:n])) and n >= len(ref) - 2,
           status=[l for l in log if l.startswith("S ")], frames_expected=want, frames_chip=got,
           sim_stderr=p.stderr[-600:])
_, pcz = M.parse_blob(open(a.blob, "rb").read()) if a.blob else M.parse_blob(open(out / "cfg.blob", "rb").read())
if pcz is not None:                      # GPIO: the settled pin state after every event, from the model
    gp = M.Gpio()
    for e in ev:
        if e["ev"] == M.EV_COMMAND:
            r = M.match(pcz, [int(v) for v in e["q"]], [int(v) for v in e["raw_values"]])
            gp.apply(pcz, e["action"], r["slot_mask"], r["vals8"])
        else:
            gp.apply(pcz, M.EV_ID[e["ev"]])
        gp.settle()
    glines = [l.split() for l in log if l.startswith("G ")]
    res["gpio_expected"] = gp.pins
    res["gpio_chip"] = int(glines[-1][3], 16) if glines else 0
    res["gpio_identical"] = res["gpio_expected"] == res["gpio_chip"]
else:
    res["gpio_identical"] = True
irq = [int(l.split()[2]) for l in log if l.startswith("I ")]
res["irq_samples"] = irq
res["segment_ends"] = [e["segment"][1] for e in segs]
lat = []                                 # end of speech (end of the last raw = 1 frame before the close) to irq
for e in segs:
    end = e["segment"][1]
    fr = [f for f in range(min(len(raw), end // 256)) if raw[f] and 256 * (f + 1) <= end]
    later = [i for i in irq if i >= end]
    if fr and later:
        lat.append(round((later[0] - 256 * (fr[-1] + 1)) / 16.0, 1))
res["latency_ms"] = lat
_, pcf = M.parse_blob(open(out / "cfg.blob", "rb").read())
if pcf is not None and pcf["host"] and (pcf["flags"] & 0xE) == 0xE:     # WAKE, NOMATCH, TIMEOUT all reach the host
    res["pairing_ok"] = pairing(got) and len(dec(got)) == len(seg_all) == len(isl(got))
res["identical"] = (res["frames_identical"] and res["q_identical"] and res["values_identical"] and res["gpio_identical"]
                    and res["pairing_ok"] is not False)
if a.neg == "model":                     # refused for the right reason: STATUS cfg error 7, not valid
    st = [int(l.split()[1], 16) for l in log if l.startswith("S ")]
    res["refused_model"] = bool(st) and st[0] >> 4 == 7 and not st[0] & 1
    res["identical"] = res["identical"] or not res["refused_model"]
json.dump(res, open(out / "result.json", "w"), indent=1)
print(json.dumps({k: res.get(k) for k in ("neg", "segments", "events", "n_frames", "frames_identical", "q_identical",
                                          "values_identical", "gpio_identical", "pairing_ok", "frontend_identical", "status",
                                          "irq_samples", "segment_ends", "latency_ms")}))
sys.exit(0 if res["identical"] == (a.neg == "") else 1)
