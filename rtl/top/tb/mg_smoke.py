"""Whole-chip smoke test (no model): LibriSpeech speech and noise through the PDM microphone pin into the simulated
chip running rtl/top/tb/fw_smoke.c, with the stand-in spotter's wakes at chosen samples. The expected event frames
come from rtl/model/mg_model.py (segment rule, matching, frames) and the bit-exact front-end and VAD models; the
chip's frames are read over I2C by the harness. Run in WSL after rtl/core/tools/env.sh:

    python3 rtl/top/tb/mg_smoke.py --sim <sim binary> --out <dir> [--neg crc|mic]
"""
import argparse, json, os, struct, subprocess, sys, glob, random
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "rtl" / "model")); sys.path.insert(0, str(ROOT / "rtl" / "tb")); sys.path.insert(0, str(ROOT / "sdk"))
import pdm_model, mg_model as M
from vad_model import Vad
from mg_util import rand_cfg
import chip_models as CM

ap = argparse.ArgumentParser()
ap.add_argument("--sim", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--neg", default="", help="negative control: crc (corrupt the blob CRC) or mic (flip PDM bits)")
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
rng = random.Random(a.seed); nrng = np.random.default_rng(a.seed)

# ---- stimulus: noise, utterance, noise, utterance, a long noise gap (a wake with no speech: TIMEOUT), noise
LS = ROOT / "data" / "librispeech" / "LibriSpeech" / "test-clean"
files = sorted(glob.glob(str(LS / "*" / "*" / "*.flac")))
utts = [f for f in files if 2.0 < sf.info(f).duration < 3.5][:2]
noise = lambda s: nrng.standard_normal(int(s * 16000)) * 0.004
parts, marks, pos = [], [], 0
for seg in (noise(0.6), "u", noise(1.2), "u", noise(5.5), noise(0.8)):
    if isinstance(seg, str):
        x, sr = sf.read(utts[len(marks)], dtype="float64")
        x = x / np.abs(x).max() * 0.5
        marks.append(pos); parts.append(x); pos += len(x)
    else:
        parts.append(seg); pos += len(seg)
x = np.concatenate(parts)
gap = marks[1] + int(sf.info(utts[1]).duration * 16000) + int(1.0 * 16000)
cache = out.parent / ("stim-%d.npz" % a.seed)          # the modulator is slow: share the stimulus between runs
if cache.exists():
    z = np.load(cache); bits, ref = z["bits"].copy(), z["ref"]
else:
    bits = pdm_model.sdm(np.clip(pdm_model.upsample_to_pdm(x, 192), -0.9, 0.9)).astype(np.uint8)
    ref = np.array(pdm_model.pdm_decimate(bits), dtype=np.int64)
    np.savez(cache, bits=bits, ref=ref)
raw = np.asarray(Vad().run(ref)["raw"])
wakes = [marks[0] + 6400, marks[1] + 6400, gap]
if a.neg == "mic":                       # corrupted microphone data after the stimulus is fixed
    i = (marks[0] + 12000) * 192
    bits[i:i + 192 * 4000] ^= 1
bits.tofile(out / "mic.pdm")
open(out / "wakes.txt", "w").write("\n".join(map(str, wakes)) + "\n")

# ---- expected: segments, then the smoke firmware's q and values, matching and frames
seg = M.Segmenter()
for w in range(len(ref)):
    seg.step(w, w in wakes, int(raw[w // 256 - 1]) if (w > 0 and w % 256 == 0) else 0)
assert not seg.on, "a segment is still open at the end of the stimulus"


def smoke_q(s, n):
    return [((((s >> (k & 15)) ^ (k * 37) ^ n) & 0xFF) ^ 0x80) - 0x80 for k in range(256)]


segq = []
for st, en, tmo in seg.segs:
    if not tmo:
        s = int(np.sum(ref[st:en])) & 0xFFFFFFFF
        segq.append((st, en, smoke_q(s, en - st), [(s & 0x3FF) - 512 + k for k in range(8)]))
cfg = rand_cfg(rng, D=128, N=10, n_actions=4, n_slots=3, n_gpio=6)
cfg.update(host=1, flags=0xF, conf_sh=9, i2c_addr=0x2A)
cfg["model_id"], cfg["teacher_id"] = CM.rom_ids()      # K1: the chip's model, as chip.mk builds it
blob = M.build_blob(cfg)
_, pc = M.parse_blob(blob)
sc = [max(range(pc["N"]), key=lambda i: sum(q[d] * pc["vec"][i][d] for d in range(128)) + pc["vt"][i][2]) for _, _, q, _ in segq]
best = [sum(q[d] * pc["vec"][b][d] for d in range(128)) + pc["vt"][b][2] for (_, _, q, _), b in zip(segq, sc)]
cfg["reject_thr"] = min(best) if len(best) > 1 and best[0] != best[1] else -(1 << 31)
cfg["margin"] = 0
blob = M.build_blob(cfg)
_, pc = M.parse_blob(blob)
exp = [(s, ev, M.frame(ev)) for s, ev in seg.events]
for st, en, q, v in segq:
    r = M.match(pc, q[:128], v)
    exp.append((en, r["ev"], M.frame(r["ev"], r["action"], r["values"], r["conf"])))
exp.sort(key=lambda e: (e[0], e[1] != M.EV_WAKE))
want = [f.hex() for s, ev, f in exp if M.host_gets(pc, ev)]
if a.neg == "crc":
    b = bytearray(blob); b[-1] ^= 0x5A; blob = bytes(b)
open(out / "cfg.blob", "wb").write(blob)

# ---- run the chip
cmd = [a.sim, "+pdm=%s" % (out / "mic.pdm"), "+blob=%s" % (out / "cfg.blob"), "+wakes=%s" % (out / "wakes.txt"),
       "+out=%s" % (out / "chip.log"), "+pcmlog=%s" % (out / "fe.pcm"), "+rawlog=%s" % (out / "raw.txt"), "+tail=0.5"]
p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
log = open(out / "chip.log").read().split("\n")
got = ["".join(l.split()[3:]) for l in log if l.startswith("F ")]
fe = np.fromfile(out / "fe.pcm", dtype="<i2").astype(np.int64)
n = min(len(fe), len(ref))
pcm_ok = bool(np.array_equal(fe[:n], ref[:n])) and n >= len(ref) - 2
rawc = open(out / "raw.txt").read().strip()
raw_ok = rawc[:len(ref) // 256 - 1] == "".join(str(int(r)) for r in raw[:len(ref) // 256 - 1])
res = dict(neg=a.neg, segments=[list(s) for s in seg.segs], events=len(exp), frames_expected=want, frames_chip=got,
           identical=(got == want), frontend_identical=pcm_ok, vad_raw_identical=raw_ok, status=[l for l in log if l.startswith("S ")],
           sim_stderr=p.stderr[-400:])
json.dump(res, open(out / "result.json", "w"), indent=1)
print(json.dumps({k: res[k] for k in ("neg", "segments", "events", "identical", "frontend_identical", "vad_raw_identical", "status")}))
print("expected", want)
print("chip    ", got)
sys.exit(0 if (got == want) == (a.neg == "") else 1)
