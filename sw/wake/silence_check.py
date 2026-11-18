"""Review S8 check: does the wake spotter miss when the audio around (or between) the words is exact digital zeros?

    python sw/wake/silence_check.py [PEAK_DBFS ...]      (default -6 -10.5 -20 -30)

The 168 eval-voice TTS wake clips (data/wake/tts/hey-kitchen, hey-breeze; held-out voices) are built into streams as
sw/wake/evalset.py's "clean" condition (0.8 s before and after, scaled to the peak level), then a white noise floor of
0, 1, 3, 10, 30 or 60 LSB RMS is added (0 = exact digital silence; 3 is the eval set's own floor). Each stream goes
through the exact VAD and the frozen wake v2 golden (format 2, gated) at the SDK default wake_sens 13. Printed: wake
rate and median peak keyword score per floor, overall and for the 24 clips with the most exact zeros inside the
phrase (between words).

Result (Nov 17, 2026, docs/decisions.md): not reproduced. Exact zeros wake as often as any floor at every level
(e.g. "hey kitchen" at -10.5 dBFS: 83.3% with zeros, 81.0% at 3 LSB, 84.5% at 60 LSB), including the clips with
silence between words.
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

R = Path(__file__).resolve().parents[2]
for p in ("sw/wake", "sw/wake/golden", "rtl/model"):
    sys.path.insert(0, str(R / p))
import golden  # noqa: E402
import template as TPL  # noqa: E402
from vadflags import vad_of  # noqa: E402

SIGMAS = (0, 1, 3, 10, 30, 60)
SENS = 13


def inner_zero_frames(clip):
    """Share of all-zero 10 ms frames between the clip's first and last nonzero sample."""
    nz = np.nonzero(clip)[0]
    inner = clip[nz[0]:nz[-1] + 1]
    fr = inner[: len(inner) // 160 * 160].reshape(-1, 160)
    return float(np.mean(np.all(fr == 0, 1)))


def one(args):
    path, phrase, peak_db = args
    P = golden.load_params(str(R / "sw/wake/golden/wake_int_v2.npz"))
    tpl = TPL.build(phrase)[0]
    clip = sf.read(path, dtype="int16")[0].astype(np.int64)
    c = clip / 32768
    lead = int(0.8 * 16000)
    out = {}
    for s in SIGMAS:
        rng = np.random.default_rng(7)
        x = np.concatenate([np.zeros(lead), c, np.zeros(lead)])
        x *= 10 ** (peak_db / 20) / (np.max(np.abs(x)) + 1e-12)
        x = np.clip(np.round(x * 32768 + rng.standard_normal(len(x)) * s), -32768, 32767).astype(np.int64)
        r = golden.run2(x, vad_of(x), tpl, SENS, P)
        out[s] = (int(len(r["wake_k"]) > 0), int(r["best"].max()))
    return phrase, inner_zero_frames(clip), out


def main():
    levels = [float(v) for v in sys.argv[1:]] or [-6.0, -10.5, -20.0, -30.0]
    for peak in levels:
        jobs = [(str(f), ph, peak) for ph, d in (("hey kitchen", "hey-kitchen"), ("hey breeze", "hey-breeze"))
                for f in sorted((R / "data/wake/tts" / d).glob("*.wav"))]
        with Pool(12) as pool:
            res = pool.map(one, jobs)
        for ph in ("hey kitchen", "hey breeze"):
            rr = [(z, o) for p, z, o in res if p == ph]
            print(f"{ph} at {peak} dBFS peak, {len(rr)} clips: " + ", ".join(
                f"{s} LSB {np.mean([o[s][0] for _, o in rr]):.1%} (score {np.median([o[s][1] for _, o in rr]):.0f})" for s in SIGMAS))
        top = sorted(((z, o) for _, z, o in res), key=lambda t: -t[0])[:24]
        print(f"  24 clips with most silence between words ({np.mean([z for z, _ in top]):.1%} of frames): wake with "
              + ", ".join(f"{s} LSB {sum(o[s][0] for _, o in top)}/24" for s in SIGMAS), flush=True)


if __name__ == "__main__":
    main()
