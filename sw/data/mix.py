#!/usr/bin/env python
"""Mix speech with noise at a set SNR.

  mix.py MANIFEST NOISE_TYPE SNR_DB OUT_DIR [--split eval] [--pad 0.25]

MANIFEST: speech JSONL (data/tts/*/manifest.jsonl). NOISE_TYPE: a type in data/noise/index.jsonl
(fan, microwave, babble, tv, music, ...). Only noise files of the given pool split are used (eval for
eval speech). Writes OUT_DIR/wav/<name>.wav and OUT_DIR/manifest.jsonl (speech fields + noise fields).

SNR = 10 log10(P_speech / P_noise). P_speech is the mean power of the active speech frames (20 ms
frames within 30 dB of the loudest frame), P_noise the mean power of the noise segment actually used.
The noise runs pad seconds before and after the speech, as an appliance runs before you talk.
If the mix would clip, speech and noise are scaled together (SNR unchanged).
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import soundfile as sf

R = Path(__file__).resolve().parents[2]
SR = 16000


def active_power(x, frame=320, floor_db=30.0):
    n = len(x) // frame
    if n == 0:
        return float(np.mean(x ** 2))
    p = np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1)
    keep = p >= p.max() * 10 ** (-floor_db / 10)
    return float(np.mean(p[keep]))


def mix(speech, noise, snr_db, rng, pad=0.25):
    s = np.concatenate([np.zeros(int(pad * SR)), speech, np.zeros(int(pad * SR))])
    if len(noise) < len(s):
        noise = np.tile(noise, int(np.ceil(len(s) / len(noise))) + 1)
    off = int(rng.integers(0, len(noise) - len(s) + 1))
    nz = noise[off: off + len(s)]
    ps, pn = active_power(speech), float(np.mean(nz ** 2))
    g = np.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    y = s + g * nz
    pk = np.max(np.abs(y))
    if pk > 0.99:
        y *= 0.99 / pk
    measured = 10 * np.log10(active_power(speech) / np.mean((g * nz) ** 2))
    return y, off, float(measured)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest"); ap.add_argument("noise_type"); ap.add_argument("snr", type=float); ap.add_argument("out")
    ap.add_argument("--split", default="eval"); ap.add_argument("--pad", type=float, default=0.25)
    a = ap.parse_args()
    pool = [json.loads(l) for l in open(R / "data/noise/index.jsonl", encoding="utf-8")]
    pool = [p for p in pool if p["type"] == a.noise_type and p["split"] == a.split]
    assert pool, f"no {a.noise_type} noise in the {a.split} pool"
    cache = {}
    out = Path(a.out) if Path(a.out).is_absolute() else R / a.out
    (out / "wav").mkdir(parents=True, exist_ok=True)
    rows = []
    for line in open(Path(a.manifest) if Path(a.manifest).is_absolute() else R / a.manifest, encoding="utf-8"):
        r = json.loads(line)
        rng = np.random.default_rng(int(hashlib.sha1(f"{r['path']}|{a.noise_type}|{a.snr}".encode()).hexdigest()[:12], 16))
        nfile = pool[int(rng.integers(0, len(pool)))]["path"]
        if nfile not in cache:
            x, sr = sf.read(str(R / nfile), dtype="float64")
            assert sr == SR, nfile
            cache[nfile] = x if x.ndim == 1 else x.mean(axis=1)
        sp, sr = sf.read(str(R / r["path"]), dtype="float64")
        assert sr == SR
        y, off, snr_meas = mix(sp, cache[nfile], a.snr, rng, a.pad)
        name = f"{Path(r['path']).stem}__{a.noise_type}{a.snr:g}db.wav"
        p = out / "wav" / name
        sf.write(str(p), (y * 32767).astype(np.int16), SR, subtype="PCM_16")
        rows.append(dict(r, path=str(p.relative_to(R)).replace("\\", "/"), clean_path=r["path"], noise=a.noise_type, noise_file=nfile,
                         noise_offset_s=round(off / SR, 3), snr_db=a.snr, snr_measured_db=round(snr_meas, 2), pad_s=a.pad,
                         duration_s=round(len(y) / SR, 3)))
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} mixed with {a.noise_type} at {a.snr} dB from {len(pool)} {a.split} noise files -> {out}")


if __name__ == "__main__":
    main()
