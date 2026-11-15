#!/usr/bin/env python
"""Marigold noise pools -> data/noise/<type>/<split>_NN.wav (16 kHz mono 16-bit) + data/noise/index.jsonl

  noise_gen.py synth      SYNTHETIC fan and microwave noise (procedural, no recordings involved), labelled synthetic
  noise_gen.py speech     babble and TV-like noise built from LibriSpeech (CC BY 4.0); eval pool from test-clean/dev-clean
                          speakers, train pool from train-clean-100 speakers (disjoint speakers)
  noise_gen.py index      rebuild data/noise/index.jsonl from the files above plus clean real recordings
                          (Speech Commands _background_noise_, MUSAN files whose license passed sw/data/musan_licenses.py)

Eval and train pools use different seeds / sources, so the noisy eval set never shares noise with training.
These stand in for John's own fan and microwave recordings until they arrive.
"""
import json, sys
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, lfilter

R = Path(__file__).resolve().parents[2]
N = R / "data" / "noise"
SR = 16000
DUR = 30.0
POOL = {"eval": (6, 1000), "train": (24, 2000)}  # files per pool, seed base


def smooth(rng, n, rate_hz):
    """Slow random wander, unit std, band-limited to about rate_hz."""
    k = max(2, int(n * rate_hz / SR) + 2)
    pts = rng.standard_normal(k)
    return np.interp(np.linspace(0, k - 1, n), np.arange(k), pts)


def pink(rng, n):
    w = rng.standard_normal(n)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]  # Kellet/Smith 1/f filter
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    x = lfilter(b, a, w)
    return x / np.std(x)


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], "bandpass", fs=SR, output="sos"), x)


def lp(x, fc, order=2):
    return sosfilt(butter(order, fc, "lowpass", fs=SR, output="sos"), x)


def hp(x, fc, order=2):
    return sosfilt(butter(order, fc, "highpass", fs=SR, output="sos"), x)


def resonator(x, f, q):
    return bp(x, f * (1 - 0.5 / q), f * (1 + 0.5 / q), 2)


def tones(rng, n, f_inst, amps):
    """Sum of harmonics k*f_inst(t) with random phases; f_inst is an array (Hz)."""
    ph = 2 * np.pi * np.cumsum(f_inst) / SR
    out = np.zeros(n)
    for k, a in enumerate(amps, 1):
        if np.max(f_inst) * k < SR / 2 - 200:
            out += a * np.sin(k * ph + rng.uniform(0, 2 * np.pi))
    return out


def fan(seed):
    rng = np.random.default_rng(seed)
    n = int(DUR * SR)
    rpm = rng.uniform(800, 1700)          # desk / pedestal fan speeds
    blades = int(rng.integers(3, 8))
    f_rot = rpm / 60 * (1 + 0.008 * smooth(rng, n, 0.3))
    bpf = blades * f_rot
    # broadband airflow: pink noise, low-passed, chopped slightly at the blade-pass rate
    air = lp(hp(pink(rng, n), 40), rng.uniform(1200, 3500))
    chop = 1 + rng.uniform(0.05, 0.2) * np.sin(2 * np.pi * np.cumsum(bpf) / SR)
    air = air * chop * (1 + 0.05 * smooth(rng, n, 0.5))
    air /= np.std(air)
    # blade-pass tones and harmonics, small once-per-rev imbalance, motor hum at 120 Hz (60 Hz mains)
    bt = tones(rng, n, bpf, [1 / k ** 1.3 for k in range(1, 7)]) * 10 ** (rng.uniform(-16, -8) / 20)
    imb = tones(rng, n, f_rot, [1.0]) * 10 ** (rng.uniform(-30, -20) / 20)
    hum = tones(rng, n, np.full(n, 120.0), [1, 0.3, 0.1]) * 10 ** (rng.uniform(-30, -20) / 20)
    x = air + bt + imb + hum
    return x, dict(rpm=round(rpm), blades=blades, bpf_hz=round(float(np.mean(bpf)), 1))


def microwave(seed):
    rng = np.random.default_rng(seed)
    n = int(DUR * SR)
    t = np.arange(n) / SR
    # magnetron transformer: 60 Hz mains, strongest at 120 Hz, harmonics to ~2 kHz (buzzy)
    amps = [0.35] + [1.0 / (k / 2) ** rng.uniform(0.9, 1.4) if k % 2 == 0 else rng.uniform(0.05, 0.3) / k for k in range(2, 34)]
    hum = tones(rng, n, np.full(n, 60.0) * (1 + 0.0005 * smooth(rng, n, 0.2)), amps)
    hum = np.tanh(1.5 * hum / np.std(hum)) * 10 ** (rng.uniform(-4, 2) / 20)
    # power cycling: below full power the magnetron switches on/off over a ~20-30 s period
    power = int(rng.choice([10, 10, 8, 6, 5]))
    period = rng.uniform(20, 30)
    gate = ((t + rng.uniform(0, period)) % period) < period * power / 10
    gate = np.convolve(gate.astype(float), np.ones(800) / 800, "same")   # 50 ms ramps
    hum = hum * gate
    # cooling fan: broadband + blade-pass tones
    rpm = rng.uniform(2000, 3200)
    blades = int(rng.integers(5, 10))
    bpf = blades * rpm / 60 * (1 + 0.005 * smooth(rng, n, 0.3))
    air = lp(hp(pink(rng, n), 60), rng.uniform(2000, 4500))
    air /= np.std(air)
    ft = tones(rng, n, bpf, [1 / k ** 1.4 for k in range(1, 5)]) * 10 ** (rng.uniform(-14, -6) / 20)
    # turntable: synchronous motor (~5 rpm) with gear whine, slow wobble of the whole sound
    rot = rng.uniform(4.5, 6.5) / 60
    whine = tones(rng, n, np.full(n, rng.uniform(250, 900)), [1, 0.4]) * 10 ** (rng.uniform(-34, -24) / 20)
    wobble = 1 + rng.uniform(0.02, 0.08) * np.sin(2 * np.pi * rot * t + rng.uniform(0, 6.3))
    # cavity / panel resonances excited by the hum and the airflow
    res = sum(resonator(hum + 0.3 * air, rng.uniform(700, 3500), rng.uniform(6, 15)) for _ in range(3))
    x = (0.8 * hum + 0.6 * air + ft + whine + 0.5 * res) * wobble
    return x, dict(power_level=power, cycle_s=round(period, 1), fan_rpm=round(rpm), fan_blades=blades)


def write(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    x = hp(x, 20)
    x = x / (np.max(np.abs(x)) + 1e-9) * 10 ** (-6 / 20)
    sf.write(str(path), (x * 32767).astype(np.int16), SR, subtype="PCM_16")


def cmd_synth():
    meta = []
    for kind, fn in (("fan", fan), ("microwave", microwave)):
        for split, (count, base) in POOL.items():
            for i in range(count):
                seed = base + (0 if kind == "fan" else 500) + i
                x, p = fn(seed)
                path = N / f"synthetic_{kind}" / f"{split}_{i:02d}.wav"
                write(path, x)
                meta.append(dict(path=str(path.relative_to(R)).replace("\\", "/"), type=kind, split=split, synthetic=True, seed=seed,
                                 params=p, source="procedural (sw/data/noise_gen.py)", license="ours (generated)"))
    json.dump(meta, open(N / "synthetic_meta.json", "w"), indent=1)
    print(f"synthetic: {len(meta)} files")


def flac_streams(roots, rng, k):
    """Pick k distinct speakers from the LibriSpeech roots; return per-speaker file lists."""
    spk = sorted({p for r in roots for p in (R / "data/librispeech/LibriSpeech" / r).iterdir() if p.is_dir()})
    pick = rng.choice(len(spk), size=k, replace=False)
    return [sorted(spk[i].rglob("*.flac")) for i in pick], [spk[i].name for i in pick]


def stream(files, rng, n):
    out, pos = np.zeros(n), int(rng.uniform(0, 1.5) * SR)
    order = rng.permutation(len(files))
    for i in order:
        if pos >= n:
            break
        a, sr = sf.read(str(files[i]), dtype="float64")
        a = a / (np.sqrt(np.mean(a ** 2)) + 1e-9)
        m = min(len(a), n - pos)
        out[pos:pos + m] = a[:m]
        pos += m + int(rng.uniform(0.1, 0.6) * SR)
    return out


def cmd_speech():
    meta = []
    srcs = {"eval": (["test-clean"], ["dev-clean"]), "train": (["train-clean-100"], ["train-clean-100"])}
    n = int(DUR * SR)
    for split, (count, base) in POOL.items():
        bab_roots, tv_roots = srcs[split]
        for i in range(count):
            rng = np.random.default_rng(base + 7000 + i)
            k = int(rng.integers(4, 9))
            files, spk = flac_streams(bab_roots, rng, k)
            x = sum(stream(f, rng, n) for f in files)
            p = N / "babble" / f"{split}_{i:02d}.wav"
            write(p, x)
            meta.append(dict(path=str(p.relative_to(R)).replace("\\", "/"), type="babble", split=split, synthetic=False, talkers=k,
                             speakers=spk, source=f"LibriSpeech {'+'.join(bab_roots)} mixed", license="CC BY 4.0 (LibriSpeech)"))
            rng = np.random.default_rng(base + 8000 + i)
            files, spk = flac_streams(tv_roots, rng, 2)
            x = stream(files[0], rng, n) + 10 ** (rng.uniform(-12, -4) / 20) * stream(files[1], rng, n)
            x = np.tanh(2.0 * bp(x, 150, 6500, 2) / 3)          # TV loudspeaker band and broadcast-style compression
            x = x + 10 ** (-40 / 20) * pink(rng, n)              # room tone
            p = N / "tv" / f"{split}_{i:02d}.wav"
            write(p, x)
            meta.append(dict(path=str(p.relative_to(R)).replace("\\", "/"), type="tv", split=split, synthetic=False, speakers=spk,
                             source=f"LibriSpeech {'+'.join(tv_roots)}, band-limited and compressed (speech-only TV; no music bed)",
                             license="CC BY 4.0 (LibriSpeech)"))
    json.dump(meta, open(N / "speech_meta.json", "w"), indent=1)
    print(f"babble/tv: {len(meta)} files")


def cmd_index():
    rows = []
    for m in ("synthetic_meta.json", "speech_meta.json", "musan_clean.json"):
        p = N / m
        if p.exists():
            rows += json.load(open(p, encoding="utf-8"))
    bg = R / "data/speech_commands/_background_noise_"
    for i, p in enumerate(sorted(bg.glob("*.wav"))):
        rows.append(dict(path=str(p.relative_to(R)).replace("\\", "/"), type="household" if "noise" not in p.stem else "stationary",
                         split="train", synthetic="noise" in p.stem, source=f"Speech Commands v0.02 _background_noise_/{p.name}", license="CC BY 4.0"))
    with open(N / "index.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(Counter((r["type"], r["split"]) for r in rows))


if __name__ == "__main__":
    dict(synth=cmd_synth, speech=cmd_speech, index=cmd_index)[sys.argv[1]]()
