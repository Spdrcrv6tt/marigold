"""Wake evaluation sets as cached integer front-end features (uint8 codes; they do not depend on the model).

  evalset.py pos    -> data/wake/eval/pos_<cond>.npz   wake phrases by EVAL TTS voices, 4 conditions
  evalset.py fa     -> data/wake/eval/fa_<src>.npz     long negative streams for false wakes per 24 h

Positive conditions (labelled synthetic in every report):
  clean   speech peak -6 dBFS, 3 LSB noise floor
  noise10 10 dB SNR, eval-split noise (babble, tv, fan, microwave, background, music), peak -6 dBFS
  reverb  synthetic room (RT60 0.3-0.7 s, direct-to-reverberant 0-6 dB, i.e. about 1-3 m), peak -16 dBFS
  far     reverb + 10 dB SNR eval noise, peak -16 dBFS
Every clip sits in a stream: 0.8 s lead + clip + 0.8 s tail of the same noise.
"""
import json, math, random, sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fe
from lexicon import R

E = R / "data/wake/eval"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def feats(x):
    """int16 numpy -> (P, 64) uint8 via the exact integer FE (GPU, chunked)."""
    x = np.asarray(x, np.int64)
    P = (len(x) - 560) // 320 + 1
    out, step = [], 320 * 20000
    for p0 in range(0, P, 20000):
        p1 = min(P, p0 + 20000)
        seg = torch.from_numpy(x[p0 * 320: p1 * 320 + 240]).to(DEV)
        out.append(fe.features(seg, p1 - p0).to(torch.uint8).cpu().numpy())
    return np.concatenate(out)


def noise_pool(split="eval"):
    rows = [json.loads(l) for l in open(R / "data/noise/index.jsonl", encoding="utf-8")]
    pools = {}
    for r in rows:
        if r["split"] != split or r["type"] not in ("babble", "tv", "fan", "microwave", "background", "music"):
            continue
        x, sr = sf.read(str(R / r["path"]), dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        if sr == 16000 and len(x) > 16000 * 4:
            pools.setdefault(r["type"], []).append(x / (np.sqrt(np.mean(x ** 2)) + 1e-9))
    return pools


def rir(rng):
    L = 8000
    t = np.arange(L) / 16000.0
    rt60 = rng.uniform(0.3, 0.7)
    tail = rng.standard_normal(L) * np.exp(-6.9 * t / rt60)
    tail[:16] = 0
    drr = 10 ** (rng.uniform(0, 6) / 10)
    tail /= np.sqrt(np.sum(tail ** 2) * drr)
    tail[0] = 1.0
    return tail


def active_power(x):
    f = x[: len(x) // 320 * 320].reshape(-1, 320)
    p = np.mean(f ** 2, 1)
    keep = p > p.max() * 10 ** (-3.0)
    return float(np.mean(p[keep]))


def make_stream(clip, cond, rng, pools):
    c = clip.astype(np.float64) / 32768
    lead = int(0.8 * 16000)
    if cond in ("reverb", "far"):
        c = np.convolve(c, rir(rng))[: len(c) + 4000]
    x = np.concatenate([np.zeros(lead), c, np.zeros(lead)])
    peak = 10 ** ((-6 if cond in ("clean", "noise10") else -16) / 20)
    x *= peak / (np.max(np.abs(x)) + 1e-12)
    ntype = None
    if cond in ("noise10", "far"):
        ntype = sorted(pools)[rng.integers(len(pools))]
        src = pools[ntype][rng.integers(len(pools[ntype]))]
        off = rng.integers(0, len(src) - len(x))
        n = src[off: off + len(x)]
        x = x + n * math.sqrt(active_power(x[lead: lead + len(c)]) / 10)
    x = x * 32768 + rng.standard_normal(len(x)) * 3
    return np.clip(np.round(x), -32768, 32767).astype(np.int16), lead, lead + len(c), ntype


def cmd_pos(split="eval"):
    """split eval: EVAL voices + eval noise (test). split train: TRAIN voices + train noise (dev, for tuning)."""
    Eo = E if split == "eval" else R / "data/wake/eval_dev"
    Eo.mkdir(parents=True, exist_ok=True)
    man = []
    for eng in ("kokoro", "piper"):
        man += [json.loads(l) for l in open(R / f"data/wake/{'tts' if split == 'eval' else 'tts_dev'}/manifest_{eng}.jsonl", encoding="utf-8")]
    assert all(m["split"] == split for m in man)
    pools = noise_pool(split)
    for cond in ("clean", "noise10", "reverb", "far"):
        rng = np.random.default_rng(abs(hash(cond)) % 2**32 if False else {"clean": 1, "noise10": 2, "reverb": 3, "far": 4}[cond])
        U, meta = [], []
        for m in man:
            clip, sr = sf.read(str(R / m["path"]), dtype="int16")
            x, s0, s1, nt = make_stream(clip, cond, rng, pools)
            U.append(feats(x))
            meta.append(dict(m, cond=cond, s0=int(s0), s1=int(s1), noise=nt, n=len(x)))
        np.savez(Eo / f"pos_{cond}.npz", u=np.concatenate(U), off=np.cumsum([0] + [len(u) for u in U]))
        json.dump(meta, open(Eo / f"pos_{cond}.json", "w"), indent=0)
        print(cond, len(U), "streams", flush=True)


def fa_sources():
    """name -> list of streams, each a list of audio file paths played back to back."""
    L = R / "data/librispeech/LibriSpeech"
    X = R / "data/librispeech-extra/LibriSpeech"
    src = {}
    for name, d in (("libri_test_clean", L / "test-clean"), ("libri_test_other", X / "test-other"), ("libri_dev_clean", L / "dev-clean")):
        src[name] = [sorted(str(p) for p in ch.glob("*.flac")) for ch in sorted(d.glob("*/*"))]
    M = R / "data/musan/musan"
    for name, sub in (("musan_speech", "speech"),):
        src[name] = [[str(p)] for p in sorted((M / sub).rglob("*.wav"))]
    train_music = {str(R / json.loads(l)["path"]).replace("\\", "/").lower() for l in open(R / "data/noise/index.jsonl", encoding="utf-8")
                   if json.loads(l)["split"] == "train"}
    src["musan_music"] = [[str(p)] for p in sorted((M / "music").rglob("*.wav")) if str(p).replace("\\", "/").lower() not in train_music]
    return src


def load(p):
    x, sr = sf.read(p, dtype="int16")
    if x.ndim > 1:
        x = x[:, 0]
    assert sr == 16000, p
    return x


def cmd_fa(only=None):
    E.mkdir(parents=True, exist_ok=True)
    src = fa_sources()
    rng = np.random.default_rng(7)
    for name, streams in src.items():
        if only and name not in only:
            continue
        U, meta = [], []
        for files in streams:
            x = np.concatenate([load(p) for p in files])
            if len(x) < 16000:
                continue
            U.append(feats(x)); meta.append(dict(files=len(files), first=files[0], n=len(x)))
        np.savez(E / f"fa_{name}.npz", u=np.concatenate(U), off=np.cumsum([0] + [len(u) for u in U]))
        json.dump(meta, open(E / f"fa_{name}.json", "w"))
        print(name, len(U), "streams", sum(m["n"] for m in meta) / 16000 / 3600, "h", flush=True)
    if not only or "tv_mix" in only:
        # TV-like: MUSAN librivox speech with a held-out MUSAN music bed at 5..12 dB SNR, band-limited 150-6500 Hz
        from scipy.signal import butter, sosfilt
        sos = butter(4, [150, 6500], btype="band", fs=16000, output="sos")
        sp = [s[0] for s in src["musan_speech"] if "librivox" in s[0]]
        mu = [s[0] for s in src["musan_music"]]
        U, meta, tot = [], [], 0
        for k in range(len(sp)):
            if tot > 10 * 3600 * 16000:
                break
            s = load(sp[k]).astype(np.float64)
            m = load(mu[k % len(mu)]).astype(np.float64)
            m = np.resize(m, len(s))
            snr = rng.uniform(5, 12)
            m *= math.sqrt(active_power(s) / (np.mean(m ** 2) + 1e-9) / 10 ** (snr / 10))
            x = sosfilt(sos, s + m)
            x = np.tanh(x / 20000) * 20000                     # broadcast-style compression
            x = np.clip(np.round(x), -32768, 32767).astype(np.int16)
            U.append(feats(x)); meta.append(dict(speech=sp[k], music=mu[k % len(mu)], snr=snr, n=len(x))); tot += len(x)
        np.savez(E / "fa_tv_mix.npz", u=np.concatenate(U), off=np.cumsum([0] + [len(u) for u in U]))
        json.dump(meta, open(E / "fa_tv_mix.json", "w"))
        print("tv_mix", len(U), "streams", tot / 16000 / 3600, "h", flush=True)


def cmd_split():
    """fa_musan_speech -> usgov (dev) / librivox (test); fa_musan_music -> a (dev) / b (test), alternate streams."""
    for name, rule in (("musan_speech", lambda k, m: "usgov" if "us-gov" in m["first"].replace("\\", "/") else "librivox"),
                       ("musan_music", lambda k, m: "a" if k % 2 == 0 else "b")):
        d = np.load(E / f"fa_{name}.npz"); meta = json.load(open(E / f"fa_{name}.json")); off = d["off"]
        u_all = d["u"]                                   # load once (an NpzFile re-reads the array on every access)
        groups = {}
        for k, m in enumerate(meta):
            groups.setdefault(rule(k, m), []).append(k)
        for g, ks in groups.items():
            U = [u_all[off[k]:off[k + 1]] for k in ks]
            np.savez(E / f"fa_{name}_{g}.npz", u=np.concatenate(U), off=np.cumsum([0] + [len(u) for u in U]))
            json.dump([meta[k] for k in ks], open(E / f"fa_{name}_{g}.json", "w"))
            print(name, g, len(ks), "streams", sum(len(u) for u in U) * 0.02 / 3600, "h")


if __name__ == "__main__":
    if sys.argv[1] == "split":
        cmd_split()
    elif sys.argv[1] == "pos":
        cmd_pos(sys.argv[2] if len(sys.argv) > 2 else "eval")
    else:
        cmd_fa(sys.argv[2:] or None)
