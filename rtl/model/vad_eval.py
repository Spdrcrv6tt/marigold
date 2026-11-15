"""Miss / false-alarm figures for the VAD reference model on LibriSpeech test-clean plus noise.

Stream: 40 utterances (the first of each of the 40 test-clean speakers), each scaled to
-26 dBFS active level, separated by 1.0-2.5 s gaps. Noise is added at a given SNR measured
against the active speech level. Noise types are synthetic (white, pink, mains hum plus
brown rumble) or babble made of 8 other LibriSpeech utterances summed.

Truth (from the clean stream, 16 ms frames): speech when the clean frame RMS is above
-56 dBFS (30 dB under the active level). Frames within 96 ms before or 320 ms after a speech
frame are a guard band and are not scored (that is where the hangover is meant to be on).
  miss        = speech frames with vad off / speech frames
  false alarm = scored non-speech frames with vad on / scored non-speech frames
  noise-only  = fraction of frames on in a 5 minute stream of the noise alone

Run inside the toolchain env: python rtl/model/vad_eval.py   (writes reports/data/phase2-vad-eval.json)
"""
import glob
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vad_model import Vad

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
LS = os.path.join(ROOT, "data", "librispeech", "LibriSpeech", "test-clean")
FS, FR = 16000, 256


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.square(x)) + 1e-20))


def load_utts():
    import soundfile as sf
    spk = sorted(os.listdir(LS))
    first, second = [], []
    for s in spk:
        files = sorted(glob.glob(os.path.join(LS, s, "*", "*.flac")))
        first.append(files[0])
        second.append(files[len(files) // 2])
    rd = lambda f: sf.read(f, dtype="float64")[0]
    return [rd(f) for f in first], [rd(f) for f in second[:8]], first


def active_level(x):
    """RMS over frames within 30 dB of the loudest frame."""
    n = len(x) // FR
    fr = np.sqrt(np.mean(np.square(x[:n * FR].reshape(n, FR)), axis=1) + 1e-20)
    act = fr > fr.max() * 10 ** (-30 / 20)
    return np.sqrt(np.mean(np.square(fr[act])))


def build_stream(utts, rng):
    parts = [np.zeros(2 * FS)]
    for u in utts:
        u = u * (10 ** (-26 / 20) / active_level(u))
        parts.append(u)
        parts.append(np.zeros(int(rng.uniform(1.0, 2.5) * FS)))
    return np.concatenate(parts)


def truth_frames(clean):
    n = len(clean) // FR
    fr = np.sqrt(np.mean(np.square(clean[:n * FR].reshape(n, FR)), axis=1) + 1e-20)
    sp = fr > 10 ** (-56 / 20)
    guard = np.zeros(n, bool)
    idx = np.flatnonzero(sp)
    for i in idx:
        guard[max(i - 6, 0):i + 21] = True
    amb = (fr > 10 ** (-70 / 20)) & ~sp          # weak sounds (breaths, tails): not scored
    return sp, (guard | amb) & ~sp


def make_noise(kind, n, rng, babble_src=None):
    if kind == "white":
        x = rng.standard_normal(n)
    elif kind == "pink":
        w = np.fft.rfft(rng.standard_normal(n))
        f = np.fft.rfftfreq(n, 1 / FS)
        f[0] = f[1]
        x = np.fft.irfft(w / np.sqrt(f), n)
    elif kind == "hum":
        # HVAC / machine room stand-in: 60 Hz + harmonics and brown (1/f^2) rumble
        t = np.arange(n) / FS
        w = np.fft.rfft(rng.standard_normal(n))
        f = np.fft.rfftfreq(n, 1 / FS)
        f[0] = f[1]
        brown = np.fft.irfft(w / np.maximum(f, 20.0), n)
        brown /= np.std(brown)
        x = brown + 0.7 * np.sin(2 * np.pi * 60 * t) + 0.4 * np.sin(2 * np.pi * 120 * t) + 0.2 * np.sin(2 * np.pi * 180 * t)
    elif kind == "babble":
        x = np.zeros(n)
        for u in babble_src:
            u = u / active_level(u)
            reps = int(np.ceil(n / len(u))) + 1
            off = int(rng.integers(0, len(u)))
            x += np.tile(u, reps)[off:off + n]
    else:
        raise ValueError(kind)
    return x / np.std(x)


def to_pcm(x):
    return np.clip(np.round(x * 32768), -32768, 32767).astype(np.int16)


def feats(pcm):
    return Vad().features(pcm)


def score(ft, sp, guard, **kw):
    v = Vad(**kw).decide(*ft)["vad"].astype(bool)
    n = min(len(v), len(sp))
    v, sp, guard = v[:n], sp[:n], guard[:n]
    non = ~sp & ~guard
    return float((sp & ~v).sum() / sp.sum()), float((non & v).sum() / non.sum())


_CACHE = {}


def prepare():
    """Build every test stream once and reduce it to frame features (the slow part)."""
    if _CACHE:
        return _CACHE
    # the frame features are slow to compute (integer filters in pure Python): keep them on disk
    import pickle
    cdir = os.path.join(ROOT, "data", "rtl-cache")
    key = "vad_feats_hp%d_lp%d_pre%d.pkl" % (Vad().p["HP_SH"], Vad().p["LP_SH"], Vad().p["PREEMPH"])
    cpath = os.path.join(cdir, key)
    if os.path.exists(cpath) and not os.environ.get("VAD_NO_CACHE"):
        with open(cpath, "rb") as f:
            _CACHE.update(pickle.load(f))
        return _CACHE
    rng = np.random.default_rng(2026)
    utts, bab, names = load_utts()
    clean = build_stream(utts, rng)
    sp, guard = truth_frames(clean)
    floor = 10 ** (-75 / 20) * rng.standard_normal(len(clean))      # mic self-noise stand-in
    c = dict(sp=sp, guard=guard, stream_s=len(clean) / FS, cond={})
    c["cond"]["clean"] = (feats(to_pcm(clean + floor)), None)
    for kind in ("white", "pink", "hum", "babble"):
        nz = make_noise(kind, len(clean), rng, bab)
        nz5 = make_noise(kind, 300 * FS, rng, bab)
        for snr in (20, 10, 5, 0):
            g = 10 ** ((-26 - snr) / 20)
            c["cond"]["%s_%d" % (kind, snr)] = (feats(to_pcm(clean + g * nz)), feats(to_pcm(g * nz5)))
    _CACHE.update(c)
    os.makedirs(cdir, exist_ok=True)
    with open(cpath, "wb") as f:
        pickle.dump(c, f)
    return _CACHE


def evaluate(verbose=True, **kw):
    c = prepare()
    sp, guard = c["sp"], c["guard"]
    res = dict(stream_s=c["stream_s"], speech_frames=int(sp.sum()), scored_nonspeech_frames=int((~sp & ~guard).sum()),
               params=Vad(**kw).p, cond={})
    if verbose:
        print("stream %.0f s, %d speech frames, %d scored non-speech frames" % (res["stream_s"], sp.sum(), (~sp & ~guard).sum()))
        print("%-8s %5s  %7s %7s %11s" % ("noise", "SNR", "miss%", "FA%", "noise-only%"))
    for name, (ft, ft5) in c["cond"].items():
        miss, fa = score(ft, sp, guard, **kw)
        r = dict(miss=miss, fa=fa)
        if ft5 is not None:
            r["noise_only"] = float(Vad(**kw).decide(*ft5)["vad"].mean())
        res["cond"][name] = r
        if verbose:
            kind, _, snr = name.partition("_")
            print("%-8s %5s  %7.2f %7.2f %11s" % (kind, snr or "-", 100 * miss, 100 * fa,
                                                  "%.2f" % (100 * r["noise_only"]) if ft5 is not None else ""))
    return res


STRICT = dict(K_D=0)      # the first version of the detector, without the mean + deviation test

if __name__ == "__main__":
    print("default parameters")
    r = evaluate()
    print("without the deviation test %s" % STRICT)
    r2 = evaluate(**STRICT)
    dst = os.path.join(ROOT, "reports", "data")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "phase2-vad-eval.json"), "w", newline="\n") as f:
        json.dump(dict(default=r, without_deviation_test=r2), f, indent=1)
