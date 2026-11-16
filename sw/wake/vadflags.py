"""VAD flags (rtl/model/vad_model.py, exact, 256-sample frames) for every wake evaluation stream, in the same stream
order as evalset.py. Long streams are processed in chunks (the VAD state carries over), 12 worker processes.

  vadflags.py pos [eval|train]   -> data/wake/eval[_dev]/vad_pos_<cond>.npz
  vadflags.py fa                  -> data/wake/eval/vad_fa_<src>.npz (+ the splits used by kws_eval)
Each file: flags uint8 (vad_o per 256-sample frame, streams concatenated), off (stream offsets).
"""
import json, math, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "rtl/model"))
from lexicon import R

E = R / "data/wake/eval"
CH = 256 * 4096


def vad_of(x):
    from vad_model import Vad
    v = Vad()
    out = []
    for i in range(0, len(x) // 256 * 256, CH):
        e, z = v.features(x[i:i + CH])
        out.append(v.decide(e, z)["vad"].astype(np.uint8))
    return np.concatenate(out) if out else np.zeros(0, np.uint8)


def vad_file(files):
    import soundfile as sf
    xs = []
    for p in files:
        x, sr = sf.read(p, dtype="int16")
        xs.append(x[:, 0] if x.ndim > 1 else x)
    x = np.concatenate(xs)
    return vad_of(x) if len(x) >= 16000 else None


def save(path, flags):
    np.savez(path, flags=np.concatenate(flags), off=np.cumsum([0] + [len(f) for f in flags]))


def cmd_pos(split):
    import evalset as ES
    import soundfile as sf
    Eo = E if split == "eval" else R / "data/wake/eval_dev"
    man = []
    for eng in ("kokoro", "piper"):
        man += [json.loads(l) for l in open(R / f"data/wake/{'tts' if split == 'eval' else 'tts_dev'}/manifest_{eng}.jsonl", encoding="utf-8")]
    pools = ES.noise_pool(split)
    for cond in ("clean", "noise10", "reverb", "far"):
        rng = np.random.default_rng({"clean": 1, "noise10": 2, "reverb": 3, "far": 4}[cond])
        xs = []
        for m in man:
            clip, _ = sf.read(str(R / m["path"]), dtype="int16")
            x, s0, s1, nt = ES.make_stream(clip, cond, rng, pools)
            xs.append(x)
        meta = json.load(open(Eo / f"pos_{cond}.json"))
        assert all(len(x) == m["n"] for x, m in zip(xs, meta)), "stream lengths differ from the cached features"
        with Pool(12) as pool:
            flags = pool.map(vad_of, xs, chunksize=8)
        save(Eo / f"vad_pos_{cond}.npz", flags)
        print(split, cond, len(flags), "streams", flush=True)


def tv_streams(src):
    """The TV-like mixes of evalset.cmd_fa, regenerated identically (generator, not stored)."""
    import evalset as ES
    from scipy.signal import butter, sosfilt
    sos = butter(4, [150, 6500], btype="band", fs=16000, output="sos")
    sp = [s[0] for s in src["musan_speech"] if "librivox" in s[0]]
    mu = [s[0] for s in src["musan_music"]]
    rng = np.random.default_rng(7)
    tot = 0
    for k in range(len(sp)):
        if tot > 10 * 3600 * 16000:
            break
        s = ES.load(sp[k]).astype(np.float64)
        m = ES.load(mu[k % len(mu)]).astype(np.float64)
        m = np.resize(m, len(s))
        snr = rng.uniform(5, 12)
        m *= math.sqrt(ES.active_power(s) / (np.mean(m ** 2) + 1e-9) / 10 ** (snr / 10))
        x = sosfilt(sos, s + m)
        x = np.tanh(x / 20000) * 20000
        x = np.clip(np.round(x), -32768, 32767).astype(np.int16)
        tot += len(x)
        yield x


def cmd_fa():
    import evalset as ES
    src = ES.fa_sources()
    for name in ("libri_test_clean", "libri_test_other", "libri_dev_clean", "musan_speech", "musan_music"):
        with Pool(12) as pool:
            res = pool.map(vad_file, src[name], chunksize=1)
        flags = [f for f in res if f is not None]
        meta = json.load(open(E / f"fa_{name}.json"))
        assert len(flags) == len(meta), (name, len(flags), len(meta))
        save(E / f"vad_fa_{name}.npz", flags)
        print(name, len(flags), "streams", flush=True)
        if name in ("musan_speech", "musan_music"):
            rule = (lambda k, m: "usgov" if "us-gov" in m["first"].replace("\\", "/") else "librivox") if name == "musan_speech" \
                else (lambda k, m: "a" if k % 2 == 0 else "b")
            groups = {}
            for k, m in enumerate(meta):
                groups.setdefault(rule(k, m), []).append(k)
            for g, ks in groups.items():
                save(E / f"vad_fa_{name}_{g}.npz", [flags[k] for k in ks])
    with Pool(12) as pool:
        flags = pool.map(vad_of, list(tv_streams(src)), chunksize=1)
    save(E / "vad_fa_tv_mix.npz", flags)
    print("tv_mix", len(flags), "streams", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "pos":
        cmd_pos(sys.argv[2] if len(sys.argv) > 2 else "eval")
    else:
        cmd_fa()
