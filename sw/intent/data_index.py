"""Training-audio index for the intent encoder (Marigold model agent).

Writes data/intent/index_train.jsonl, one row per clip: {"path", "text", "src", "dur", "kind"} with
  src  = "tts"   data/tts/train/manifest.jsonl and data/tts/train_v3/manifest.jsonl when present (train voices,
                 training products only; never fan or microwave)
         "libri" LibriSpeech train-clean-100 and train-other-500 utterances of at most --max-s seconds, plus one
                 1.5-5 s crop from --crop-frac of the longer ones (words estimated by position) (never dev or test)
         "sc"    Speech Commands words outside validation_list.txt and testing_list.txt
  kind = "cmd" (a command to some training product), "none" (non-command speech)
Every text passes the eval leak filter (sw/textfilter.py). CPU only.
"""
import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sw"))
from textfilter import EvalFilter  # noqa: E402

HELD = {"fan", "microwave"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-s", type=float, default=5.5)
    ap.add_argument("--crop-frac", type=float, default=0.4, help="share of longer LibriSpeech utterances that give one crop")
    a = ap.parse_args()
    flt = EvalFilter()
    out = []
    # TTS training commands and non-commands
    n_tts = 0
    mans = [ROOT / "data/tts/train/manifest.jsonl"] + [ROOT / f"data/tts/{d}/manifest.jsonl" for d in ("train_v3", "train_v4")
                                                        if (ROOT / f"data/tts/{d}/manifest.jsonl").exists()]   # train_v4: iteration 2
    for line in (l for m_ in mans for l in open(m_, encoding="utf8")):
        r = json.loads(line)
        if r.get("split") != "train" or r.get("product") in HELD or not flt.ok(r["text"]):
            continue
        if r["duration_s"] > a.max_s:
            continue
        kind = "none" if r.get("category") == "none" or r.get("action") in (None, "", "none") else "cmd"
        out.append({"path": r["path"], "text": r["text"], "src": "tts", "dur": r["duration_s"], "kind": kind,
                    "product": r.get("product"), "action": r.get("action")})
        n_tts += 1
    print("tts", n_tts, flush=True)
    # LibriSpeech training subsets (short utterances only)
    files = []
    for base in ("data/librispeech/LibriSpeech/train-clean-100", "data/librispeech-extra/LibriSpeech/train-other-500"):
        for tf in sorted((ROOT / base).rglob("*.trans.txt")):
            for line in open(tf, encoding="utf8"):
                uid, _, txt = line.strip().partition(" ")
                files.append((tf.parent / f"{uid}.flac", txt.lower()))
    print("libri candidates", len(files), flush=True)

    def dur(item):
        try:
            i = sf.info(str(item[0]))
            return i.frames / i.samplerate
        except Exception:
            return 1e9
    with ThreadPoolExecutor(16) as ex:
        durs = list(ex.map(dur, files, chunksize=256))
    n_l = n_c = 0
    import random
    rng = random.Random(0)
    for (p, t), d in zip(files, durs):
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        if d <= a.max_s:
            if flt.ok(t):
                out.append({"path": rel, "text": t, "src": "libri", "dur": round(d, 3), "kind": "none"})
                n_l += 1
        elif d < 1e8 and rng.random() < a.crop_frac:
            # one crop of a longer utterance; its words are estimated by position (uniform speaking rate), which is
            # good enough for non-command speech whose job is to land away from commands
            w = t.split(); L = rng.uniform(1.5, 5.0); s0 = rng.uniform(0.2, max(0.2, d - L - 0.2))
            i0 = int(len(w) * s0 / d); i1 = max(i0 + 1, int(round(len(w) * (s0 + L) / d)))
            txt = " ".join(w[i0:i1])
            if flt.ok(txt):
                out.append({"path": rel, "text": txt, "src": "libri", "dur": round(L, 3), "kind": "none",
                            "start": int(s0 * 16000), "stop": int((s0 + L) * 16000)})
                n_c += 1
    print("libri short", n_l, "crops", n_c, flush=True)
    # Speech Commands (training portion)
    sc = ROOT / "data/speech_commands"
    held = set()
    for f in ("validation_list.txt", "testing_list.txt"):
        held |= {l.strip() for l in open(sc / f) if l.strip()}
    n_s = 0
    for d in sorted(sc.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        for w in sorted(d.glob("*.wav")):
            rel = f"{d.name}/{w.name}"
            if rel in held:
                continue
            out.append({"path": str(w.relative_to(ROOT)).replace("\\", "/"), "text": d.name, "src": "sc", "dur": 1.0, "kind": "none"})
            n_s += 1
    print("speech commands", n_s, flush=True)
    (ROOT / "data/intent").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "data/intent/index_train.jsonl", "w", encoding="utf8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print("rows", len(out), "eval-identical texts dropped", flt.dropped, flush=True)


if __name__ == "__main__":
    main()
