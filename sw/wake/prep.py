"""Training manifest for the wake phone model: LibriSpeech train-clean-100 + train-other-500 (CC BY 4.0)
and the TTS train set (train voices only). Utterances with a word outside CMUdict are skipped.
Writes data/wake/train.jsonl and data/wake/dev.jsonl (dev-clean, for monitoring only)."""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lexicon import utterance_phones, R

OUT = R / "data/wake"


def libri(split_dir):
    for tr in sorted(Path(split_dir).rglob("*.trans.txt")):
        for line in open(tr, encoding="utf-8"):
            uid, text = line.strip().split(" ", 1)
            yield str(tr.parent / f"{uid}.flac"), text, uid.split("-")[0]


def info(p):
    try:
        return sf.info(p).frames
    except Exception:
        return -1


def build(items, name):
    rows, skipped = [], 0
    for path, text, spk in items:
        ph = utterance_phones(text)
        if not ph:
            skipped += 1
            continue
        rows.append(dict(path=path, ph=ph, spk=spk))
    with Pool(16) as pool:
        ns = pool.map(info, [r["path"] for r in rows], chunksize=256)
    with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r, n in zip(rows, ns):
            if n > 0:
                r["n"] = n
                f.write(json.dumps(r) + "\n")
    print(name, len(rows), "kept,", skipped, "skipped (OOV),", sum(n for n in ns if n > 0) / 16000 / 3600, "h", flush=True)


if __name__ == "__main__":
    L = R / "data/librispeech/LibriSpeech"
    X = R / "data/librispeech-extra/LibriSpeech"
    tts = []
    for line in open(R / "data/tts/train/manifest.jsonl", encoding="utf-8"):
        j = json.loads(line)
        assert j["split"] == "train"
        tts.append((str(R / j["path"]), j["text"], j["voice"]))
    items = list(libri(L / "train-clean-100")) + list(libri(X / "train-other-500")) + tts
    build(items, "train")
    build(list(libri(L / "dev-clean")), "dev")
