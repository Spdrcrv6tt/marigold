"""Wake-phrase evaluation audio, spoken by the 21 EVAL voices only (data/tts/voices.json, split=eval),
with the data agent's synthesis banks (sw/data/tts_build.py). Writes data/wake/tts/<slug>/*.wav and
data/wake/tts/manifest_<engine>.jsonl.   usage: tts_wake.py piper|kokoro
"""
import json, os, re, sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "sw/data"))
import tts_build as T          # noqa: E402

PHRASES = ["hey breeze", "hey kitchen", "hey marigold", "okay toaster", "hello lumen", "hey computer"]
CARRIER = {"hey breeze": "Hey Breeze, turn the fan up a little.", "hey kitchen": "Hey Kitchen, start for two minutes."}
OUT = R / "data/wake/tts"


def jobs(engine, split="eval"):
    voices = json.load(open(R / "data/tts/voices.json", encoding="utf-8"))["voices"]
    ev = [v for v in voices if v["split"] == split and v["engine"] == engine]
    if split == "train":                       # dev set for tuning: 14 Kokoro + 16 LibriTTS train voices
        import random
        ev = ev if engine == "kokoro" else random.Random(3).sample(ev, 16)
    out = []
    for v in ev:
        for p in PHRASES:
            for k, (text, ls) in enumerate([(p.title() + ".", 1.0), (p, 1.12), (p.title() + "!", 0.92)]):
                out.append(dict(phrase=p, text=text, kind="isolated", length_scale=ls, v=v, k=k))
        for p, text in CARRIER.items():
            out.append(dict(phrase=p, text=text, kind="carrier", length_scale=1.0, v=v, k=9))
    return out


def main(engine, split="eval"):
    bank = T.PiperBank(3) if engine == "piper" else T.KokoroBank("cpu")
    import soundfile as sf
    rows = []
    out_root = OUT if split == "eval" else OUT.parent / "tts_dev"
    for j in jobs(engine, split):
        v = j["v"]
        slug = re.sub(r"[^a-z]+", "-", j["phrase"])
        name = f"{j['kind']}-{v['id'].replace('/', '-')}-{j['k']}.wav"
        out = out_root / slug / name
        out.parent.mkdir(parents=True, exist_ok=True)
        job = dict(text=j["text"], model=v["model"], speaker=v["speaker"], length_scale=j["length_scale"],
                   noise_scale=0.667, noise_w=0.8)
        if engine == "piper":
            job["model"] = v["id"].split("/", 1)[1]
            job["model"] = v["model"] if split == "train" else job["model"]
            job["speaker"] = v.get("speaker") if isinstance(v.get("speaker"), int) else None
        a, sr = bank.say(job)
        pcm = T.to16k(a, sr)
        sf.write(str(out), pcm, 16000, subtype="PCM_16")
        rows.append(dict(path=str(out.relative_to(R)).replace("\\", "/"), phrase=j["phrase"], text=j["text"], kind=j["kind"],
                         voice=v["id"], length_scale=j["length_scale"], dur_s=round(len(pcm) / 16000, 3), split=split))
    with open(out_root / f"manifest_{engine}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(engine, len(rows), "clips")


if __name__ == "__main__":
    os.environ.setdefault("KOKORO_THREADS", "8")
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "eval")
