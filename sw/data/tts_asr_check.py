#!/usr/bin/env python
"""QA only: transcribe synthetic speech with Whisper (openai/whisper-base.en, MIT) and report word error
rate per voice, so a voice that garbles commands is caught before it skews evaluation. Whisper is a
checking tool here; nothing it produces is used for training.

  tts_asr_check.py MANIFEST OUT_JSON [cpu|cuda]
"""
import json, re, sys
from pathlib import Path
import numpy as np
import soundfile as sf

R = Path(__file__).resolve().parents[2]


def words(t):
    from num2words import num2words
    t = t.lower().replace("’", "'")
    t = re.sub(r"(\d+)%", r"\1 percent", t)
    t = re.sub(r"\d+", lambda m: " " + num2words(int(m.group())) + " ", t)
    t = re.sub(r"[^a-z' ]", " ", t.replace("-", " "))
    return t.split()


def wer(ref, hyp):
    d = np.arange(len(hyp) + 1)
    for i, r in enumerate(ref, 1):
        prev, d[0] = d.copy(), i
        for j, h in enumerate(hyp, 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r != h))
    return d[-1], len(ref)


def main(manifest, out, device="cpu"):
    import torch
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device=0 if device == "cuda" else -1,
                   torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    rows = [json.loads(l) for l in open(R / manifest, encoding="utf-8")]
    todo = [r for r in rows if r.get("duration_s", 0) <= 29.5]  # Whisper takes at most 30 s
    res, per = [], {}
    B = 16
    for s in range(0, len(todo), B):
        chunk = todo[s:s + B]
        audio = [{"raw": sf.read(str(R / r["path"]), dtype="float32")[0], "sampling_rate": 16000} for r in chunk]
        for r, h in zip(chunk, asr(audio, batch_size=B)):
            e, n = wer(words(r["text"]), words(h["text"]))
            res.append(dict(path=r["path"], voice=r["voice"], text=r["text"], hyp=h["text"].strip(), errors=int(e), words=n))
            v = per.setdefault(r["voice"], [0, 0, 0, 0])
            # errors capped at 100% per clip: Whisper hallucination loops add hundreds of insertions
            v[0] += min(e, n); v[1] += n; v[2] += 1; v[3] += e / max(n, 1) > 0.5
    tot_e, tot_n = sum(v[0] for v in per.values()), sum(v[1] for v in per.values())
    summary = dict(model="openai/whisper-base.en", wer_note="per-clip errors capped at the reference length", utterances=len(res), wer=round(tot_e / tot_n, 4),
                   utt_wer_over_50pct=int(sum(v[3] for v in per.values())),
                   per_voice={k: dict(wer=round(v[0] / v[1], 4), utts=v[2], over_50pct=int(v[3])) for k, v in sorted(per.items())})
    json.dump(dict(summary=summary, rows=res), open(R / out, "w", encoding="utf-8"), indent=1)
    with open(R / manifest, "w", encoding="utf-8") as f:   # QA fields for consumers that want to drop doubtful clips
        q = {x["path"]: x for x in res}
        for r in rows:
            x = q.get(r["path"])
            f.write(json.dumps(dict(r, asr_hyp=x and x["hyp"], asr_wer=x and round(x["errors"] / max(x["words"], 1), 3)), ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=0)[:3000])


if __name__ == "__main__":
    main(*sys.argv[1:])
