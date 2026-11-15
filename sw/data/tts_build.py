#!/usr/bin/env python
"""Marigold synthetic speech: voice split, job lists, synthesis, manifests.

  tts_build.py voices                        -> data/tts/voices.json (eval/train voice split)
  tts_build.py jobs                          -> data/tts/jobs_eval.jsonl, data/tts/jobs_train.jsonl
  tts_build.py synth JOBS ENGINE I N [cpu|cuda] -> wavs + data/tts/done/<jobs>.<engine>.<I>.jsonl
  tts_build.py merge                         -> data/tts/{eval,train}/manifest.jsonl, voice F0 in voices.json

Held-out rules (docs/decisions.md, decision 1): eval/text lines are spoken only by eval voices;
train text that matches any eval/text line is dropped; the eval and train voice sets are disjoint
(by person, so Cori Samuel's two models stay together). Licenses: docs/licenses.md.
Audio: 16 kHz mono 16-bit PCM, peak-normalized to -3 dBFS.
"""
import hashlib, json, os, random, re, sys, time
from pathlib import Path

R = Path(__file__).resolve().parents[2]
D = R / "data" / "tts"
PIPER = D / "piper" / "models"
SR = 16000
os.environ.setdefault("HF_HOME", str(R / "data" / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

KOKORO_EVAL = ["af_heart", "af_nicole", "af_kore", "af_sky", "am_michael", "am_puck", "am_eric",
               "am_onyx", "am_fenrir", "bf_emma", "bf_lily", "bm_george", "bm_lewis", "bm_daniel"]
KOKORO_TRAIN = ["af_bella", "af_aoede", "af_sarah", "af_alloy", "af_nova", "af_jessica", "af_river",
                "am_echo", "am_liam", "am_santa", "am_adam", "bf_isabella", "bf_alice", "bm_fable"]
# en_US-norman-medium is license-clean but dropped for quality: Whisper WER 50% on the eval phrases
# (it swallows word onsets: "fan on now" -> "the end one in the island") with any noise setting.
# Single-speaker Piper voices: (model, locale, gender, person, data source, data license, lineage, split)
PIPER_SINGLE = [
    ("en_US-ljspeech-high", "en-US", "F", "ljspeech-linda-johnson", "LJ Speech (keithito.com)", "public domain", "trained from scratch", "eval"),
    ("en_US-kristin-medium", "en-US", "F", "librivox-kristin", "LibriVox recordings", "public domain", "trained from scratch", "eval"),
    ("en_US-john-medium", "en-US", "M", "librivox-john", "LibriVox recordings", "public domain", "fine-tuned from kristin-medium", "eval"),
    ("en_US-joe-medium", "en-US", "M?", "ohf-joe", "OHF-Voice/voice-datasets", "CC0", "fine-tuned from rdh base (nl_BE, CC0)", "eval"),
    ("en_US-mike-medium", "en-US", "M?", "ohf-mike", "OHF-Voice/voice-datasets", "CC0", "fine-tuned from rdh base (nl_BE, CC0)", "eval"),
    ("en_GB-alba-medium", "en-GB", "F?", "cstr-alba", "Edinburgh DataShare 10283/3270", "CC BY 4.0", "fine-tuned from hifi_tts_92 base", "eval"),
    ("en_GB-jenny_dioco-medium", "en-GB", "F", "dioco-jenny", "Jenny (Dioco) dataset", "Jenny license: commercial use permitted, attribution 'Jenny (Dioco)' in interactive products", "fine-tuned from hifi_tts_92 base", "eval"),
    ("en_GB-cori-high", "en-GB", "F", "cori-samuel", "LibriVox recordings (Cori Samuel)", "public domain", "trained from scratch", "train"),
    ("en_US-hifi_tts_92-medium", "en-GB", "F", "cori-samuel", "Hi-Fi TTS reader 92 (NVIDIA, openslr.org/109)", "CC BY 4.0", "trained from scratch", "train"),
]


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", t.lower().replace("’", "'"))).strip()


def jdump(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def jload(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------- voices
def cmd_voices():
    vs = []
    for v in KOKORO_EVAL + KOKORO_TRAIN:
        vs.append(dict(id=f"kokoro/{v}", engine="kokoro", model="hexgrad/Kokoro-82M (kokoro-v1_0.pth)", speaker=v,
                       locale="en-US" if v[0] == "a" else "en-GB", gender=v[1].upper(), gender_source="voice name prefix",
                       person=f"kokoro-{v}", data_license="Kokoro-82M weights Apache 2.0; training audio per model card: permissive/public domain plus synthetic audio from closed TTS providers",
                       source="huggingface.co/hexgrad/Kokoro-82M", split="eval" if v in KOKORO_EVAL else "train"))
    for m, loc, g, person, src, lic, lin, split in PIPER_SINGLE:
        vs.append(dict(id=f"piper/{m}", engine="piper", model=m, speaker=None, locale=loc, gender=g.rstrip("?"),
                       gender_source="model card" if not g.endswith("?") else "estimate (name; checked by F0 at merge)",
                       person=person, data_license=f"{src}: {lic}; {lin}", source=f"huggingface.co/rhasspy/piper-voices ({m})", split=split))
    # LibriTTS (train-clean-360 readers; LibriSpeech trains the encoder, so these are train only)
    sex = {}
    for line in open(R / "data/librispeech/LibriSpeech/SPEAKERS.TXT", encoding="utf-8"):
        if not line.startswith(";") and "|" in line:
            p = [x.strip() for x in line.split("|")]
            sex[p[0]] = p[1]
    cfg = json.load(open(PIPER / "en_US-libritts-high.onnx.json", encoding="utf-8"))
    for spk, idx in sorted(cfg["speaker_id_map"].items(), key=lambda kv: kv[1]):
        vs.append(dict(id=f"piper/en_US-libritts-high#{spk}", engine="piper", model="en_US-libritts-high", speaker=idx,
                       locale="en-US", gender=sex.get(spk.lstrip("p"), "?"), gender_source="LibriSpeech SPEAKERS.TXT",
                       person=f"librispeech-{spk.lstrip('p')}", data_license="LibriTTS train-clean-360 (openslr.org/60): CC BY 4.0; trained from scratch",
                       source="huggingface.co/rhasspy/piper-voices (en_US-libritts-high)", split="train"))
    ev = {v["person"] for v in vs if v["split"] == "eval"}
    tr = {v["person"] for v in vs if v["split"] == "train"}
    assert not ev & tr, ev & tr
    out = dict(created=time.strftime("%Y-%m-%d"), rule="eval and train voices are disjoint by person; eval/text lines are spoken only by eval voices",
               counts={s: sum(v["split"] == s for v in vs) for s in ("eval", "train")}, voices=vs)
    json.dump(out, open(D / "voices.json", "w", encoding="utf-8"), indent=1)
    print(out["counts"], "eval persons", len(ev), "train persons", len(tr))


# ---------------------------------------------------------------- texts
def parse_eval(path, period):
    rows, block, i = [], None, 0
    for line in open(path, encoding="utf-8"):
        s = line.strip()
        if not s or (s.startswith("#") and not s.startswith("## ")):
            continue
        if s.startswith("## "):
            block, i = s[3:].split(), 0
            continue
        text, _, vals = s.partition("|")
        values = {}
        for kv in filter(None, vals.strip().split(";")):
            k, _, v = kv.partition("=")
            values[k.strip()] = int(v) if v.strip().lstrip("-").isdigit() else v.strip()
        rows.append(dict(block=block, idx=i, text=text.strip(), values=values, split="dev" if i % period == 0 else "test"))
        i += 1
    return rows


def eval_texts():
    out = []
    for r in parse_eval(R / "eval/text/paraphrases.txt", 4):
        out.append(dict(set="paraphrases", product=r["block"][0], action=" ".join(r["block"][1:]), category=None, **{k: r[k] for k in ("idx", "text", "values", "split")}))
    for r in parse_eval(R / "eval/text/noncommands.txt", 3):
        out.append(dict(set="noncommands", product=None, action=None, category=r["block"][0], **{k: r[k] for k in ("idx", "text", "values", "split")}))
    return out


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def vshort(v):
    return slug(v["id"].replace("/", "-").replace("#", "-"))


def cmd_jobs():
    voices = json.load(open(D / "voices.json", encoding="utf-8"))["voices"]
    ev = [v for v in voices if v["split"] == "eval"]
    tr = [v for v in voices if v["split"] == "train"]
    evk, evp = [v for v in ev if v["engine"] == "kokoro"], [v for v in ev if v["engine"] == "piper"]
    trk, trp = [v for v in tr if v["engine"] == "kokoro"], [v for v in tr if v["engine"] == "piper"]

    def job(src, v, rng, rate_lo, rate_hi, root, group, uid):
        ls = round(rng.uniform(rate_lo, rate_hi), 3)  # length scale (>1 slower)
        return dict(id=uid, text=src["text"], product=src["product"], action=src["action"], values=src.get("values") or {},
                    category=src.get("category"), set=src["set"], split=src["split"], voice=v["id"], engine=v["engine"],
                    model=v["model"], speaker=v["speaker"], locale=v["locale"], length_scale=ls,
                    noise_scale=round(rng.uniform(0.5, 0.8), 3), noise_w=round(rng.uniform(0.6, 1.0), 3),
                    out=f"{root}/wav/{group}/{uid}.wav")

    # Config example phrases (fan, microwave) are spoken by EVAL voices only, as split "typed" (target 1):
    # decision 2026-11-15, the audio encoder never trains on fan or microwave text, not even config examples.
    typed = []
    for p in sorted((R / "sdk/configs").glob("*.json")):
        c = json.load(open(p, encoding="utf-8"))
        for a in c["actions"]:
            for k, e in enumerate(a["examples"]):
                typed.append(dict(set=f"config:{p.stem}", product=p.stem, action=a["name"], category=None, text=e, idx=k, values={}, split="typed"))
    ej = []
    for t in eval_texts() + typed:
        rng = random.Random(hashlib.sha1(f"{t['set']}|{t['product']}|{t['action']}|{t['category']}|{t['idx']}".encode()).hexdigest())
        pick = [rng.choice(evk), rng.choice(evp)]
        pick.append(rng.choice([v for v in ev if v not in pick]))
        group = slug(f"{t['set']}-{t['product'] or t['category']}-{t['action'] or ''}")
        for v in pick:
            ej.append(job(t, v, rng, 0.9, 1.15, "eval", group, f"{group}-{t['idx']:03d}-{vshort(v)}"))
    assert all(j["voice"] in {v["id"] for v in ev} for j in ej)
    jdump(D / "jobs_eval.jsonl", ej)

    banned = {norm(t["text"]) for t in eval_texts() + typed}
    src, dropped, seen = [], [], set()
    # Training text: data/gate0b/train_text.json (32 training appliances), plus lines of a later
    # train_text_v3.json that v1 lacks (v1 clip ids stay stable). Never fan or microwave text.
    for name, setname in (("train_text.json", "gate0b"), ("train_text_v3.json", "gate0b-v3")):
        tt = R / "data/gate0b" / name
        if not tt.exists():
            continue
        d = json.load(open(tt, encoding="utf-8"))
        for prod, v in d.items():
            if prod.lower() in ("fan", "desk fan", "microwave", "countertop microwave"):
                continue
            items = [(act, None, lines) for act, lines in v.get("actions", {}).items()] + [(None, "none", v.get("none", []))]
            for act, cat, lines in items:
                for k, e in enumerate(lines):
                    key = (prod, act, cat, norm(e))
                    if key in seen:
                        continue
                    seen.add(key)
                    src.append(dict(set=setname, product=prod, action=act, category=cat, text=e, idx=k, split="train", nvoice=3))
    tj = []
    for t in src:
        if norm(t["text"]) in banned:
            dropped.append(dict(t, reason="matches an eval/text line"))
            continue
        if re.search(r"[^\x00-\x7f\u2019]", t["text"]):  # the LLM wrote some lines in Chinese
            dropped.append(dict(t, reason="not English (non-ASCII script)"))
            continue
        rng = random.Random(hashlib.sha1(f"{t['set']}|{t['product']}|{t['action']}|{t['category']}|{t['idx']}|{t['text']}".encode()).hexdigest())
        pick = [rng.choice(trk)]
        while len(pick) < t["nvoice"]:
            v = rng.choice(trk) if rng.random() < 0.25 else rng.choice(trp)
            if v not in pick:
                pick.append(v)
        group = slug(f"{t['set']}-{t['product']}-{t['action'] or t['category']}")
        for v in pick:
            tj.append(job(t, v, rng, 0.85, 1.25, "train", group, f"{group}-{t['idx']:03d}-{vshort(v)}"))
    assert not any(j["voice"] in {v["id"] for v in ev} for j in tj)
    jdump(D / "jobs_train.jsonl", tj)
    jdump(D / "train_dropped_eval_overlap.jsonl", dropped)
    from collections import Counter
    print(f"eval jobs {len(ej)} from {len(eval_texts())} lines; train jobs {len(tj)} from {len(src) - len(dropped)} lines; dropped {dict(Counter(d['reason'] for d in dropped))}")


# ---------------------------------------------------------------- synthesis
def to16k(x, sr):
    import numpy as np
    from fractions import Fraction
    from scipy.signal import resample_poly
    x = np.asarray(x, dtype=np.float64)
    if sr != SR:
        f = Fraction(SR, sr)
        x = resample_poly(x, f.numerator, f.denominator)
    pk = float(np.max(np.abs(x))) if x.size else 0.0
    if pk > 0:
        x = x * (10 ** (-3 / 20) / pk)
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


class PiperBank:
    def __init__(self, threads):
        import onnxruntime as ort
        from piper import PiperVoice
        self.ort, self.PV, self.threads, self.m = ort, PiperVoice, threads, {}

    def get(self, model):
        if model not in self.m:
            p = PIPER / f"{model}.onnx"
            v = self.PV.load(p)
            o = self.ort.SessionOptions()
            o.intra_op_num_threads, o.inter_op_num_threads = self.threads, 1
            v.session = self.ort.InferenceSession(str(p), sess_options=o, providers=["CPUExecutionProvider"])
            self.m[model] = v
        return self.m[model]

    def say(self, j):
        import numpy as np
        from piper import SynthesisConfig
        v = self.get(j["model"])
        cfg = SynthesisConfig(speaker_id=j["speaker"], length_scale=j["length_scale"], noise_scale=j["noise_scale"], noise_w_scale=j["noise_w"])
        a = np.concatenate([c.audio_int16_array for c in v.synthesize(j["text"], syn_config=cfg)]).astype(np.float64) / 32768
        return a, v.config.sample_rate


class KokoroBank:
    def __init__(self, device):
        import torch
        from kokoro import KPipeline
        if device == "cpu":
            torch.set_num_threads(int(os.environ.get("KOKORO_THREADS", "4")))
        self.a = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=device)
        self.b = KPipeline(lang_code="b", repo_id="hexgrad/Kokoro-82M", model=self.a.model)

    def say(self, j):
        import numpy as np
        p = self.a if j["speaker"][0] == "a" else self.b
        parts = [r.audio.cpu().numpy() for r in p(j["text"], voice=j["speaker"], speed=1.0 / j["length_scale"]) if r.audio is not None]
        return np.concatenate(parts), 24000


def cmd_synth(jobs, engine, i, n, device="cpu"):
    import soundfile as sf
    i, n = int(i), int(n)
    rows = [j for k, j in enumerate(x for x in jload(Path(jobs)) if x["engine"] == engine) if k % n == i]
    bank = PiperBank(int(os.environ.get("PIPER_THREADS", "3"))) if engine == "piper" else KokoroBank(device)
    done_path = D / "done" / f"{Path(jobs).stem}.{engine}.{i}.jsonl"
    done_path.parent.mkdir(parents=True, exist_ok=True)
    seen = {r["id"] for p in done_path.parent.glob(f"{Path(jobs).stem}.*.jsonl") for r in jload(p) if "dur_s" in r}  # any shard layout
    t0, k, fails = time.time(), 0, 0
    with open(done_path, "a", encoding="utf-8") as log:
        for j in rows:
            if j["id"] in seen:
                continue
            out = D / j["out"]
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                a, sr = bank.say(j)
                pcm = to16k(a, sr)
                sf.write(str(out), pcm, SR, subtype="PCM_16")
                log.write(json.dumps(dict(id=j["id"], dur_s=round(len(pcm) / SR, 3))) + "\n")
            except Exception as e:  # keep going; failures are listed in the log
                fails += 1
                log.write(json.dumps(dict(id=j["id"], error=repr(e)[:300])) + "\n")
            k += 1
            if k % 200 == 0:
                log.flush()
                print(f"{engine} {i}/{n}: {k}/{len(rows) - len(seen)} in {time.time() - t0:.0f}s, fails {fails}", flush=True)
    print(f"{engine} {i}/{n} done: {k} new, {fails} failed, {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- merge
def f0_median(x):
    import numpy as np
    x = x.astype(np.float64) / 32768
    fl, hop, f0s = 640, 160, []
    thr = 0.1 * np.max(np.abs(x)) if x.size else 0
    for s in range(0, len(x) - fl, hop):
        fr = x[s:s + fl] - x[s:s + fl].mean()
        if np.sqrt(np.mean(fr ** 2)) < thr * 0.3:
            continue
        ac = np.correlate(fr, fr, "full")[fl - 1:]
        lo, hi = SR // 400, SR // 60
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[0] > 0 and ac[k] / ac[0] > 0.5:
            f0s.append(SR / k)
    return float(np.median(f0s)) if f0s else None


def cmd_merge():
    import numpy as np
    import soundfile as sf
    from num2words import num2words
    vfile = D / "voices.json"
    V = json.load(open(vfile, encoding="utf-8"))
    vid = {v["id"]: v for v in V["voices"]}
    f0 = {}
    for name in ("eval", "train"):
        jp = D / f"jobs_{name}.jsonl"
        if not jp.exists():
            continue
        done = {}
        for p in (D / "done").glob(f"jobs_{name}.*.jsonl"):
            for r in jload(p):
                if "dur_s" in r:
                    done[r["id"]] = r["dur_s"]
        rows, missing, bad = [], 0, []
        for j in jload(jp):
            if j["id"] not in done:
                missing += 1
                continue
            # TTS failure (e.g. a 51 s clip for 40 characters); numbers count as spoken ("375" -> words)
            spoken = re.sub(r"\d+", lambda m: num2words(int(m.group())), j["text"])
            if done[j["id"]] > 1.5 + 0.15 * len(spoken):
                bad.append(dict(id=j["id"], text=j["text"], voice=j["voice"], duration_s=done[j["id"]], reason="duration > 1.5 s + 0.15 s per spoken character"))
                continue
            rows.append(dict(path=f"data/tts/{j['out']}", text=j["text"], product=j["product"], action=j["action"], values=j["values"],
                             split=j["split"], voice=j["voice"], engine=j["engine"], set=j["set"], category=j["category"],
                             locale=j["locale"], gender=vid[j["voice"]]["gender"], length_scale=j["length_scale"], duration_s=done[j["id"]]))
        old = D / name / "manifest.jsonl"   # keep Whisper QA fields from an earlier run (same clip, same path)
        qa = {r["path"]: r for r in jload(old) if "asr_wer" in r} if old.exists() else {}
        for r in rows:
            if r["path"] in qa:
                r["asr_hyp"], r["asr_wer"] = qa[r["path"]]["asr_hyp"], qa[r["path"]]["asr_wer"]
        jdump(D / name / "manifest.jsonl", rows)
        jdump(D / name / "rejected.jsonl", bad)
        hrs = sum(r["duration_s"] for r in rows) / 3600
        print(f"{name}: {len(rows)} utterances, {hrs:.2f} h, {missing} jobs missing, {len(bad)} rejected as duration outliers")
        if name == "eval":  # median F0 per eval voice over up to 20 of its utterances
            by = {}
            for r in rows:
                by.setdefault(r["voice"], []).append(r["path"])
            for v, ps in by.items():
                vals = [f0_median(sf.read(str(R / p), dtype="int16")[0]) for p in ps[:20]]
                vals = [x for x in vals if x]
                f0[v] = round(float(np.median(vals)), 1) if vals else None
    for v in V["voices"]:
        if v["id"] in f0:
            v["f0_median_hz"] = f0[v["id"]]
    json.dump(V, open(vfile, "w", encoding="utf-8"), indent=1)
    print("eval voice F0:", {k.split("/")[-1]: x for k, x in sorted(f0.items())})


if __name__ == "__main__":
    c, a = sys.argv[1], sys.argv[2:]
    dict(voices=cmd_voices, jobs=cmd_jobs, synth=cmd_synth, merge=cmd_merge)[c](*a)
