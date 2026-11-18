"""Real voices at scale: SLURP's smart-light commands, said by real people in real rooms, against the SAME sentences
spoken by the held-out TTS eval voices, through the chip model (weights/mg-v3.npz, integer golden).

    python sw/intent/slurp_eval.py [--voices 7] [--neg 150] [--blob BLOB] [--out JSON]

SLURP audio is CC BY-NC 4.0: EVALUATION ONLY (docs/licenses.md), never training. The product is eval/slurp/light.json,
a smart-light config whose typed examples were written before reading any SLURP sentence.
1. SLURP devel + test utterances with a light intent (on, off, dim, brighten); each has close-talk recordings
   (-headset) and far-field ones. --neg utterances of other SLURP scenarios are real non-commands.
2. TTS: each light sentence by --voices held-out eval voices (data/tts/slurp_light/, eval voices only, CPU).
3. The blob: the light config compiled with the real teacher (text prototypes, typed examples only, no expansion: the
   same prototypes for every condition), unless --blob is given.
4. Per condition (synthetic, close-talk, far-field): the closest action (reject rule off; the hearing alone), the
   median best score, and how well the best score separates right commands from the real non-commands (AUC).
"""
import argparse, json, os, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "data/hf")); os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(ROOT / "sdk")); sys.path.insert(0, str(ROOT / "sw/data"))
import numpy as np  # noqa: E402
import compile as C  # noqa: E402

LIGHT = {"iot_hue_lighton": "on", "hue_lighton": "on", "iot_hue_lightoff": "off", "hue_lightoff": "off",
         "iot_hue_lightdim": "dim", "hue_lightdim": "dim", "iot_hue_lightup": "brighten", "hue_lightup": "brighten"}
VOICES = ["kokoro/af_heart", "kokoro/am_michael", "kokoro/bf_emma", "kokoro/bm_george", "piper/en_US-joe-medium",
          "piper/en_US-kristin-medium", "piper/en_GB-alba-medium", "kokoro/af_nicole", "kokoro/am_puck",
          "piper/en_US-mike-medium", "kokoro/bf_lily", "piper/en_GB-jenny_dioco-medium", "kokoro/bm_lewis",
          "kokoro/af_kore", "kokoro/am_eric", "piper/en_US-john-medium", "kokoro/am_onyx", "kokoro/af_sky",
          "kokoro/am_fenrir", "kokoro/bm_daniel", "piper/en_US-ljspeech-high"]


def slurp_rows(neg_n, seed=1):
    cmd, neg = [], []
    for sp in ("devel", "test"):
        for r in (json.loads(l) for l in open(ROOT / f"data/slurp/{sp}.jsonl", encoding="utf8")):
            files = [ROOT / "data/slurp/slurp_real" / x["file"] for x in r["recordings"]]
            files = [f for f in files if f.exists()]
            if r["intent"] in LIGHT:
                cmd.append(dict(text=r["sentence"], action=LIGHT[r["intent"]], files=files))
            elif sp == "test" and r["scenario"] not in ("iot",):
                neg.append(dict(text=r["sentence"], action=None, files=files))
    random.Random(seed).shuffle(neg)
    return cmd, neg[:neg_n]


def tts_jobs(texts, n_voices):
    """Eval-voice jobs for the light sentences, voice fields copied from data/tts/jobs_eval.jsonl."""
    proto = {}
    for j in (json.loads(l) for l in open(ROOT / "data/tts/jobs_eval.jsonl", encoding="utf8")):
        proto.setdefault(j["voice"], j)
    jobs = []
    for k, t in enumerate(sorted(texts)):
        for v in VOICES[:n_voices]:
            p = proto[v]
            vid = v.replace("/", "-").replace("_", "-").lower()
            jobs.append(dict({x: p[x] for x in ("voice", "engine", "model", "speaker", "locale", "length_scale", "noise_scale",
                                                 "noise_w")}, id=f"slurp-light-{k:04d}-{vid}", text=t, set="slurp_light",
                             out=f"slurp_light/wav/slurp-light-{k:04d}-{vid}.wav"))
    return jobs


def auc(pos, neg):
    """P(score of a right command > score of a non-command), ties half."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    allv = np.concatenate([pos, neg]); order = allv.argsort(); ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", type=int, default=7); ap.add_argument("--neg", type=int, default=150)
    ap.add_argument("--blob", default=None); ap.add_argument("--out", default=str(ROOT / "data/slurp_eval/results.json"))
    a = ap.parse_args()
    cfg = json.load(open(ROOT / "eval/slurp/light.json", encoding="utf-8"))
    name2id = {x["name"]: x["id"] for x in cfg["actions"]}
    cmd, neg = slurp_rows(a.neg)
    print(f"SLURP light commands: {len(cmd)} utterances, {sum(len(r['files']) for r in cmd)} recordings; "
          f"non-commands: {len(neg)} utterances, {sum(len(r['files']) for r in neg)} recordings", flush=True)
    jobs = tts_jobs({r["text"] for r in cmd}, a.voices)
    jp = ROOT / "data/tts/jobs_slurp_light.jsonl"
    with open(jp, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(j) + "\n" for j in jobs)
    import tts_build as T
    for eng in ("kokoro", "piper"):
        if any(j["engine"] == eng for j in jobs):
            T.cmd_synth(str(jp), eng, 0, 1, "cpu")
    if a.blob:
        blob = Path(a.blob).read_bytes()
    else:
        teacher = ROOT / "data/models" / __import__("chip_models").TEACHER["v3"]
        blob, _ = C.compile_config(cfg, "light", C.STEmbedder(teacher), None, proj="teacher", model_tag="v3")
    b = C.parse(blob)
    syn = [(ROOT / "data/tts" / j["out"], name2id[next(r["action"] for r in cmd if r["text"] == j["text"])]) for j in jobs
           if (ROOT / "data/tts" / j["out"]).exists()]
    close = [(f, name2id[r["action"]]) for r in cmd for f in r["files"] if "-headset" in f.name]
    far = [(f, name2id[r["action"]]) for r in cmd for f in r["files"] if "-headset" not in f.name]
    ncl = [f for r in neg for f in r["files"] if "-headset" in f.name]
    nfar = [f for r in neg for f in r["files"] if "-headset" not in f.name]
    paths = [str(p) for p, _ in syn + close + far] + [str(p) for p in ncl + nfar]
    Q = dict(zip(paths, C.chip_q(C.golden_path("v3"), paths).astype(np.int64)))
    res = {"blob_N": int(b["N"]), "conds": {}}
    negs = {"close-talk": ncl, "far-field": nfar}
    for name, rows in (("synthetic (eval voices)", syn), ("close-talk", close), ("far-field", far)):
        Qc = np.stack([Q[str(p)] for p, _ in rows]); y = np.array([t for _, t in rows])
        pure = C.pure_action(b, Qc); best = C.scores(b, Qc)[0]
        nq = negs.get(name, ncl + nfar)
        nb = C.scores(b, np.stack([Q[str(p)] for p in nq]))[0] if nq else []
        d = dict(n=len(rows), closest_right=float((pure == y).mean()), median_best=float(np.median(best)),
                 median_best_right=float(np.median(best[pure == y])) if (pure == y).any() else None,
                 auc_vs_noncommands=auc(best[pure == y], nb), noncommand_median_best=float(np.median(nb)) if len(nb) else None,
                 per_action={nm: float((pure[y == i] == i).mean()) for nm, i in name2id.items() if (y == i).any()})
        res["conds"][name] = d
        print(f"{name:<24} n={d['n']:>4}  closest action right {d['closest_right']:.1%}  median best score {d['median_best']:>8.0f} "
              f"(non-commands {d['noncommand_median_best'] or 0:>8.0f})  AUC {d['auc_vs_noncommands'] or 0:.3f}  "
              + " ".join(f"{k} {v:.0%}" for k, v in d["per_action"].items()), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
