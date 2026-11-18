"""Score a real recording (John's reading sets) through the whole-chip golden chain, as the chip would hear it.

    python sw/intent/real_eval.py WAV LINES BLOB [BLOB ...] [--out OUT.json] [--hang 8] [-v]

WAV: 16 kHz mono 16-bit (decode a phone .m4a first, e.g. ffmpeg -i in.m4a -ac 1 -ar 16000 -c:a pcm_s16le out.wav).
LINES: a reading-set number of eval/real/reading_sets_v2.jsonl, or a .jsonl of lines {line, text, action, category}.
The PCM goes in directly (no PDM model): VAD (rtl/model/vad_model.py) -> wake v2, format 2 (sw/wake/golden) -> the
segmenter and matcher (rtl/model/mg_model.py via sw/golden/chip_golden.py) with IntIntent v3 (weights/mg-v3.npz). Each
closed segment is transcribed by local Whisper base.en (data/hf, offline) from 0.5 s before it opens, and matched to
the closest line of the set (difflib, after removing the wake phrase); a decision is scored against that line's label.
Segments matching no line (ratio < 0.5) count as unmatched. --hang overrides the segmenter's hangover (frames of 16 ms)
for what-if runs; the chip's is 8.

Real recordings are EVALUATION ONLY: they stay on this PC under data/real/ (git-ignored), never train or tune anything,
and never go in the public repo.
"""
import argparse, difflib, json, os, re, struct, sys, zlib
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
for p in ("rtl/model", "sw/golden", "sw/intent", "sdk", "sw/wake", "sw/wake/golden"):
    sys.path.insert(0, str(ROOT / p))
os.environ.setdefault("HF_HOME", str(ROOT / "data/hf")); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import mg_model as M  # noqa: E402
_SEG0 = M.Segmenter    # run() always subclasses the chip's own segmenter (it patches M.Segmenter per call)
from vad_model import Vad  # noqa: E402
import golden as WG  # noqa: E402
import compile as C  # noqa: E402


def load_lines(spec):
    if str(spec).isdigit():
        rows = [json.loads(l) for l in open(ROOT / "eval/real/reading_sets_v2.jsonl", encoding="utf8")]
        return [r for r in rows if r["set"] == int(spec)]
    return [json.loads(l) for l in open(spec, encoding="utf8")]


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower().replace("hey breeze", "").replace("hey kitchen", "")).strip()


def run(wav, lines, blobs, hang=8):
    x = sf.read(wav, dtype="int16")[0].astype(np.int64)
    if x.ndim != 1:
        raise SystemExit("WAV must be mono")

    class Seg(_SEG0):
        def __init__(self):
            super().__init__(hang=hang)
    import chip_golden as CG
    CG.MG.Segmenter = Seg                       # what-if hangover; the chip's default is 8
    G = dict(np.load(C.golden_path("v3")))
    B = {str(p): Path(p).read_bytes() for p in blobs}
    err, c0 = M.parse_blob(next(iter(B.values())))
    if err:
        raise SystemExit(f"blob refused, error {err}")
    v = Vad().run(x)
    raw, vflag = np.asarray(v["raw"]), np.asarray(v["vad"])
    wakes = WG.run2(x, vflag, bytes(c0["wake"]), c0["wake_sens"], WG.load_params(str(ROOT / "sw/wake/golden/wake_int_v2.npz")))["wake_k"]
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device="cpu")
    texts = {}
    res = {"wav": str(wav), "seconds": len(x) / 16000, "vad_on": float(vflag.mean()), "wakes": [int(w) for w in wakes],
           "lines": lines, "hang": hang, "blobs": {}}
    for name, blob in B.items():
        b = C.parse(blob)
        rows = []
        for e in CG.run_stream(G, blob, x, raw, wakes):
            if "q" not in e:
                continue
            s0, s1 = e["segment"]
            if (s0, s1) not in texts:
                texts[(s0, s1)] = asr(x[max(0, s0 - 8000): s1].astype(np.float32) / 32768)["text"].strip()
            t = texts[(s0, s1)]
            sc = [difflib.SequenceMatcher(None, norm(t), norm(l["text"])).ratio() for l in lines]
            k = int(np.argmax(sc))
            Q = np.array([e["q"]], np.int64)
            best, act, _ = C.scores(b, Q)
            rows.append(dict(t0=s0 / 16000, t1=s1 / 16000, asr=t, line=lines[k]["line"], match=round(sc[k], 2),
                             label=lines[k]["action"], category=lines[k]["category"], action=int(e["action"]),
                             values=[[int(u) for u in v_] if isinstance(v_, (tuple, list)) else int(v_) for v_ in e["values"]],
                             best=int(best[0]), best_is_reject=bool(act[0] == 0), pure=int(C.pure_action(b, Q)[0]),
                             thr=int(b["reject_thr"])))
        res["blobs"][name] = rows
    return res


def summary(res, cfg_actions, verbose=False):
    id2 = {a["id"]: a["name"] for a in cfg_actions}; id2[0] = "-"
    lines = res["lines"]
    out = []
    for name, rows in res["blobs"].items():
        segs = [x for x in rows if x["match"] >= 0.5]
        heard = {x["line"] for x in segs}
        cmd = [l for l in lines if l["category"] == "command"]; neg = [l for l in lines if l["category"] != "command"]
        n = Counter()
        for x in segs:
            if x["category"] == "command":
                n["cmd"] += 1; n["right"] += id2[x["action"]] == x["label"]; n["rejected"] += x["action"] == 0
                n["pure"] += id2[x["pure"]] == x["label"]
            else:
                n["neg"] += 1; n["neg_rejected"] += x["action"] == 0
        n["wrong"] = n["cmd"] - n["right"] - n["rejected"]
        out.append(f"== {name}  threshold {segs[0]['thr'] if segs else '?'}\n"
                   f"  wake heard: commands {sum(l['line'] in heard for l in cmd)}/{len(cmd)}, non-commands "
                   f"{sum(l['line'] in heard for l in neg)}/{len(neg)}; unmatched segments {len(rows) - len(segs)}\n"
                   f"  heard commands: right {n['right']}/{n['cmd']}, wrong {n['wrong']}, rejected {n['rejected']}; rejection off "
                   f"(closest action) right {n['pure']}/{n['cmd']}\n  heard non-commands: rejected {n['neg_rejected']}/{n['neg']}")
        if verbose:
            out += [f"   {x['t0']:6.1f} L{x['line']:>2} {x['category']:<12} {x['label']:<10} -> {id2[x['action']]:<10} "
                    f"closest {id2[x['pure']]:<10} best {x['best']:>7}{' R' if x['best_is_reject'] else '  '} m{x['match']:.2f} | "
                    f"{x['asr'][:60]}" for x in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav"); ap.add_argument("lines"); ap.add_argument("blobs", nargs="+")
    ap.add_argument("--out", default=None); ap.add_argument("--hang", type=int, default=8)
    ap.add_argument("--config", default=None, help="config for action names (default: from the first line's product)")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    lines = load_lines(a.lines)
    prod = a.config or str(ROOT / f"sdk/configs/{lines[0].get('product', 'fan')}.json")
    res = run(a.wav, lines, a.blobs, a.hang)
    print(f"{res['seconds']:.1f} s, VAD on {res['vad_on'] * 100:.0f}%, {len(res['wakes'])} wakes for {len(lines)} lines")
    print(summary(res, json.load(open(prod, encoding="utf-8"))["actions"], a.v))
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
