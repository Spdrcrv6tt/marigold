"""Test vectors for the RTL (docs/golden-format.md section 7).

  python sw/golden/intent_vectors.py --tag v3 [--blob data/sdk/microwave-v3.mgc]
The blob defaults to data/sdk/microwave-<tag>.mgc (sdk/compile.py sdk/configs/microwave.json --model-tag <tag> -o it)
and must be built for that model (K1: blob version 1 records the model and teacher; the chip refuses another).

Per utterance n: data/golden/vectors-mg-<tag>-<n>.npz with
  version, pcm (int16 segment), every trace of IntIntent (fe.cmvn, fe.asinh, fe.lin, fe.c1, fe.feat, enc<i>.out,
  enc.ln, pool.m, proj, q, ctc.k), tokens, raw_values (8, per config slot of the blob), blob (uint8), the island's
  match result (ev, action, values, conf) and the host frame (uint8), all F:<site>;
  stream.* : the same utterance inside a stream (0.5 s silence, 0.3 s of silence before the wake sample, the
  utterance, 1 s silence) with EXAMPLE VAD raw flags (energy rule below, not the RTL VAD) and the whole-chip golden
  events (rtl/model/mg_model.py rules): stream.pcm, stream.vad_raw, stream.wake, stream.segments, stream.events.
Utterances are training clips only (TTS train voices, LibriSpeech train).
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sw" / "golden"))
import intent_np as S  # noqa: E402
import chip_golden as CG  # noqa: E402

CLIPS = [("tts", "cmd_value"), ("tts", "cmd"), ("tts", "none"), ("libri", None)]


def pick():
    rows = [json.loads(l) for l in open(ROOT / "data/intent/index_train.jsonl", encoding="utf8")]
    out = []
    num = [r for r in rows if r["src"] == "tts" and r["kind"] == "cmd" and any(ch.isdigit() for ch in r["text"])]
    out.append(num[len(num) // 3])
    cmd = [r for r in rows if r["src"] == "tts" and r["kind"] == "cmd" and r.get("product") == "space heater"]
    out.append(cmd[len(cmd) // 2])
    non = [r for r in rows if r["src"] == "tts" and r["kind"] == "none"]
    out.append(non[len(non) // 2])
    lib = [r for r in rows if r["src"] == "libri" and "start" not in r and 2.0 < r["dur"] < 4.0]
    out.append(lib[len(lib) // 2])
    return out


def vad_example(pcm):
    """Example raw VAD per 256-sample frame: frame RMS above max(200, 0.1 x the loudest frame's RMS)."""
    n = len(pcm) // 256
    e = np.sqrt((pcm[: n * 256].astype(np.float64).reshape(n, 256) ** 2).mean(1))
    return (e > max(200.0, 0.1 * e.max())).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--blob", default="", help="default data/sdk/microwave-<tag>.mgc")
    a = ap.parse_args()
    gp = next((q for q in (ROOT / f"data/golden/mg-{a.tag}.npz", ROOT / f"weights/mg-{a.tag}.npz") if q.exists()),
              ROOT / f"data/golden/mg-{a.tag}.npz")              # weights/ holds the shipped exports in git
    G = {k: np.asarray(v) for k, v in np.load(gp).items()}
    bp = ROOT / (a.blob or f"data/sdk/microwave-{a.tag}.mgc")
    if not bp.exists():
        sys.exit(f"{bp} not found: python sdk/compile.py sdk/configs/microwave.json --model-tag {a.tag} -o {bp}")
    blob = bp.read_bytes()
    import chip_models as CM                    # on the path through chip_golden (sdk/)
    mid, tid = CM.ids(a.tag)
    err, cfg = CG.MG.parse_blob(blob, model_id=mid, teacher_id=tid)
    assert not err, f"blob error {err} (7: built for another model than {a.tag})"
    slots = CG.slot_specs(blob)
    for n, r in enumerate(pick()):
        x = sf.read(str(ROOT / r["path"]), dtype="int16")[0].astype(np.int64)
        tr = {}
        I = S.IntIntent(G, tr)
        q, toks, raw = CG.core_segment(I, x, slots)
        res = CG.MG.match(cfg, [int(v) for v in q], raw)
        frame = CG.MG.frame(res["ev"], res["action"], res["values"], res["conf"])
        sil = lambda s: np.zeros(int(s * 16000), np.int64)
        stream = np.concatenate([sil(0.5), sil(0.3), x, sil(1.0)])
        wake = int(0.8 * 16000)
        vad = vad_example(stream)
        ev = CG.run_stream(G, blob, stream, vad, [wake])
        out = {"version": np.int64(1), "pcm": x.astype(np.int16), "tokens": np.array(toks, np.int64),
               "raw_values": np.array(raw, np.int64), "blob": np.frombuffer(blob, np.uint8),
               "match.ev": np.int64(res["ev"]), "match.action": np.int64(res["action"]), "match.conf": np.int64(res["conf"]),
               "match.best": np.int64(res["sb"]), "match.values": np.array(res["values"], np.int64).reshape(-1, 2),
               "frame": np.frombuffer(frame, np.uint8), "text": np.array(r["text"]),
               "stream.pcm": stream.astype(np.int16), "stream.vad_raw": vad, "stream.wake": np.int64(wake),
               "stream.events": np.array([[e["sample"], e["ev"], e["action"], e["conf"]] for e in ev], np.int64),
               "stream.frames": np.array([bytes(e["frame"]).hex() for e in ev])}
        segs = [e["segment"] for e in ev if "segment" in e]
        out["stream.segments"] = np.array(segs, np.int64).reshape(-1, 2)
        for k, v in tr.items():
            out["tr." + k] = v
        for k, v in G.items():
            if k.startswith("F:"):
                out[k] = v
        p = ROOT / f"data/golden/vectors-mg-{a.tag}-{n}.npz"
        np.savez(p, **out)
        print(f"{p.name}: '{r['text'][:40]}' T={tr['enc.ln'].shape[0]} tokens={toks} raw={raw[:2]} ev={res['ev']} "
              f"action={res['action']} conf={res['conf']}; stream events {[(e['sample'], e['ev']) for e in ev]}", flush=True)


if __name__ == "__main__":
    main()
