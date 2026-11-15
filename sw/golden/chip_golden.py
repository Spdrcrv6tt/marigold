"""Marigold whole-chip golden model: stream in, events out (docs/golden-format.md, docs/architecture.md section 8).

The island rules (segmenter of 8.1, matching of 8.2, confidence of 8.3, frames and CRC-8 of 8.4, config checks of
8.6, events of 8.7) are the core agent's reference implementation, rtl/model/mg_model.py, used here unchanged so
the two can never drift. This file adds what the core computes: the intent encoder (IntIntent, sw/golden/intent_np.py)
on each closed, non-TIMEOUT segment, and the value head's raw values per config slot (sw/intent/values.py).

    run_stream(G, blob, pcm, vad_raw, wakes) -> list of dicts {sample, ev, action, values, conf, frame, to_host}
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rtl" / "model")); sys.path.insert(0, str(ROOT / "sw" / "golden"))
sys.path.insert(0, str(ROOT / "sw" / "intent")); sys.path.insert(0, str(ROOT / "sdk"))
import mg_model as MG  # noqa: E402
import intent_np as S  # noqa: E402
import values as VAL  # noqa: E402
import compile as C  # noqa: E402


def slot_specs(blob):
    """Slot table with the level-word bytes (golden-format section 6), from the blob."""
    b = C.parse(blob)
    return [dict(type=s["type"], min=s["min"], max=s["max"],
                 words={w: s[w] for w in ("low", "medium", "high") if s[w]}) for s in b["slots"]]


def core_segment(I, pcm_seg, slots):
    """What the core computes for one segment: int8 q [D] and the raw value of every config slot (8 entries)."""
    q, k = I.run(np.asarray(pcm_seg, np.int64))
    toks = VAL.ctc_greedy(k)
    raw = VAL.raw_values(toks, slots)
    return q, toks, raw + [0] * (8 - len(raw))


def run_stream(G, blob, pcm, vad_raw, wakes, trace=None):
    err, cfg = MG.parse_blob(blob)
    if err:
        return [{"error": err}]
    I = S.IntIntent(G, trace)
    if int(G["proj.q"].shape[0]) != cfg["D"]:
        raise ValueError(f"config D {cfg['D']} != model D {G['proj.q'].shape[0]}")
    slots = slot_specs(blob)
    seg = MG.Segmenter()
    wk = set(int(w) for w in wakes)
    for w in range(len(pcm) + 1):
        raw_prev = int(vad_raw[w // 256 - 1]) if (w > 0 and w % 256 == 0 and w // 256 - 1 < len(vad_raw)) else 0
        seg.step(w, w in wk, raw_prev)
    ev = [{"sample": s, "ev": e, "action": 0, "values": [], "conf": 0} for s, e in seg.events]
    for start, end, tmo in seg.segs:
        if tmo:
            continue
        q, toks, raw = core_segment(I, pcm[start:end], slots)
        r = MG.match(cfg, [int(x) for x in q], raw)
        ev.append({"sample": end, "ev": r["ev"], "action": r["action"], "values": r["values"], "conf": r["conf"],
                   "segment": (start, end), "q": q, "tokens": toks, "raw_values": raw, "best": r["sb"]})
    ev.sort(key=lambda e: (e["sample"], e["ev"] != MG.EV_WAKE))
    for e in ev:
        e["frame"] = MG.frame(e["ev"], e["action"], e["values"], e["conf"])
        e["to_host"] = MG.host_gets(cfg, e["ev"])
    return ev
