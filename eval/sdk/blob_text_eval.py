#!/usr/bin/env python
"""Text-path check of compiled config blobs (evaluation side; the SDK compiler never reads eval/).

For each product blob in data/sdk/<product>.mgc, the eval/text paraphrases and non-commands are embedded with
the blob's own embedder, projection and q scale (from <blob>.json), quantized to int8 and matched with the exact
blob matching rule (sdk/compile.py match). This is the upper bound an audio encoder that reproduced the text
embedding perfectly would reach with this blob; compare with Gate 0's float bound (eval/gate0).

  python eval/sdk/blob_text_eval.py [fan microwave]   -> prints and writes reports/sdk-blob-text.json
"""
import json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sdk"))
sys.path.insert(0, str(ROOT / "sw/data"))
import compile as C                        # noqa: E402
from tts_build import parse_eval           # noqa: E402


def main(products):
    para = parse_eval(ROOT / "eval/text/paraphrases.txt", 4)
    nonc = parse_eval(ROOT / "eval/text/noncommands.txt", 3)
    out = {}
    emb_cache = {}
    for p in products:
        bd = Path(os.environ.get("BLOBDIR", ROOT / "data/sdk"))
        blob = (bd / f"{p}.mgc").read_bytes()
        rep = json.load(open(bd / f"{p}.mgc.json", encoding="utf-8"))
        b = C.parse(blob)
        cfg = json.load(open(ROOT / f"sdk/configs/{p}.json", encoding="utf-8"))
        ids = {a["name"]: a["id"] for a in cfg["actions"]}
        if rep["embedder"] not in emb_cache:
            emb_cache[rep["embedder"]] = C.StubEmbedder() if rep["embedder"] == "stub" else C.STEmbedder(rep["embedder"])
        emb = emb_cache[rep["embedder"]]
        rows = [r for r in para if r["block"][0] == p]
        texts = [r["text"] for r in rows] + [r["text"] for r in nonc]
        E = emb.encode(texts)
        P = C.projection(rep["projection"], E.shape[1], b["D"])
        E = C.unit(E @ P) if P is not None else C.unit(E)
        Q = np.clip(np.round(E * rep["q_scale"]), -127, 127).astype(np.int32)
        pred = [C.match(b, q) for q in Q]
        res = {"embedder": rep["embedder"], "blob_bytes": len(blob), "N": b["N"], "D": b["D"]}
        for split in ("dev", "test"):
            sel = [i for i, r in enumerate(rows) if r["split"] == split]
            res[f"paraphrase_correct_{split}"] = round(float(np.mean([pred[i] == ids[" ".join(rows[i]["block"][1:])] for i in sel])), 4)
            res[f"paraphrase_rejected_{split}"] = round(float(np.mean([pred[i] is None for i in sel])), 4)
            res[f"paraphrase_n_{split}"] = len(sel)
            off = len(rows)
            head = [off + i for i, r in enumerate(nonc) if r["split"] == split and r["block"][0] in ("chat", "near", "tv")]
            oth = [off + i for i, r in enumerate(nonc) if r["split"] == split and r["block"][0] == "other_device"]
            res[f"noncommand_rejected_{split}"] = round(float(np.mean([pred[i] is None for i in head])), 4)
            res[f"other_device_rejected_{split}"] = round(float(np.mean([pred[i] is None for i in oth])), 4)
        out[p] = res
        print(p, json.dumps(res))
    if "BLOBDIR" not in os.environ:
        json.dump(out, open(ROOT / "reports/sdk-blob-text.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["fan", "microwave"])
