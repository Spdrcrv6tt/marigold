"""Exactness tests of the Marigold integer golden model (docs/golden-format.md section 7).

  python sw/golden/intent_test.py --tag v1 [--n 6]

1. FAST (float64 BLAS) == int64 on every trace, for n segments.
2. Streaming: the encoder on every chunk-aligned prefix (1,280 samples per chunk) gives, for all frames that are
   final at that point (all but the last 4), exactly the whole-segment integers; so the chunked hardware schedule
   and the whole-segment golden agree frame for frame.
"""
import argparse, json, random, sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sw" / "golden"))
import intent_np as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    G = dict(np.load(ROOT / f"data/golden/mg-{a.tag}.npz"))
    rows = [json.loads(l) for l in open(ROOT / "data/tts/eval/manifest.jsonl", encoding="utf8")]
    rng = random.Random(1)
    clips = [sf.read(str(ROOT / r["path"]), dtype="int16")[0].astype(np.int64) for r in rng.sample(rows, a.n)]
    ok_fast = ok_stream = True
    for x in clips:
        tr1, tr2 = {}, {}
        S.IntIntent.FAST = True; q1, k1 = S.IntIntent(G, tr1).run(x)
        S.IntIntent.FAST = False; q2, k2 = S.IntIntent(G, tr2).run(x)
        S.IntIntent.FAST = True
        same = all(np.array_equal(tr1[k], tr2[k]) for k in tr1) and np.array_equal(q1, q2) and np.array_equal(k1, k2)
        ok_fast &= same
        I = S.IntIntent(G)
        e = I.encode(x)
        for n in range(1, len(x) // S.CHUNK + 1):
            ep = I.encode(x[: n * S.CHUNK])
            fin = ep.shape[0] - 4
            if fin > 0 and not np.array_equal(ep[:fin], e[:fin]):
                ok_stream = False
                print("stream mismatch at chunk", n, flush=True)
                break
        print(f"segment {len(x)} samples: FAST==int64 {same}; q[:8] {q1[:8].tolist()}", flush=True)
    print("ALL FAST==int64:", ok_fast, " streaming prefix check:", ok_stream, flush=True)
    sys.exit(0 if ok_fast and ok_stream else 1)


if __name__ == "__main__":
    main()
