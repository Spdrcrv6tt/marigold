"""Model size as built (target 8) and compute / response-time estimate (target 7) for an exported golden model.

  python sw/intent/budget.py --tag v1

ROM image (packed, as the core's mkimage would store it): every weight at its site's bit width; per output row a
15-bit multiplier M, a 6-bit shift and (where present) a 24-bit bias; norm gains 16 bits per channel plus a 5-bit
gain format and a 32-bit eps per norm; the LUT ROMs (asinh, SiLU, GELU: 513 x 16 bits; exp2: 256 x 17 bits) and
the per-site formats (8 bits each). Firmware is not counted here (the core agent's).
Response time (ESTIMATE until RTL timing exists): end-of-speech hang (8 VAD frames = 128 ms, architecture 8.1) +
the encoder work left at end of speech (the last chunk plus the 4 look-ahead frames) at the core's effective
1.37G MACs/s (16 lanes x 98 MHz x 0.87, architecture.md section 4) + head and value parse + matching (at most
10.7 ms, architecture.md) + the frame.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sw" / "golden"))
import intent_np as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    G = dict(np.load(ROOT / f"data/golden/mg-{a.tag}.npz"))
    meta = json.load(open(ROOT / f"data/golden/mg-{a.tag}.json"))
    bits = meta["weight_bits"]
    wbits = rows = 0
    per = {}
    for w, _, _ in S.mac_sites():
        q = G[w + ".q"]; b = bits[w]
        n = q.size * b + q.shape[0] * (15 + 6 + (24 if w + ".bi" in G else 0))
        per[w] = n; wbits += q.size * b; rows += n - q.size * b
    norms = sum(G[n + ".G"].size * 16 + 5 + 32 for n, _ in S.norm_sites())
    luts = 3 * 513 * 16 + 256 * 17 + 16
    fmts = 8 * sum(1 for k in G if k.startswith("F:"))
    total = wbits + rows + norms + luts + fmts
    D = G["proj.q"].shape[0]
    # MACs per encoder frame (50 per second)
    sz = lambda w: int(G[w + ".q"].size)
    fe = 4 * sz("fe.lin") + 2 * sz("fe.c1") + sz("fe.c2")           # 5 ms frames x4, c1 at 10 ms x2, c2 at 20 ms
    layer = sum(sz(f"enc0.{c}") for c in ("q", "k", "v", "o", "fc1", "fc2"))
    att = sum(S.NH * (p + f + 1) * S.HD * 2 for p, f in S.WIN)
    ctc = sz("ctc")
    per_frame = fe + S.NL * layer + att + ctc
    rate = 16 * 98e6 * 0.87
    tail_frames = 4 + 4
    tail_ms = tail_frames * per_frame / rate * 1e3
    head_ms = (D * S.W) / rate * 1e3 + 0.3           # projection + norm + parse + bridge writes (estimate)
    match_ms = 10.7
    est = 128 + tail_ms + head_ms + match_ms
    out = {"tag": a.tag, "D": int(D), "weights": int(sum(G[w + ".q"].size for w, _, _ in S.mac_sites())),
           "weight_bits": int(wbits), "row_params_bits": int(rows), "norm_bits": int(norms), "lut_bits": luts,
           "rom_bits_total": int(total), "rom_mbit": round(total / 1e6, 3),
           "macs_per_frame": int(per_frame), "macs_per_audio_second": int(per_frame * 50),
           "real_time_factor_core": round(rate / (per_frame * 50), 2),
           "response_ms_estimate": {"hang": 128, "encoder_tail": round(tail_ms, 1), "head_parse_bridge": round(head_ms, 1),
                                    "match_max": match_ms, "total": round(est, 1)}}
    print(json.dumps(out, indent=1))
    Path(ROOT / "reports/p0").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ROOT / f"reports/p0/budget-{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
