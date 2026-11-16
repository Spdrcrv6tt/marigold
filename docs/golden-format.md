# Marigold integer golden model: format and arithmetic

**Format v1.0, frozen Nov 15, 2026, 8:25 PM** (coordinator: freeze the format now, weights later). Owner: model agent
(`sw/`). Consumer: `rtl/`. The weights are not part of the format: a later export (new `mg-<tag>.npz`, same keys
and shapes) swaps in as a new ROM image and new test vectors. Any change to keys, shapes or arithmetic needs a new
format version and a change-log entry. Where this text and the reference code (`sw/golden/intent_np.py`,
`sw/intent/values.py`, `sw/golden/chip_golden.py`) disagree, the code is right.

The arithmetic is Poinsetta's (D:\poinsetta\docs\golden-format.md, format 3.6, sections 2 and 3): the same
`rshift_round`, `sat`, `mac`, `norm`, `lut` (asinh, SiLU, GELU ROMs), band attention with the exp2 softmax, and causal
`patches`. Marigold uses only the encoder half of that graph, shrunk, plus a pooled head and a CTC value head.

## 1. Model

**Draft 0.3 (Nov 15, 7 PM): the encoder is now Moonshine Streaming Tiny's own frontend plus its first two encoder
layers (MIT), fine-tuned, instead of a from-scratch W = 240 model.** The from-scratch model barely learned in 5k
steps (cosine 0.12 to the teacher); the pretrained start reaches that in 1k. Same op set and arithmetic as before.

| | Value |
|---|---|
| Width W | 320 (8 heads of HD = 40, `sm.C` = 14949, exactly Poinsetta's encoder) |
| Frontend | Poinsetta's: fe.lin [320, 80], fe.c1 [640, 1600] + SiLU, fe.c2 [320, 3200] |
| Encoder layers NL | 2, pre-norm, band attention + GELU MLP (FF = 1280) |
| Frame rate | 80-sample frames (5 ms) in; 50 encoder frames per second (20 ms) |
| Band windows | layer 0: keys t-16 .. t+4; layer 1: keys t-16 .. t (no look-ahead) |
| Look-ahead | 4 frames = 80 ms = one chunk of 1,280 samples (architecture.md section 4) |
| Output D | **128** (fixed at the teacher freeze) |
| Value head | CTC over V = 45 tokens (section 6) |
| Weights | 4,586,560 (matrices; norm gains and biases extra, about 5k) |

Weight sites (`.q` is [out, fan_in]; convolution columns are channel-major, column = in_channel * 5 + tap):

| Site | `.q` shape | Bits | Bias | Input format | Output |
|---|---|---|---|---|---|
| `fe.lin` | [320, 80] | 8 | no | `fe.asinh` | `fe.lin`, then SiLU |
| `fe.c1` | [640, 1600] | 4 | yes | `fe.lin` (after SiLU) | `fe.c1`, then SiLU |
| `fe.c2` | [320, 3200] | 4 | yes | `fe.c1` (after SiLU) | `enc.res` |
| `enc<i>.q/.k/.v` | [320, 320] | 4 | no | `enc<i>.ln1` | `enc<i>.q/.k/.v` |
| `enc<i>.o` | [320, 320] | 4 | no | `enc<i>.v` | `enc.res` |
| `enc<i>.fc1` | [1280, 320] | 4 | yes | `enc<i>.ln2` | `enc<i>.fc1`, then GELU |
| `enc<i>.fc2` | [320, 1280] | 4 | yes | `enc<i>.fc1` (after GELU) | `enc.res` |
| `proj` | [128, 320] | 4 | no | `pool` | `proj` |
| `ctc` | [45, 320] | 8 | yes | `enc.ln` | `ctc` (logits) |

A b-bit site holds -(2^(b-1)-1) .. 2^(b-1)-1 (symmetric, per output row). Each site carries `.M` and `.sh` per output
row (2^14 <= M < 2^15, 1 <= sh < 62) and `.bi` (bias in accumulator units), exactly as Poinsetta.

Norm sites: `fe.cmvn` (per 80-sample frame, gain 1, eps 1e-6 at input format 15), `enc<i>.ln1`, `enc<i>.ln2`,
`enc.ln` (gains learned), `out` (gain 1, no learned gain). eps 1e-5 except `fe.cmvn`.

ROM image (estimate): 4.55M x 4 bits + 0.04M x 8 bits = 18.5 Mbit of weights, plus per-row multipliers, shifts,
biases, gains and the LUTs: about 18.9 Mbit packed (under target 8's 20 Mbit). Compute: 5.6M MACs per encoder
frame (frontend 3.2M, layers 2.5M), 280M MACs per second of audio, about 4.9x real time on the 16-lane array at
98 MHz (architecture.md section 4 estimated 5.5x for 5M weights).

## 2. Graph (one segment; everything starts from zero at the segment start)

    frames = segment PCM (int16, format 15), zero-padded to a multiple of 80, as rows of 80        n rows
    y   = norm(frames, fe.cmvn, 15)
    a   = lut(y, F[fe.cmvn], lut.asinh, R 4, no tails, f_out F[fe.asinh])
    h   = lut(mac(a, fe.lin), F[fe.lin], lut.silu, R 3, tails)                        [n, 320]
    c   = lut(mac(patches(h), fe.c1), F[fe.c1], lut.silu, R 3, tails)                 [ceil(n/2), 640]
    x   = mac(patches(c), fe.c2, sat 2^24 - 1)                                        [T = ceil(n/4), 320] at F[enc.res]

    encoder layer i (i = 0, 1), window (16, 4) for i = 0 and (16, 0) for i = 1:
        y = norm(x, enc<i>.ln1, F[enc.res]);  q, k, v = mac(y, enc<i>.q/.k/.v)   (8 heads of 40)
        h = sat(x + mac(band_attn(q, k, v, F[enc<i>.q], F[enc<i>.k]), enc<i>.o, 2^24 - 1), 2^24 - 1)
        y = lut(mac(norm(h, enc<i>.ln2, F[enc.res]), enc<i>.fc1), F[enc<i>.fc1], lut.gelu, R 3, tails)
        x = sat(h + mac(y, enc<i>.fc2, 2^24 - 1), 2^24 - 1)

    e = norm(x, enc.ln, F[enc.res])                                                   [T, 320] int16 at F[enc.ln]

    intent head (at end of speech):
        s[c]  = sum over t < T of e[t][c]                                             int64
        m[c]  = sat(rshift_round(s[c], b), 32767),  b = bit length of (T - 1)  (0 when T = 1)   at F[enc.ln]
        p     = mac(m, proj)                                                          [D] at F[proj]
        q     = sat(norm(p, out, F[proj]) , 127)       norm output format F[out] = 5 (so |q| <= 127)   int8 [D]

    value head: logits[t] = mac(e[t], ctc) for t < T; section 6.

Band attention, softmax and patches are Poinsetta's (its section 3): query frame t sees the keys in its window that
exist in [0, T); the maximum is over those keys only.

Streaming: frames are released in chunks of 4 (1,280 samples). Frame t of layer 0 needs features up to t + 4, the
later layers need nothing later, so after a chunk arrives every frame up to (frames so far) - 4 is final, and the
last 4 frames are finished at end of speech with the keys that exist. The integers are identical to computing the
whole segment at once (the golden model does the latter; Poinsetta's EncoderState pattern proves the equivalence).
The sum s[c] can be accumulated frame by frame.

## 3. Segment

The encoder input is exactly the segment of architecture.md section 8.1 (the core agent's rule, copied into the
golden model at the freeze): from start = max(k - pre, end of the previous segment) to the end of speech (pre = 4800, mn = 8000 after the
Nov 15 rulings). The
golden model takes the stream, the VAD raw flags and the wake sample, cuts the segment by that rule, and runs the
graph above on it. TIMEOUT segments run no model.

Pre-roll (coordinator ruling, Nov 15): pre = 4800 samples (0.3 s) before the wake sample. The encoder is trained
with a random 0.1 to 0.5 s pre-roll in front of the command (wake-phrase-like word tails or silence).

## 4. Matching (docs/config-blob.md, architecture.md 8.2 and 8.3)

q is the int8 vector above; config vectors v_i are int8 in the same scale: unit RMS, zero mean over D, 5 fractional
bits. The SDK makes them from text (section 5). With |q|, |v| <= 127 and D <= 256 every score fits in int32.
Confidence (architecture.md 8.3): the shift conf_sh is read from the blob (byte 0x1C, u8; coordinator ruling). A
perfect match scores about D * 2^10, so the SDK writes conf_sh = 9 at D = 128 (10 at D = 256), mapping cosine 1.0 to
about 255.

## 5. Config vectors (SDK side; the chip never sees the text model)

    t = teacher(text), L2-normalised        (teacher = data/models/teacher-v3, a sentence-transformers model)
    v = P (t - mu)                          (P [D, E], mu [E] from data/models/teacher-v3/marigold_proj.npz)
    v = v - mean(v);  v = v / sqrt(mean(v^2))
    v_q = clip(round(v * 32), -127, 127)    int8

The encoder is trained so that q approximates v_q of what was said. Reject prototypes (action 0) are made the same way
from non-command text.

## 6. Value head

Vocabulary (id: token): 0 blank, 1-20 zero..nineteen, 21-28 twenty..ninety, 29 hundred, 30 thousand, 31 a,
32 half, 33 quarter, 34 second, 35 minute, 36 hour, 37 degree, 38 percent, 39 level, 40 low, 41 medium, 42 high,
43 max, 44 min.

1. k[t] = the lowest index with the largest logits[t] (t < T).
2. tokens = k with repeats collapsed, then blanks removed.
3. raw_values(tokens, slots) gives one i32 per CONFIG slot (in slot order, up to 8), each slot extracted on its
   own, independent of the action (the core computes VAL before the island picks the action); reference
   `sw/intent/values.py` (`items`, `raw_values`), integer only, ported line for line to firmware.
   - Number runs become integers: "seventy two" 72; "three fifty" 350 marked as a clock reading; "two five" 25;
     hundred and thousand as usual.
   - n + second/minute/hour accumulate a duration in seconds ("two a half minute" 150, "half a minute" 30,
     "a minute a half" 90, "one minute thirty" 90); n + degree and n + percent are numbers; "level n" is a level
     (training targets emit "level" for level/speed/setting/notch/power before a number, optionally with one of
     to/at/on/of/number in between).
   - duration slot: the duration, else a clock reading (m:ss -> 60 m + ss), else the first bare number;
   - level slot: "level n", else the first level word as a code (-1 low, -2 medium, -3 high, -4 max, -5 min;
     "half" alone is medium), else the first bare number;
   - number slot: the first degree or percent number, else the first bare number, else "level n";
   - nothing heard: 0.
4. The island (architecture.md 8.7, ruling of Nov 15) maps codes -1..-5 with the slot entry's low/medium/high
   bytes (0 = default: low = min, medium = floor((min + max) / 2), high = max; max -> max, min -> min), then clamps
   to [min, max] (`island_value` in values.py; rtl/model/mg_model.py must do the same).

## 7. Files and test vectors

| File | Made by | Content |
|---|---|---|
| `data/golden/mg-<tag>.npz` | `sw/golden/intent_export.py CKPT --tag <tag>` | The integer model: every key of section 1 (`<w>.q` int8, `.M`, `.sh`, `.bi`; `<n>.G`, `.fg`, `.eps`), `lut.asinh`, `lut.silu`, `lut.gelu` [513], `sm.exp2` [256], `sm.C`, `F:<site>`, `version` = 1 |
| `data/golden/mg-<tag>.json` | same | bits per site, weight count, weight ROM bits, formats, calibration ranges, `fe.k` |
| `data/golden/mg-<tag>-float.npz` | same | dequantised weights (float reference only) |
| `data/golden/vectors-mg-<tag>-<n>.npz` | `sw/golden/intent_vectors.py --tag <tag>` | per utterance: `pcm` (segment, int16), traces `tr.fe.cmvn`, `tr.fe.asinh`, `tr.fe.lin` (before SiLU), `tr.fe.c1` (before SiLU), `tr.fe.feat`, `tr.enc<i>.out`, `tr.enc.ln`, `tr.pool.m`, `tr.proj`, `tr.q`, `tr.ctc.k`; `tokens`, `raw_values` (8, the blob's slots), `blob`, the island's `match.*` and `frame`; `stream.*` = the utterance inside a stream with EXAMPLE VAD raw flags (an energy rule, not the RTL VAD), the wake sample, the segment table and the event frames from `chip_golden.run_stream` |

Bit-exact means: FAST = False (int64 only) equals FAST = True (float64 BLAS) on every trace, and the encoder on every
chunk-aligned prefix gives the whole-segment integers for every frame that is final (all but the last 4):
`sw/golden/intent_test.py --tag <tag>` checks both. The island rules (segmenter, match, confidence, CRC-8, events)
are rtl/model/mg_model.py's, used unchanged by `chip_golden.py`.

Interim export for the core agent (Nov 15, 8:25 PM): tag `qsmoke` (a short QAT run from a mid-training float
checkpoint; real format, placeholder weights). The final tag comes with the 2 AM freeze.

## Change log

- 2026-11-15 v1.0: format frozen (draft 0.3 dimensions, D = 128, value codes); test vectors and interim export `qsmoke`.

- 2026-11-15 draft 0.1: first draft for the core agent (dimensions, op list, D, quantisation, head and value rules),
  with the coordinator's rulings: pre-roll 0.3 s, conf_sh in the blob, level-word bytes in the slot entry.
- 2026-11-15 draft 0.3: encoder = Moonshine Streaming Tiny frontend + first 2 layers (W 320, FF 1280, C1 640),
  4.59M weights, frontend convs 4-bit to stay under 20 Mbit; windows (16,4), (16,0).
- 2026-11-15 draft 0.2: values are per config slot and action-independent; level words leave the core as codes
  -1..-5 and the island maps them (ruling); segment mn = 0.5 s.
