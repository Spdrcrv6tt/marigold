# Marigold integer golden model: format and arithmetic

**DRAFT 0.1 (Nov 15, 2026, 5:50 PM), for the core agent to start firmware.** Owner: model agent (`sw/`). Consumer:
`rtl/`. Dimensions, op list and quantisation scheme below are fixed for v1 unless a change-log entry says otherwise;
the trained weights, the per-site formats (`F:<site>`) and the test vectors come with the v1 freeze (target 2 AM).
Where this text and the reference code (`sw/golden/intent_np.py`, from the freeze) disagree, the code is right.

The arithmetic is Poinsetta's (D:\poinsetta\docs\golden-format.md, format 3.6, sections 2 and 3): the same
`rshift_round`, `sat`, `mac`, `norm`, `lut` (asinh, SiLU, GELU ROMs), band attention with the exp2 softmax, and causal
`patches`. Marigold uses only the encoder half of that graph, shrunk, plus a pooled head and a CTC value head.

## 1. Model

| | Value |
|---|---|
| Width W | 240 (6 heads of HD = 40, so `sm.C` = 14949 as Poinsetta) |
| Encoder layers NL | 6, pre-norm, band attention + GELU MLP (FF = 480) |
| Frame rate | 80-sample frames (5 ms) in; 50 encoder frames per second (20 ms) |
| Band windows | layer 0: keys t-16 .. t+4; layers 1-5: keys t-16 .. t (no look-ahead) |
| Look-ahead | 4 frames = 80 ms = one chunk of 1,280 samples (architecture.md section 4) |
| Output D | **128** (provisional until about 7 PM tonight; 256 is the alternative). Same D in the config blob. |
| Value head | CTC over V = 45 tokens (section 6) |
| Weights | 3.43M at D = 256, 3.40M at D = 128 (matrices; norm gains and biases extra, about 8k) |

Weight sites (`.q` is [out, fan_in]; convolution columns are channel-major, column = in_channel * 5 + tap):

| Site | `.q` shape | Bits | Bias | Input format | Output |
|---|---|---|---|---|---|
| `fe.lin` | [240, 80] | 8 | no | `fe.asinh` | `fe.lin`, then SiLU |
| `fe.c1` | [240, 1200] | 8 | yes | `fe.lin` (after SiLU) | `fe.c1`, then SiLU |
| `fe.c2` | [240, 1200] | 8 | yes | `fe.c1` (after SiLU) | `enc.res` |
| `enc<i>.q/.k/.v` | [240, 240] | 4 | no | `enc<i>.ln1` | `enc<i>.q/.k/.v` |
| `enc<i>.o` | [240, 240] | 4 | no | `enc<i>.v` | `enc.res` |
| `enc<i>.fc1` | [480, 240] | 4 | yes | `enc<i>.ln2` | `enc<i>.fc1`, then GELU |
| `enc<i>.fc2` | [240, 480] | 4 | yes | `enc<i>.fc1` (after GELU) | `enc.res` |
| `proj` | [D, 240] | 4 | no | `pool` | `proj` |
| `ctc` | [45, 240] | 8 | yes | `enc.ln` | `ctc` (logits) |

Bits are the plan; QAT may move a site between 4 and 8 bits, and the freeze's `.json` lists the bits per site.
A b-bit site holds -(2^(b-1)-1) .. 2^(b-1)-1 (symmetric, per output row). Each site carries `.M` and `.sh` per output
row (2^14 <= M < 2^15, 1 <= sh < 62) and `.bi` (bias in accumulator units), exactly as Poinsetta.

Norm sites: `fe.cmvn` (per 80-sample frame, gain 1, eps 1e-6 at input format 15), `enc<i>.ln1`, `enc<i>.ln2`,
`enc.ln` (gains learned), `out` (gain 1, no learned gain). eps 1e-5 except `fe.cmvn`.

ROM image (estimate at D = 128): 2.79M x 4 bits + 0.61M x 8 bits = 16.0 Mbit of weights, plus multipliers, shifts,
biases, gains and the LUTs (`lut.asinh` with its own k, `lut.silu`, `lut.gelu`, `sm.exp2`): about 16.3 Mbit packed.
Compute: 3.77M MACs per encoder frame (frontend 0.94M, layers 2.77M, attention about 0.06M), 188M MACs per second
of audio, about 7x real time on the 16-lane array at 98 MHz.

## 2. Graph (one segment; everything starts from zero at the segment start)

    frames = segment PCM (int16, format 15), zero-padded to a multiple of 80, as rows of 80        n rows
    y   = norm(frames, fe.cmvn, 15)
    a   = lut(y, F[fe.cmvn], lut.asinh, R 4, no tails, f_out F[fe.asinh])
    h   = lut(mac(a, fe.lin), F[fe.lin], lut.silu, R 3, tails)                        [n, 240]
    c   = lut(mac(patches(h), fe.c1), F[fe.c1], lut.silu, R 3, tails)                 [ceil(n/2), 240]
    x   = mac(patches(c), fe.c2, sat 2^24 - 1)                                        [T = ceil(n/4), 240] at F[enc.res]

    encoder layer i (i = 0..5), window (16, 4) for i = 0 and (16, 0) otherwise:
        y = norm(x, enc<i>.ln1, F[enc.res]);  q, k, v = mac(y, enc<i>.q/.k/.v)   (6 heads of 40)
        h = sat(x + mac(band_attn(q, k, v, F[enc<i>.q], F[enc<i>.k]), enc<i>.o, 2^24 - 1), 2^24 - 1)
        y = lut(mac(norm(h, enc<i>.ln2, F[enc.res]), enc<i>.fc1), F[enc<i>.fc1], lut.gelu, R 3, tails)
        x = sat(h + mac(y, enc<i>.fc2, 2^24 - 1), 2^24 - 1)

    e = norm(x, enc.ln, F[enc.res])                                                   [T, 240] int16 at F[enc.ln]

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
golden model at the freeze): from start = max(k - pre, end of the previous segment) to the end of speech. The
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
3. parse(tokens, slots) gives one i32 per slot of the accepted action, in slot order; reference `sw/intent/values.py`
   (`parse`), integer only, ported line for line to firmware. Summary: runs of number words become integers
   ("seventy two" 72, "three fifty" 350 marked as a clock reading, "two five" 25, hundred and thousand as usual);
   n + second/minute/hour accumulate a duration in seconds ("two a half minute" 150, "half a minute" 30, "a minute
   a half" 90, "one minute thirty" 90); n + degree and n + percent are numbers; "level n" is a level. Slots: a
   duration slot takes the duration, else the first bare number (a clock reading m:ss gives 60 m + ss); a level slot
   takes "level n", else a level word (low, medium, high, max, min; "half" alone means medium), else a bare
   number; a number slot takes a degree or percent number, else a bare number. Each value is clamped to the slot's
   [min, max]; a slot with nothing heard gets 0, clamped.
4. Level words: max -> slot max, min -> slot min; low, medium, high -> the config's per-slot values when given,
   else min, (min + max + 1) / 2 and max. For a level slot, bytes 1 to 3 of the slot entry carry the low, medium and
   high values (0 = the default rule; coordinator ruling Nov 15), so microwave "high" can mean 10 and "low" 3.

## 7. Files and test vectors (at the freeze)

`data/golden/mg-<tag>.npz` (every array integer; keys as section 1 plus `F:<site>`, `version` = 1), `.json` (bits per
site, formats, calibration), `-float.npz` (dequantised weights). `data/golden/vectors-mg-<tag>-<n>.npz`: `pcm`,
`vad_raw`, `wake`, the segment table, and traces `fe.cmvn`, `fe.asinh`, `fe.lin`, `fe.c1`, `fe.feat`, `enc<i>.out`,
`enc.ln`, `pool.m`, `proj`, `q`, `ctc.k` (argmax per frame), `tokens`, plus a config blob and the expected event
frames. Bit-exact means: FAST = False (int64 only) equals FAST = True (float64 BLAS) on every trace, and the
streaming (chunked) encoder equals the whole-segment one.

## Change log

- 2026-11-15 draft 0.1: first draft for the core agent (dimensions, op list, D, quantisation, head and value rules),
  with the coordinator's rulings: pre-roll 0.3 s, conf_sh in the blob, level-word bytes in the slot entry.
