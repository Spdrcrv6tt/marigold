# Golden wake model, format 1 (wake agent, Nov 15-16, 2026)

**Frozen Nov 15, 2026, 7 PM.** Parameters: sw/wake/golden/wake_int.npz, identical word for word to the RTL ROM image rtl/wake/mem/wake_rom.mem (checked by sw/wake/golden/vectors.py); trained checkpoint data/wake/q1/best.pt. Results: reports/p1-wake.md.

The exact integer rules of Marigold's wake spotter. The reference is sw/wake/golden/golden.py with the front end in sw/wake/fe.py; rtl/wake/ must match it bit for bit (rtl/wake/tb). Changing anything here means a new format number, new vectors and a dated entry in docs/decisions.md.

## 1. What it is

An always-on, type-to-train wake-word spotter inside the island, on clk_fe (3.072 MHz):

1. **Front end (FE):** integer log-mel features, 32 bands, one 64-byte vector per 20 ms network frame.
2. **Phone network (NN):** a causal depthwise-separable 1D CNN, 58,464 four-bit weights in ROM, giving 40 logits per frame (39 CMUdict phones + CTC blank) in 1/16 nat.
3. **Keyword scorer (DP):** a CTC keyword search against the OEM's template (the phones of the typed wake phrase), with free start, a per-frame deficit cap, a sensitivity threshold, reset and refractory on a fire.

No retraining per product: the OEM's wake phrase is data (the template bytes and wake_sens in the config blob).

## 2. Timing and interface (docs/architecture.md section 3.1, agreed Nov 15)

- Sample n = 0 is the first `in_valid` with `en = 1` (the same sample as capture-ring index 0). While `en = 0` all spotter state clears.
- Network frame t covers samples [320t, 320t + 560): FE frame a = x[320t, 320t + 400), FE frame b = x[320t + 160, 320t + 560). Frame t is computed after sample 320t + 559 arrives.
- A fire for frame t raises `wake` for one clock together with the `in_valid` of sample **k = 320t + 879** (one frame later). The golden model reports only k inside the stream.
- `in_vad` and `in_raw` are unused in format 1: the spotter runs every frame while `en` (no VAD gating, no VAD qualifier).
- Config port: the spotter re-reads the template from config SRAM every frame, word by word (cfg_rdata = mem[a >> 2], valid on the clock after cfg_gnt; request and address held until the grant). `tpl_len` is not needed.
- Work per frame, measured in simulation: 40.3k of the 61,440 clk_fe clocks (45.8k with ROM and config-port stalls): FE about 22k, NN about 18k, DP under 1k. Each engine has its own clock gate (on from its start to one clock after done).

## 3. Front end (fe.py)

All integers; `>>` is an arithmetic shift (floor).

1. **Window.** w[i] = round(32767 sin²(π(i + 0.5)/400)), i = 0..399 (symmetric: the ROM keeps w[0..199]). y[i] = (x[i] w[i] + 16384) >> 15 for frames a and b.
2. **FFT.** re[i] = y_a[i], im[i] = y_b[i] for i < 400, 0 for 400 ≤ i < 512. Radix-2 decimation in time over 9 stages on bit-reversed input. Stage s (1..9), half h = 2^(s−1), butterfly j = 0..255: pos = j mod h, top = (j div h)·2h + pos, bot = top + h, twiddle index k = pos·2^(9−s). With S[i] = round(32767 sin(2πi/512)), i = 0..128: c = S[128 − k] if k ≤ 128 else −S[k − 128]; s = S[k] if k ≤ 128 else S[256 − k]. Then tr = (b_r c + b_i s + 16384) >> 15, ti = (b_i c − b_r s + 16384) >> 15; top ← a + t, bot ← a − t. No scaling; every value fits 28 bits signed.
3. **Split.** For bin k, with m = (512 − k) mod 512: A = (Zr[k] + Zr[m], Zi[k] − Zi[m]), B = (Zi[k] + Zi[m], Zr[m] − Zr[k]) (twice the spectra of frames a and b).
4. **Magnitude.** mag(r, i) = max(|r|, |i|) + ((3 min(|r|, |i|)) >> 3).
5. **Mel.** 32 triangular bands on the HTK mel scale between 60 and 7600 Hz (34 edge frequencies c_0..c_33 equally spaced in mel). For bins k = 2..243 (f = 31.25k Hz), i_k = the largest j with c_j ≤ f and wu_k = round(256 (f − c_i)/(c_{i+1} − c_i)). Bin k adds mag·wu_k to band i_k (if i_k < 32) and mag·(256 − wu_k) to band i_k − 1 (if i_k ≥ 1). The table (i_k, wu_k) is rtl/wake/mem/wake_fe_mel.mem; i_k rises by at most 1 per bin, so the RTL streams the bands with two accumulators per frame.
6. **Log code.** For band energy E (< 2^48): L = 0 if E = 0, else p = position of the leading one, f = the 4 bits below it (shifted up when p < 4), L = 16p + LUT[f] with LUT = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15] (round(16 log2(1 + f/16))).
7. **Feature.** v = clamp((L >> 1) − 96, 0, 255): one step is 1/8 octave of magnitude (0.75 dB). The NN input byte order is [frame a bands 0..31, frame b bands 0..31], and the network sees x = v − 128 (int8).

## 4. Phone network (golden.py `net`)

Twelve layers, causal in time (history before frame 0 of a session is zero):

| # | Layer | Weights |
|---|---|---|
| 0 | pointwise 64 → 96 | 6,144 |
| 1, 3, 5, 7, 9 | depthwise, kernel 5, dilation 1, 1, 2, 2, 2 (96 channels) | 480 each |
| 2, 4, 6, 8, 10 | pointwise 96 → 96 | 9,216 each |
| 11 | pointwise 96 → 40 (output) | 3,840 |

Receptive field 33 frames (660 ms). Total 58,464 weights.

Per layer: acc[o] = b[o] + Σ W[o, i]·x[i] (pointwise) or b[c] + Σ_k W[c, k]·h[c, t − (4 − k)d] (depthwise, tap k = 0 oldest), with W in [−7, 7] (4-bit), b int32, x uint8 (int8 for layer 0). Then y = (acc·M[o] + 2^(S−1)) >> S with M < 2^15 per output channel and S per layer; hidden layers clamp y to [0, 255] (this is the ReLU), the output layer clamps to [−32768, 32767] and gives z in 1/16 nat.

**ROM image** (rtl/wake/mem/wake_rom.mem, 16-bit words, 17,916 words = 286.7 kbit, exactly the engine's read order): per layer one word S; then per group of 4 output channels (o = 4g..4g+3): 8 words of biases (lane j: low 16 bits, high 16 bits), the weight words (pointwise: one per input i; depthwise: one per tap k), each holding lane j's 4-bit weight in bits 4j+3..4j, then 4 words M (lane 0..3).

## 5. Keyword scorer (golden.py `kws`)

**Template bytes** (config blob field "wake template", wake_len ≤ 255):

| Byte | Meaning |
|---|---|
| 0 | format = 1 |
| 1 | n_var, 1..4 pronunciation variants |
| 2 | clip: per-frame deficit cap, 1/16 nat |
| 3 | refractory: frames (20 ms) after a fire during which no new fire is allowed |
| then per variant | K (1..30), then K phone ids (1..39 in the order AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH) |

Sum of K over the variants ≤ 60; 0 is the CTC blank and never appears in a template.

**wake_sens** (u16 in the blob header): the allowed mean deficit per phone, in 1/16 nat.

**Per frame t:**
1. m = max_c z[c]; d[c] = max(z[c] − m, −clip).
2. For each variant (states s = 0..2K, label(s) = blank for even s, phone s div 2 for odd s), with A the previous frame's scores (int16; −32768 = impossible):
   cand(s) = max of A[s]; A[s − 1] if s ≥ 1; A[s − 2] if s is odd, s ≥ 3 and its phone differs from the previous phone; 0 if s ≤ 1 (free start).
   A'[s] = max(cand(s) + d[label(s)], −32768).
3. score_v = max(A'[2K − 1], A'[2K]) + K·wake_sens; best = max over variants.
4. If the refractory count is above zero, decrement it. Otherwise, if best ≥ 0: **fire** (wake at k = 320t + 879), set every A to −32768 and the refractory count to the template's refractory.

At the start of a session every A is −32768 and the refractory count is 0.

## 6. SDK side (sw/wake/template.py, called by sdk/compile.py)

- G2P: CMUdict (BSD-2-Clause style, docs/licenses.md), up to 4 variants from words with several pronunciations. Unknown words: a split into two dictionary words, else crude letter rules, and the SDK report warns and suggests giving phones (config "wake": {"phones": ["HH EY ..."]}).
- clip = 96 (6 nat) and refractory = 50 frames (1 s): fixed SDK constants (CLIP, REFRACTORY in template.py). Clip 160 gave the same dev curve; 48 was worse.
- Sensitivity s in [0, 1] from the config maps linearly to wake_sens = round(6 + 12 s): 0.5 gives 12, the dev operating point (at most 1 false wake per 24 h on 58 h of dev audio); 1.0 gives 18. Curves in reports/p1-wake.md.

## 7. Test vectors

sw/wake/golden/vectors.py writes three cases (breeze: one wake at a forced sensitivity and refractory 10; lumen_segments: two pronunciation variants, 11 wakes, an en drop between two segments; carrier_op: the SDK's own template at sensitivity 0.5 on "Hey Breeze, turn the fan up a little.", one wake) into rtl/wake/tb/vectors/<case>/: x.txt (samples), case.json (en segments, sens, template), cfg.mem (config SRAM with the template at byte 0x123) and expected.txt (per frame: 64 feature lines, 40 logit lines, the best score; plus "W k" per wake). rtl/wake/tb/run.sh runs them through Verilator and cocotb, with ROM and config-port stalls, and two negative controls (a corrupted input sample and a flipped ROM bit) that must fail.

## 8. Format 2 (wake v2): bigger network, 16 lanes, VAD gating

**Frozen Nov 15, 2026, 9 PM.** Parameters: sw/wake/golden/wake_int_v2.npz, identical word for word to rtl/wake/mem2/wake_rom.mem; checkpoint data/wake/q3/best.pt. Format 2 is the chip's wake model from now on; format 1 (mem/) stays available through the parameters.

Format 2 keeps sections 3 to 6 unchanged (front end, integer layer rules, keyword scorer, template bytes, SDK G2P) and changes three things. The same RTL serves both formats through parameters (mg_wake: LANES, C, NOUT, GATE, PREROLL); format 1 is LANES = 4, C = 96, NOUT = 40, GATE = 0, PREROLL = 0.

1. **Network width.** C = 240 channels (same 12-layer structure and dilations): 318,960 weights at 4 bits. 16 MAC lanes and 4 requantization multipliers; the output layer is padded to NOUT = 48 channels with zero weights, biases and multipliers, and the padded logits are dropped.
2. **ROM image** (rtl/wake/mem2/wake_rom.mem, 22,083 words of 64 bits = 1,413 kbit): the format 1 order with LANES lanes per word: per layer one word holding S; per group of 16 output channels 8 bias words (lane j's int32 at bit 32j of the 512-bit concatenation of the 8 words, word 0 lowest), one weight word per input or tap (lane j's 4-bit weight at bit 4j), 4 multiplier words (lane j's M at bit 16j of the 256-bit concatenation). Format 1's image is the LANES = 4 case of the same rule (sw/wake/export.py rom_words).
3. **VAD gating** (golden.py run2):
   - in_vad as seen with sample n is the flag of VAD frame floor(n/256) − 1 (held between VAD frame ends; 0 before the first). The spotter samples it with the trigger sample of real-time frame t' (n = 320t' + 559): v(t').
   - The front end runs every frame (features go into a ring of PREROLL + 1 = 11 slots). Frame t is run through the network and the keyword DP at real-time frame t + PREROLL (PREROLL = 10, 200 ms), and only if v(t), ..., v(t + PREROLL) contains a 1 ("active"). Frames whose t + PREROLL is past the end of the stream are never run.
   - A run is a maximal stretch of active frames. At the first frame of a run the network's history is zero (taps before the run read 0), every DP state is −32768 and the refractory count is 0.
   - A fire for frame t raises `wake` with the in_valid of sample **k = 320(t + PREROLL) + 879**.
   - Power intent: outside runs only the front end, the sample ring and the schedule are clocked; the network, ROM and DP are clock-gated off.
