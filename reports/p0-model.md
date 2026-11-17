# P0: the Marigold intent model (Nov 15-16, 2026)

Model agent report for iteration 1. Everything here is software; numbers on audio are measured on **synthetic TTS
speech only** (held-out voices), since no real recordings exist yet. Targets and rules: docs/iteration-1.md.

## Summary

v1 is frozen and works end to end in integers: PCM in, int8 meaning vector out, matched against a config blob,
values parsed, host frame out (export `data/golden/mg-v1.npz`, test vectors `vectors-mg-v1-0..3.npz`, format
docs/golden-format.md v1.0). It is bit-exact against itself (int64 path equals the fast path, and the chunked
streaming schedule equals whole-segment evaluation), 4,586,560 weights, 18.8 Mbit of ROM as built, about 4.8x real
time on the core, and about 172 ms estimated response time. Accuracy is far below targets 1 to 4 on synthetic speech:
unseen phrasings 32.5% (fan) and 30.1% (microwave) at the 95% non-command rejection point chosen on dev (53% and 73%
with rejection off), typed phrasings 44% and 47%, values 65% and 39% exact. The encoder learned from about 39 hours
of synthetic commands and 70 hours of other speech in one evening; it sits far from its text teacher (cosine about
0.6), and rejection is where it fails. A v1.1 accuracy run (same format) is training overnight.

| # | Target | v1 (integer golden, synthetic TTS, held-out voices) | Met? |
|---|---|---|---|
| 1 | Typed phrasings >= 97% | fan 43.8%, microwave 46.9% | no |
| 2 | Unseen phrasings >= 90% | fan 32.5%, microwave 30.1% (53.3% / 72.8% with rejection off) | no |
| 3 | Values >= 95% exact | fan 64.7%, microwave 38.9% (given the right action) | no |
| 4 | Rejection >= 95% | fan 89.5%, microwave 97.1% on test (threshold set for 95% on dev) | microwave only |
| 8 | <= 5M weights, <= 20 Mbit | 4.59M weights, 18.8 Mbit as built | yes |
| 10 | Noise <= 5 points loss | fan config -4.5 (fan noise) / -5.3 (microwave noise); microwave config -6.9 / -7.2 | no (borderline on fan) |
| 11 | One model, two configs | yes: same weights, fan and microwave blobs | yes |
| 7 | <= 300 ms median | about 172 ms estimated (128 ms end-of-speech hang + 33 ms encoder tail + 11 ms match), not measured | estimate only |

## 1. What was built

| Piece | Where | State |
|---|---|---|
| Command embedder v3 (the teacher, SDK side only) | data/models/teacher-v3, sw/gate0b/ | frozen 6:35 PM |
| Teacher-side projection to D = 128 | data/models/teacher-v3/marigold_proj.npz | frozen with the teacher |
| Intent encoder (float, QAT) | sw/intent/model.py, train.py | v1 |
| Value head (CTC + integer parse) | sw/intent/values.py | v1 |
| Integer golden model + export + tests | sw/golden/intent_np.py, intent_export.py, intent_test.py | v1 |
| Whole-chip golden (segment, match, events) | sw/golden/chip_golden.py (island rules from rtl/model/mg_model.py) | v1 |
| SDK compiler (teacher projection, conf_sh, level words) | sdk/compile.py | updated |
| Evaluation on synthetic audio | sw/intent/evaluate.py | v1 |
| Format | docs/golden-format.md | v1 |

## 2. Teacher v3 (command embedder)

bge-small-en-v1.5 (MIT), fine-tuned with the gate-0b v2 recipe (supervised contrastive loss within one product per
batch, tau 0.1, 32 LibriSpeech transcript snippets per batch as their own classes) on 72 training appliances: the
32 of gate 0b plus 40 new ones chosen for the action types the fan failed on (swing, oscillate, rotate, tilt, patrol,
sweep) and for levels, timers, temperatures and opposites (sw/gate0b/products_v3.json; no fans of any kind, no
fan-like blowers, no microwaves). Text: 19.6k command phrasings and 6.6k non-commands including near misses
(data/gate0b/train_text_v3.json), written by local Qwen2.5 (32B for the first 32 products, 14B for the new 40 after
the 32B model spilled off the shared GPU), every line identical to an eval line dropped (sw/textfilter.py; the v1/v2
text had 433 such lines, see reports/gate0/text-bound.md).

Selection used 10 held-out TRAINING products only (8 typed examples per action, threshold at 95% rejection of half
of their non-commands): bge-small 92.1% accuracy / 94.9% rejection in the 128-d chip space, bge-base 93.2% / 91.6%;
the small model won on the mean and is also the smaller SDK model. D = 128: the projection scored as well as 256 and
the full 384-d space, and it doubles the prototypes that fit in the 32 KB config.

Eval test, measured once after the freeze (text only, the upper bound for audio): `knn_neg` with the SDK expansion
fan 85.6% unseen / 93.2% rejection, microwave 87.5% / 90.1%. Text matching still misses targets 2 and 4 together.

## 3. Intent encoder

Moonshine Streaming Tiny's frontend and its first two encoder layers (MIT), fine-tuned; same op set and integer
arithmetic as Poinsetta (docs/golden-format.md). 80-sample frames, per-frame norm, asinh, linear 80 to 320 + SiLU,
two causal k5 s2 convolutions (to 640, then 320; 50 frames per second), two pre-norm layers of band attention (8 heads
of 40; keys t-16..t+4 in layer 0, t-16..t in layer 1, so 80 ms look-ahead) and GELU MLPs (1280), final norm. Head:
mean over frames, projection 320 to 128, gain-free norm, int8 q (5 fractional bits). Value head: per-frame CTC over 45
value tokens. 4,586,560 weights. A first from-scratch design (W 240, 6 layers, 3.4M weights) learned too slowly
(cosine 0.12 to the teacher after 5k steps; the pretrained start passed that in 1k) and was dropped at 7 PM.

Training (sw/intent/train.py): distillation to the teacher vectors of the transcripts (1 - cosine, plus in-batch
relational KD at tau 0.05) and CTC on the value words (weight 0.3). Data, all commercially clean and never fan,
microwave or eval text: TTS training commands and non-commands (data/tts/train: train voices, the 32 gate-0b
products, 26.3k clips, 13.2 h), LibriSpeech train-clean-100 and train-other-500 (11.2k utterances up to 5.5 s plus
38.3k 1.5-5 s crops of longer ones with position-estimated words, no CTC on crops), Speech Commands words outside
its validation and test lists (84.8k). Augmentation per clip: 0.1-0.5 s pre-roll (a word tail or silence, standing in
for the end of the wake phrase), trailing silence, reverb (30%), noise from data/noise train split (75%: fan or
microwave at 0-20 dB, otherwise babble, TV, music, background at 5-25 dB), random gain, int16 rounding.

Training ran as a chain of warm restarts (logs data/logs/train_f2..f4*.log), all on the RTX 4090 shared with the wake
and data agents: f2 from the Moonshine start (42k steps, mixed batches); f3 with product-grouped batches (the TTS
commands of a batch come from two products, so the in-batch negatives are the same product's other actions); f4 after
the 46.8k v3 TTS clips (40 new appliances, near-miss non-commands) landed at 7:40 PM, with the CTC weight raised to 1.0
when the value head was found to drop most number words. Training cosine to the teacher (augmented batches, all
sources) rose 0.10 (from scratch, dropped) / 0.33 (f2, 10k) / 0.48 (f2, 42k) / 0.50 (f4). Three CUDA "illegal memory
access" faults on the shared GPU were survived by sw/intent/run_resume.sh (restart from the last 1k-step checkpoint).

## 4. Value head

Per-frame CTC over 45 tokens (blank, zero..nineteen, twenty..ninety, hundred, thousand, a, half, quarter, second,
minute, hour, degree, percent, level, low, medium, high, max, min) on the encoder output, greedy decoding, then an
integer parse into one raw value per config slot, each slot on its own and independent of the action, because the
core computes VAL before the island picks the action (docs/golden-format.md section 6). Level words leave the core as
codes -1..-5 and the island maps them with the slot entry's low/medium/high bytes, then clamps (coordinator rulings).
Training targets are the value words of each transcript (digits, times, % and degree signs expanded first; "level"
marked before a number after level/speed/setting/notch/power); approximate LibriSpeech crops carry no CTC loss. The
parse handles compound numbers ("three fifty" = 350), clock readings ("one thirty" = 90 s for a duration slot),
fractions ("two and a half minutes" = 150 s, "a minute and a half" = 90 s) and minute-second pairs; unit tests in
sw/intent/values.py.

## 5. Quantisation and the integer golden model

Weights only, as Poinsetta: per-output-row symmetric fake quantisation with learned log-scales (initialised at the
MSE-optimal scale), straight-through rounding, 4-bit for every matrix except fe.lin and ctc (8-bit, 40k weights), so
the ROM stays under 20 Mbit. QAT starts from the final float model and distils from it (1 - cosine to the float q,
weight 0.5) on top of the training loss, at LR 1e-4 (scales 5e-4). Activations follow Poinsetta's integer
arithmetic: int16 at static per-site formats calibrated on 400 training clips of segment length (half with training
noise), 24-bit residual, LUT SiLU/GELU/asinh, exp2 softmax, integer LayerNorm (docs/golden-format.md). Export rule:
norm gains clamped to int16 (core agent).

Exactness (sw/golden/intent_test.py): FAST (float64 BLAS) equals int64-only on every trace, and the encoder on every
chunk-aligned prefix reproduces the whole-segment integers for every final frame (streaming equivalence).
Float graph vs integer q: cosine 0.98 on average on calibration clips; the worst cases come from the int8 clip of
q (5 fractional bits, about +-4 standard deviations), by design.

## 6. Results on synthetic audio

Integer golden model v1 (`sw/intent/evaluate.py --golden data/golden/mg-v1.npz --enroll`, raw:
reports/p0/eval-golden-v1.json). Audio: data/tts/eval (held-out TTS voices, 3,191 clips) and its 10 dB fan and
microwave copies. SYNTHETIC ONLY. Per product, the threshold and margin are the dev-tuned pair that rejects 95% of
the clean dev non-commands while keeping the most dev commands; the matching variant (text prototypes, audio
enrollment, speech-bank reject prototypes, bias normalisation) is also chosen on dev; test numbers use them unchanged.
Rejections count as errors.

| Product (variant chosen on dev) | Condition | Unseen (no reject) | Typed | Rejection (other-device) | Values given action / end to end |
|---|---|---|---|---|---|
| Fan (text prototypes, N 208) | clean | 32.5% (53.3%) | 43.8% | 89.5% (67%) | 64.7% / 35.3% |
| | fan 10 dB | 28.0% (52.3%) | 31.9% | 92.0% | 61.5% / 28.2% |
| | microwave 10 dB | 27.2% (51.3%) | 32.6% | 92.8% | 59.0% / 25.0% |
| Microwave (audio enrollment + speech bank + bias norm, N 176) | clean | 30.1% (72.8%) | 46.9% | 97.1% (91%) | 38.9% / 19.4% |
| | fan 10 dB | 23.2% (66.8%) | 41.7% | 96.7% | 31.9% / 11.8% |
| | microwave 10 dB | 22.9% (68.1%) | 43.8% | 97.1% | 34.0% / 11.5% |

Per action (clean, unseen): fan on 20%, off 19%, faster 39%, slower 38%, set_level 46%, oscillate 33%; microwave
start 28%, add_30s 28%, stop 23%, power_level 42%. Test sizes: fan 935 unseen, 144 typed; microwave 624 and 96;
486 headline non-commands.

The float model before QAT (reports/p0/eval-f4-final.json) scored within a few points of the integer model (fan
30.4% unseen with audio enrollment, microwave 34.6%), so 4-bit QAT and the integer arithmetic cost little; the
encoder is the limit. How the numbers moved during the evening (clean, unseen with rejection, best variant): step 8k
of the first Moonshine-start run 9-11%; 42k 20-22%; after the v3 TTS and grouped batches 28-30%; v1 30-33%.

Matching variants on dev decide per product, and they differ: audio enrollment (the config's phrases spoken by TTS
train voices through the chip model) raised rejection-off accuracy by 10 to 15 points, mostly on microwave; a
LibriSpeech reject bank and per-prototype bias normalisation (cohort mean) helped rejection. These are SDK-side
steps; the chip and its weights do not change.

## 7. Size and response time

`sw/intent/budget.py --tag v1` (reports/p0/budget-v1.json): 4,586,560 weights; packed ROM 18.82 Mbit = 18.51 Mbit
of weights (4.55M at 4 bits, 40k at 8 bits) + 0.25 Mbit per-row multipliers, shifts and biases + norm gains and LUTs
(target 8: met). Compute 5.67M MACs per 20 ms frame (frontend 3.17M, two layers 2.46M), 284M MACs per second of
audio, 4.8x real time on the 16-lane, 98 MHz core at Poinsetta's 87% efficiency. Response-time estimate (target 7,
not measured): 128 ms end-of-speech hang (architecture.md 8.1) + 33 ms for the last chunk and the 4 look-ahead frames
+ 0.3 ms head, parse and bridge + 10.7 ms matching = about 172 ms, plus the VAD's own lag; RTL timing will replace
it.

## 8. Gaps and next steps

Missed in v1 (all measured on synthetic audio): see section 6. The encoder is far from its teacher (training cosine
about 0.5), so commands and ordinary speech overlap; with the reject threshold set for 95% non-command rejection, most
commands fall under it. Text matching itself (the teacher) also stays below 90 / 95, so even a perfect encoder would
miss targets 2 and 4 with this teacher.

Iteration-2 levers, roughly in order of expected gain:
1. More audio for the encoder: real recordings (many speakers, rooms, far field) and much more TTS of the command
   text (we used 39 h of TTS for 72 appliances; the text set is ready for more voices), plus more general speech
   (all of LibriSpeech and other CC BY sets cut into segment-length clips with forced alignments instead of crops).
2. A deeper student within 5M weights: distill the full 6-layer Moonshine encoder (or a pruned 4-layer one with a
   narrower frontend) instead of 2 layers; the 2-layer model must turn acoustics into meaning with mean pooling only.
3. Longer training (we had about 3.5 GPU-hours, shared): the curves were still rising when stopped.
4. Attention pooling (one learned query over all frames, the same softmax-mix op) so filler, the wake-phrase tail and
   silence stop diluting the mean.
5. A better teacher: more appliances and non-command data, a margin-aware loss, and a larger base model (the
   teacher lives only in the SDK, so its size is free); text is still short of 90 / 95.
6. SDK side: audio enrollment helped (section 6); add more TTS voices per phrase, per-product reject banks, score
   normalisation with wider bias range, and a calibrated threshold per product chosen on the OEM's own phrases.
7. Value head: a dedicated CTC tap from an earlier layer or a small hidden layer, and number-heavy TTS (prices,
   times, temperatures) for the CTC.

## 9. Iteration 2: model v2 (Nov 16, 2026)

**Model v2 ships** (tag `model-v2`, export `data/golden/mg-v2.npz`, vectors `vectors-mg-v2-0..3.npz`; same golden format
v1.0, D = 128, same 4,586,560 weights and 18.82 Mbit ROM, so it swaps in as a ROM image). Ship rule: better than v1 on
dev. Integer dev unseen accuracy at the 95%-rejection point, mean of the two products: **47.0% against v1's 36.7%**.
Whole-chip regression `rtl/top/regress_mg.sh v2` (Nov 16, 18:59, rc 0): every positive bit-exact (q, values, GPIO,
front end; 149 to 163 ms), the ROM, mic and CRC negative controls fail as required, MAC lane tests 0 failures.

What changed (chip side unchanged):
- **Teacher v4** (SDK only): Qwen3-Embedding-4B (Apache 2.0) + LoRA 16, trained in the 128-d chip space (a trained
  head saved as marigold_proj.npz) with the gate-0b supervised contrastive loss, opposite-action hard negatives,
  a margin and LibriSpeech spread, on 120 training appliances (72 + 48 written by qwen2.5:14b; no fans or
  microwaves). Chosen on 10 held-out training products: chip-space 94.4% / 94.3% against v3's 92.3% / 94.9%
  (Qwen3-0.6B 93.6 / 94.2, bge-large 92.5 / 92.6; Qwen3-8B did not fit under the 20 GB cap). sw/teacher4/.
- **Encoder trained for the decision** (sw/intent/train.py --protos): a softmax over the product's action
  prototypes (8 per training product-action, max over prototypes) plus a reject class (16 prototypes per product or
  a learned threshold), margin 0.05, temperature 0.05; LibriSpeech is rejected against every product; Speech Commands
  words get no decision loss; cosine and relational distillation to teacher-v4 and CTC kept. 150 min float (f6, from
  v1.1's float model) on 22.4 h of new TTS (48 products) plus the old data, then 40 min QAT with the same losses.
- **Audio enrollment in the SDK**: sdk/compile.py enroll_protos is now the one implementation the evaluation uses.

Gate 0 (text only, teacher-v4, SDK blob, integer scores, eval/text dev thresholds; sw/teacher4/text_gate.py):
eval/text test fan 91.7% / 93.8%, microwave 94.7% / 93.2% (teacher-v3 under the same protocol 50 / 93 and 56 / 93);
the blind **eval/text_v2: fan 71.9% / 93.8%, microwave 75.0% / 93.3%: gate (90 / 95) missed**. Review D3: the JSON
files of that run had an older layout; reports/gate0/text_gate_teacher_v4.json and _v3.json were re-run on Nov 17 with
the current script and the same protocol: teacher-v4 fan 90.4 / 93.8 (test) and 71.7 / 93.8 (v2), microwave 94.7 /
93.2 and 75.0 / 93.8; teacher-v3 identical to the first run. The teacher-v4 drift of up to 1.3 points comes from bf16
embedding numerics with a different batch order, which moves the knife-edge threshold. Most v2 misses are
rejected long, chatty commands; short fillers are accepted.

v1 against v2, integer golden models, synthetic speech by held-out voices (eval/text test and its 10 dB copies;
eval/text_v2 spoken by the same eval voices), threshold and margin for 95% rejection of the clean dev non-commands,
matching variant chosen on dev (v1: fan text, microwave audio+bank+cal; v2: audio+bank+cal for both); rejections count
as errors. SYNTHETIC ONLY. Raw: reports/p0/eval-golden-v1-v2.json, eval-golden-v2.json.

| | Fan v1 | Fan v2 | Microwave v1 | Microwave v2 |
|---|---|---|---|---|
| Dev unseen (selection) | 33.6% | 47.2% | 39.8% | 46.8% |
| T2 unseen, clean (rejection off) | 32.5% (53.3%) | **40.0%** (64.6%) | 30.4% (72.0%) | **39.7%** (75.5%) |
| T4 rejection, clean (other-device) | 89.5% (67%) | **95.1%** (72%) | 97.3% (91%) | 96.9% (77%) |
| T1 typed | 43.8% | 58.3% | 49.0% | 63.5% |
| T3 values given the action | 64.7% | 64.1% | 38.9% | 41.0% |
| Fan noise 10 dB, unseen | 28.0% | 34.3% | 23.2% | 30.1% |
| Microwave noise 10 dB, unseen | 27.2% | 33.8% | 22.9% | 27.9% |
| eval/text_v2 unseen (rejection off) / rejection | 24.1% (45.7%) / 92.4% | 25.7% (50.6%) / 90.8% | 24.7% (64.4%) / 94.9% | 25.3% (64.7%) / 94.2% |

v1's clean and eval/text_v2 cells are the Nov 16 rerun (reports/p0/eval-golden-v1-v2.json; microwave differs from section 6 by
under 3 points because its speech bank is drawn from the rebuilt training index); v1's noise rows are section 6's.

Targets 1 to 4 are still far off; target 10 (noise within 5 points) is worse than v1 in points lost (fan config -5.7 /
-6.2, microwave config -9.6 / -11.8). QAT cost (review D1; the earlier "within about a point" was wrong for the microwave): integer qv2 against float f6,
unseen accuracy, same protocol (reports/p0/eval-f6.json against eval-golden-v2.json): fan clean 40.0 -> 40.0, fan noise
33.6 -> 34.3, microwave noise 33.4 -> 33.8, eval/text_v2 24.1 -> 25.7 (dev 49.7 -> 47.2); microwave clean 43.9 -> 39.7
(-4.2), fan noise 33.8 -> 30.1 (-3.7), microwave noise 35.7 -> 27.9 (-7.8), eval/text_v2 30.0 -> 25.3 (-4.7) (dev
47.7 -> 46.8). Review S1 applied: QAT restarted the learned decision threshold at 0.45 (f6 had learned 0.462), a small
change; the microwave loss is not explained yet.
The audio encoder, not the teacher, is the limit: text matching with the same teacher reaches 72 to 95%.

Next (approved, running as milestone c): an audio teacher (all six Moonshine layers, PC only) distilled into the same
2-layer chip student on LibriSpeech crops plus the TTS; and the reject fix (SDK reject bank, calibration on the new
dev set eval/text_dev2, blind eval/text_v3 scored once).

## 10. Milestone (c): audio teacher, distilled student, model v3 (Nov 16-17, 2026)

**Model v3 ships** (tag `model-v3`; export `data/golden/mg-v3.npz`, vectors `vectors-mg-v3-0..3.npz`, ROM image
rtl/core/imh_rom.svh from data/core/mg-v3; same golden format v1.0, 4,586,560 weights, 18.82 Mbit). Ship rule: better
than v2 on dev under the same protocol. Integer, reject-fix protocol (below), dev2 unseen accuracy at the dev2
95%-rejection point: **v3 37.2% against v2 32.9%** (fan 34.8 / 32.2, microwave 39.7 / 33.6). Whole-chip regression on
v3 (Nov 17, 01:13): libri-pos and breeze-pos identical in frames, q, values, GPIO and front end; the ROM, mic and CRC
negatives fail; MAC lanes 0 failures (judged per check: the run used the old script, whose rc is not meaningful, R9).

What was done (John approved; chip side unchanged):
- **Audio teacher at6** (PC only): the full 6-layer Moonshine Streaming Tiny encoder (MIT, Moonshine's own windows
  4,4,0,0,4,4) plus the same head, trained 150 min into teacher-v4's chip space with the decision loss. Float, eval/text
  dev protocol: fan 46.6% unseen / 97.5% rejection, microwave 47.9% / 93.4% (f6: 40.0 / 95.3, 43.9 / 94.9); dev 54.9 /
  53.7%. Better than the 2-layer student, but far from text (72 to 95%).
- **Student f7**: f6 distilled from at6 (1 - cos and relational KD on the same augmented audio) for 180 min, keeping the
  text distillation, decision and CTC losses, plus 25% LibriSpeech crops without text from train-clean-100 and
  train-other-500 (600 h; train-clean-360 was still downloading). Then 40 min QAT (qv3) with the same losses.
- **Reject fix** (SDK; sw/teacher4/text_gate.py, sdk/compile.py --reject-bank): a generic bank of 35,201 lines (commands
  of the 120 training appliances, their non-commands, LibriSpeech snippets) gives action-0 prototypes; bank lines close
  to the product's own phrases are dropped; bank size, closeness cut-off, threshold and margin are chosen on the new dev
  set eval/text_dev2 (qwen2.5:14b, 360 commands, 250 non-commands). The blind set eval/text_v3 (gemma4:12b, 541
  commands, 276 non-commands) was written before any result and scored once.
  - Text (teacher-v4): v2 (seen) fan 71.9 -> 74.2% unseen, 93.8 -> 94.6% rejection; microwave 75.0 -> 72.9%, 93.3 ->
    92.5%. v3 (blind): fan 95.0% / 94.5%, microwave 55.7% / 91.1% (start 18%, power_level 33%: 78 of 80 errors are
    rejections, mostly lines that write numbers as digits, "start for 180", with the 10,000 margin chosen on dev2).
  - Audio, float f6, eval/text-dev protocol -> fix protocol: fan clean 40.0 -> 34.9%, v2 24.1 -> 20.5%; microwave
    clean 43.9 -> 37.0%, v2 30.0 -> 32.4%. The fix does not help audio overall: the dev2 operating point sits higher.
    It is kept as the protocol so that v2 and v3 are compared on the same footing; rejection-off and calibration need
    more work (review K2, S7).

Integer v2 against v3, both under the reject-fix protocol (calibration on eval/text_dev2 audio, reject bank, variant
chosen on dev2: "audio" for both), synthetic speech by held-out voices, rejections count as errors (raw:
reports/p0/eval-golden-v2-fix.json, eval-golden-v3-fix.json). Cells: unseen (rejection off) / headline rejection.

| Set | Fan v2 | Fan v3 | Microwave v2 | Microwave v3 |
|---|---|---|---|---|
| dev2 (selection) | 32.2 (55.6) / 95.1 | **34.8** (57.5) / 95.1 | 33.6 (64.7) / 95.1 | **39.7** (63.2) / 95.1 |
| eval/text clean | 34.5 (61.3) / 95.5 | **39.3** (65.5) / 97.5 | 41.8 (70.2) / 94.4 | **43.8** (72.4) / 95.3 |
| eval/text, fan noise 10 dB | 28.7 / 97.1 | 33.7 / 97.7 | 31.6 / 96.9 | 36.4 / 94.9 |
| eval/text, microwave noise 10 dB | 28.7 / 97.7 | 32.4 / 97.7 | 31.2 / 97.1 | 36.2 / 95.1 |
| eval/text_v2 (seen) | 23.0 (49.0) / 95.1 | 25.7 (52.3) / 95.0 | 31.0 (62.6) / 94.6 | 36.9 (63.6) / 93.1 |
| eval/text_v3 (blind, once per model) | 39.9 (68.1) / 92.8 | **42.2** (69.4) / 91.5 | 39.6 (70.0) / 95.1 | **43.5** (68.2) / 93.8 |
| Typed phrasings | 50.7 | 63.9 | 61.5 | 63.5 |
| Values given the action (clean) | 64.1 | 64.1 | 41.0 | 41.3 |

Float to integer (review D1), unseen accuracy, same protocol: f7 -> qv3: fan clean 39.0 -> 39.3, fan noise 33.6 ->
33.7, microwave noise 33.6 -> 32.4, v2 25.1 -> 25.7, v3 42.8 -> 42.2, dev2 35.1 -> 34.8; microwave clean 45.2 -> 43.8,
fan noise 37.0 -> 36.4, microwave noise 38.0 -> 36.2, v2 38.1 -> 36.9, v3 41.2 -> 43.5, dev2 40.1 -> 39.7 (within
-1.8 to +2.3 points). Review S1 also applied to qv3: QAT restarted the threshold at 0.45 (f7 had learned 0.458).

Still missed: targets 1 to 4 (best unseen about 44% at 95% rejection on synthetic speech) and 10 (noise loss 5 to 7
points). The audio encoder remains the limit: the 6-layer audio teacher reaches about 47 to 48% at the same point, so
distillation can at best close part of the gap to text.

## 11. Post-merge reruns (Nov 17, 2026; review S6, S7)

The integer golden models v1, v2 and v3 scored once each with the merged review fixes: the frozen enrollment speech bank
(sw/intent/speech_bank.txt, S6), both rejection-off figures (S7), the budget check on every variant (K3), and the
reject-fix protocol (calibration and variant choice on eval/text_dev2 audio, reject bank with the per-product settings the
text gate chose on dev2; v1 uses a teacher-v3 bank built the same way, with the same settings, no tuning). No tuning
on any test set. Raw: reports/p0/eval-pm-golden-v1.json, -v2.json, -v3.json; sw/intent/run_postmerge.sh.
Cells: end to end / rejection off (threshold and margin off) / pure rejection off (reject prototypes masked too)
(headline rejection), unseen accuracy in %, synthetic speech by held-out voices.

| Set | v1 | v2 | v3 |
|---|---|---|---|
| **fan** dev2 (selection; variants: v1 mix+bank+cal, v2 audio, v3 mix+bank) | 27.5 | 32.2 | 35.3 |
| fan eval/text clean | 29.1 / 56.1 / 61.6 (94.0) | 34.5 / 61.3 / 66.3 (95.5) | 37.1 / 58.3 / 66.6 (97.9) |
| fan eval/text, fan noise 10 dB | 24.0 / 50.2 / 56.6 (95.3) | 28.7 / 56.1 / 63.4 (97.1) | 31.1 / 54.3 / 64.5 (96.5) |
| fan eval/text, microwave noise 10 dB | 23.3 / 49.6 / 55.5 (95.3) | 28.7 / 53.3 / 61.7 (97.7) | 29.2 / 51.9 / 63.1 (97.3) |
| fan eval/text_v2 (seen) | 21.8 / 41.5 / 50.8 (95.7) | 23.0 / 49.0 / 59.0 (95.1) | 24.2 / 45.2 / 56.1 (94.2) |
| fan eval/text_v3 (blind) | 40.8 / 65.0 / 71.8 (94.2) | 39.9 / 68.1 / 74.9 (92.8) | 37.9 / 62.3 / 71.0 (92.4) |
| **microwave** dev2 (selection; variants: v1 audio+bank+cal, v2 audio, v3 audio) | 37.9 | 33.6 | 39.7 |
| microwave eval/text clean | 38.3 / 69.4 / 75.6 (94.2) | 41.8 / 70.2 / 81.1 (94.4) | 43.8 / 72.4 / 82.9 (95.3) |
| microwave eval/text, fan noise 10 dB | 30.0 / 64.6 / 72.1 (95.3) | 31.6 / 64.1 / 76.6 (96.9) | 36.4 / 68.9 / 80.8 (94.9) |
| microwave eval/text, microwave noise 10 dB | 30.6 / 65.4 / 72.3 (94.9) | 31.2 / 63.0 / 74.8 (97.1) | 36.2 / 68.8 / 79.6 (95.1) |
| microwave eval/text_v2 (seen) | 32.6 / 62.4 / 76.5 (92.5) | 31.0 / 62.6 / 74.7 (94.6) | 36.9 / 63.6 / 74.3 (93.1) |
| microwave eval/text_v3 (blind) | 46.7 / 71.8 / 79.5 (91.4) | 39.6 / 70.0 / 79.5 (95.1) | 43.5 / 68.2 / 78.7 (93.8) |

**Ship ordering on dev2 (mean of the two products): v3 37.5% > v2 32.9% > v1 32.7%.** S6 and S7 do
not change the v2-against-v3 call. Two plain findings:
- v1 and v2 nearly tie under this protocol (0.2 points on dev2), while v2 beat v1 by 10 points under the iteration-1
  protocol (eval/text dev, section 9). The calibration set decides much of the end-to-end number.
- v3's fan variant changed between runs (now mix+bank, dev2 35.3%, against audio at 34.8% in section 10) because the
  text-side prototypes move slightly from run to run (bf16 teacher numerics, as in the D3 note). With it, v3's fan
  scores on the blind v3 set drop from 42.2% to 37.9%, below v2's 39.9%. The microwave choice and numbers repeat.
- The pure rejection-off figures are 5 to 15 points above the plain rejection-off ones: reject prototypes win for many
  commands even with no threshold. Action confusion alone (pure) is about 61 to 67% (fan) and 75 to 83% (microwave) on
  clean speech; the rest of the gap to the end-to-end number is rejection.

**Deterministic SDK embedding (Nov 17, later).** The run-to-run drift was an SDK bug: sdk/compile.py's STEmbedder (also
used by the evaluation and the text gate) embedded phrases in bf16 batches, so a phrase's vector depended on its batch
neighbours. Each phrase now runs alone (no padding, no batch-composition effect) with deterministic kernels and TF32 off;
sdk/tests/test_determinism.py compiles fan.json with teacher-v4 in two processes at caller batch sizes 1000 and 7 and
gets identical blob bytes (27,599 bytes). v3's integer eval re-run once under it (reports/p0/eval-pm-golden-v3-det.json,
same protocol, no tuning): fan chooses **audio** (dev2 34.8%; mix+bank 34.3%), clean 39.3 / 65.7 / 73.2 (97.5), v2
25.7 / 52.3 / 60.4 (95.0), blind v3 42.2 / 69.5 / 77.9 (91.7); microwave audio, dev2 39.4%, clean 43.4 / 72.4 / 82.9
(95.5), v2 36.4 / 63.3 / 74.3 (93.3), blind v3 43.5 / 67.9 / 78.7 (93.8). This repeats section 10's v3 numbers; v3's
dev2 mean is 37.1% against v2's 32.9%, so the v2-against-v3 call stands (v2 was not re-run; its chosen variants are
audio-only for both products).

