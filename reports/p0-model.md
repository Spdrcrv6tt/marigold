# P0: the Marigold intent model (Nov 15-16, 2026)

Model agent report for iteration 1. Everything here is software; numbers on audio are measured on **synthetic TTS
speech only** (held-out voices), since no real recordings exist yet. Targets and rules: docs/iteration-1.md.

## Summary

RESULTS_SUMMARY

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

RESULTS_TABLE

## 7. Size and response time

BUDGET_SECTION

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
