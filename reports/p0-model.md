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

TRAINING_RESULTS

## 4. Value head

VALUES_SECTION

## 5. Quantisation and the integer golden model

QAT_SECTION

## 6. Results on synthetic audio

RESULTS_TABLE

## 7. Size and response time

BUDGET_SECTION

## 8. Gaps and next steps

GAPS_SECTION
