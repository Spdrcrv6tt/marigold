# Phase 1: wake spotter (wake agent, Nov 15, 2026)

## Wake v2 (golden format 2, frozen Nov 15, 9 PM): the bigger spotter, now the chip's wake model

John asked for a bigger spotter after v1. v2 keeps v1's front end, layer rules, keyword scorer and template, widens the network to 240 channels (318,960 weights at 4 bits, ROM image 1,413 kbit), runs it on 16 MAC lanes, and adds VAD gating (docs/golden-wake.md section 8). The RTL is the same parametric code as v1 (rtl/wake/, mg_wake #(.LANES(16), .C(240), .NOUT(48), .GATE(1), .PREROLL(10), .MEMDIR(".../rtl/wake/mem2/"))); the interface to the core is unchanged except that a wake now comes 200 ms later (k = 320(t + 10) + 879).

**v1 vs v2, frozen integer models, same test protocol** (synthetic TTS from the 21 eval voices, 84 clips per condition; false wakes over the same 53.5 h; SDK default sensitivity 0.5 = wake_sens 12, chosen on dev for both):

| | v1 "hey breeze" | v2 "hey breeze" | v1 "hey kitchen" | v2 "hey kitchen" |
|---|---|---|---|---|
| clean | 64.3% | **70.2%** | 70.2% | **75.0%** |
| 10 dB SNR | 45.2% | **57.1%** | 46.4% | **69.0%** |
| synthetic room (1-3 m) | 69.0% | 67.9% | 65.5% | **82.1%** |
| room + 10 dB noise | 51.2% | 50.0% | 48.8% | **69.0%** |
| false wakes per 24 h | 2.2 | **0.4** | 0.0 | 0.0 |

- Over all six phrases at the test operating point with at most 1 false wake per 24 h: v1 (wake_sens 11) 71% clean, 39% at 10 dB SNR, 61% room, 30% room + noise; v2 (wake_sens 15, 0.97 per 24 h) 91 / 67 / 84 / 55%.
- v2 at sensitivity 1.0 (wake_sens 18): "hey breeze" 88 / 74 / 82 / 64% at 3.1 false wakes per 24 h, "hey kitchen" 91 / 81 / 93 / 86% at 5.4.
- **Targets 5 and 6 are still not met together**: at about 1 false wake per 24 h, detection is 50-82% depending on condition (95% needed). v2 roughly halves the gap in noise.
- Phone error on dev-clean: v2 17.4% float, 18.6% at 4 bits (v1: 29.8% and 32.1%). The 4-bit loss fell from 2.3 to 1.2 points.
- **What VAD gating costs:** v2 without gating at wake_sens 12 scores 71 / 58 / 68 / 57% ("hey breeze") and 74 / 70 / 82 / 70% ("hey kitchen") at the same false wakes. Gating costs 0-7 points (the most in room + noise). On dev it cost v1 up to 10 points at 10 dB SNR, because the VAD stays off in some speech-like noise. The VAD (rtl/model/vad_model.py, exact) was on for 91-95% of the speech streams, 49-51% of the music and 85% of the TV-like audio.

**Hardware v2** (estimates for 28 nm unless marked measured):
- RTL bit-exact (measured): 3 format-2 cases (VAD flags from the exact VAD model driving in_vad, 5 wakes, an en drop), plain and with ROM/config-port stalls, 0 mismatches. A corrupted sample and a flipped ROM bit fail as required. The parametric RTL still passes all v1 vectors (LANES 4, no gating).
- Clocks (measured in simulation): front end about 22k every frame; when the network runs, about 45k of 61,440 (about 52k with stalls). Averaged over the test streams: 28-35k.
- Area: ROM 1,413 kbit ≈ 0.085 mm²; logic 93k NAND2-equivalents (Yosys generic gates, memories excluded) ≈ 0.045-0.067 mm²; RAM about 127 kbit ≈ 0.015-0.03 mm². Total about **0.15-0.18 mm²**, at or a little over the 0.15 mm² limit.
- Power (hand estimate, not measured): front end always on about 3 µW; network + DP about 12-15 µW while running, so about 2.5-3 µW at 20% speech duty; leakage about 2-3 µW. Total **about 7-9 µW at 20% speech duty** (about 17-21 µW with speech all the time). Target was 25 µW or less. Cheap next cut: read each pointwise input word once per 16 inputs instead of every clock (about 25% of the network's RAM energy).
- Training: as v1 (LibriSpeech + train-voice TTS, the same augmentation), 8 float epochs + 2 QAT epochs. Sporadic CUDA "illegal memory access" faults on the shared GPU (three times, in different code) were handled by resumable checkpoints (sw/wake/train_loop.sh, eval_retry.sh).

Files for v2: sw/wake/golden/wake_int_v2.npz (= rtl/wake/mem2/wake_rom.mem, checked word for word), rtl/wake/tb/vectors2/, sw/wake/vadflags.py (exact VAD flags for every eval stream), results data/wake/res/v2g_*.json (gated) and v2_*.json (ungated), checkpoints data/wake/f3, data/wake/q3.

## Wake v1 (golden format 1, frozen Nov 15, 7 PM)

Marigold's always-on, type-to-train wake-word spotter: the OEM types the wake phrase in the SDK, the SDK turns it into phones (CMUdict), and the chip matches a phone-posterior stream against that template. Golden model frozen (format 1, docs/golden-wake.md), RTL bit-exact (rtl/wake/). **Targets 5 and 6 are not met together by this v1**; the numbers are below. Every detection number is measured on **synthetic TTS speech** (no real recordings exist yet).

## Summary

| Item | Result |
|---|---|
| Target 5, wake detection (synthetic proxy, at the SDK default sensitivity 0.5 = wake_sens 12) | "hey breeze": 64% clean, 45% at 10 dB SNR, 69% synthetic room (about 1-3 m), 51% room + 10 dB noise. "hey kitchen": 70 / 46 / 65 / 49%. **Not met** (target 95%). |
| Target 6, false wakes per 24 h at the same setting | "hey breeze" 2.2, "hey kitchen" 0.0, pooled over 53.5 h of test audio (speech, music, TV-like). Met for one phrase, not the other. |
| Sensitivity 1.0 (wake_sens 18) instead | "hey breeze" 87 / 73 / 85 / 79%, "hey kitchen" 92 / 75 / 85 / 74%, at 9.9 and 12.1 false wakes per 24 h |
| Model | 58,464 weights at 4 bits (286.7 kbit ROM image as built, tables included), 39 phones + blank, dev-clean phone error 32.1% (float 29.8%) |
| Bit-exact RTL | rtl/wake/mg_wake.sv: 0 mismatches over 3 cases (46k trace lines: every feature, logit and score of 439 frames, 13 wakes, an en drop), with and without ROM and config-port stalls; a corrupted sample and a flipped ROM bit both fail as required |
| Clocks | 40.3k of 61,440 clk_fe clocks per 20 ms frame (45.8k with stalls), measured in simulation |
| Area, 28 nm (estimate) | about 0.05-0.07 mm²: logic 52k NAND2-equivalents from Yosys (generic gates, memories excluded) about 0.025-0.037 mm², ROM 0.017 mm², RAMs 77.6 kbit about 0.009-0.018 mm². Budget about 0.05 mm²: at or somewhat over. |
| Power, 28 nm (hand estimate, not measured) | about 6-10 µW while listening (no VAD gating in v1; per-engine clock gates included). The core agent's activity-based flow should replace this number. |

## Design

1. **Front end** (docs/golden-wake.md section 3): Hann 400 samples, 10 ms hop, two frames packed into one 512-point integer FFT (real and imaginary parts), split with adds, magnitude max + 3/8 min, 32 mel bands (60-7600 Hz), log2 with a 16-entry table, uint8 codes at 0.75 dB per step. No transcendental functions, no division. The same integer code computes the training features on the GPU, so training sees exactly what the chip computes.
2. **Phone network**: causal depthwise-separable CNN at 20 ms (pointwise 64→96, five depthwise k=5 blocks with dilations 1,1,2,2,2 plus pointwise 96→96, output 96→40), receptive field 660 ms, trained with CTC on CMUdict phone transcripts.
3. **Keyword scorer**: CTC keyword DP over the template's phones with free start, a per-frame deficit d = max(z − max z, −clip) in 1/16 nat, fire when the best path score + K·wake_sens ≥ 0, then reset and a 1 s refractory. Up to 4 pronunciation variants per phrase.
4. **SDK** (sw/wake/template.py, called by sdk/compile.py): CMUdict G2P (BSD-2-Clause style license, docs/licenses.md), compound split or letter rules for unknown words with a warning, optional phones given by the OEM; sensitivity 0..1 maps to wake_sens 6..18 (0.5 → 12, the dev operating point). data/sdk/fan.mgc and microwave.mgc regenerated with real templates; `pytest sdk/tests` 12 pass.

Why this design: a phone-posterior spotter is the only way to type a wake phrase without retraining, and a 4-bit CNN with an FFT front end keeps everything integer and small. The front end costs about 22k clocks per frame (half the work), which is why it stays at 32 bands and one shared pair of multipliers.

## Training and data (all commercially clean)

- LibriSpeech train-clean-100 + train-other-500 (CC BY 4.0) plus the TTS train set (train voices only): 91,474 utterances, 225 h after dropping utterances with words outside CMUdict. Dev-clean for monitoring only.
- On-GPU augmentation before the integer front end: gain −30 to +3 dB, train-split noise (MUSAN license-clean background and music, babble, TV-like, synthetic fan and microwave) at 0-25 dB SNR on half the utterances, synthetic rooms (RT60 0.2-0.8 s) on a quarter.
- Float: 6 epochs, AdamW, OneCycle; 4-bit QAT (BN folded, per-channel weight scales, uint8 activations, fixed requantization multipliers): 1.5 epochs.
- A second run with heavier noise and reverb (80% noisy at −5 to 20 dB, 50% reverb) was worse at equal false wakes on dev (44% vs 62% detection at 10 dB SNR), so it was dropped.

## Evaluation method

- **Positives (target 5 proxy, synthetic):** 6 phrases ("hey breeze", "hey kitchen" from the configs, plus "hey marigold", "okay toaster", "hello lumen", "hey computer"), each spoken by the 21 EVAL voices (data/tts/voices.json; Kokoro and Piper) at 3 rates, plus a carrier sentence per config phrase: 84 clips per config phrase per condition. Conditions: clean (−6 dBFS), 10 dB SNR with eval-split noise (babble, TV, fan, microwave, background, music), synthetic room (RT60 0.3-0.7 s, direct-to-reverberant 0-6 dB, −16 dBFS), and room + 10 dB noise. A hit is a wake between the clip start and 0.6 s after its end. Caveats: TTS voices, synthetic rooms from the same generator family as training (different seeds), synthetic fan and microwave noise.
- **False wakes (target 6):** 53.5 h test: LibriSpeech test-clean 5.4 h and test-other 5.3 h, MUSAN librivox speech 20.4 h, held-out MUSAN music 12.4 h (files not in the training noise pool), TV-like mixes 10 h (librivox speech over held-out music at 5-12 dB SNR, band-limited, compressed). Streams are cut into 5-minute chunks (state resets at chunk starts). Counted with the exact integer scorer (same rules as the golden model) on the integer network's logits.
- **Operating point chosen on dev only:** train-voice positives (30 train voices) and 58 h dev negatives (dev-clean, MUSAN us-gov speech, the other half of the music). wake_sens 12 is the largest value with at most 1 false wake per 24 h on dev, averaged over the two config phrases (dev: 0.83 per 24 h, 80% clean detection).

## Trade-off curve (frozen integer model, test)

| phrase | wake_sens | clean | 10 dB SNR | room | room + noise | LS test-clean | LS test-other | librivox | music | TV-like | pooled FA/24 h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hey breeze | 8 | 56.0% | 25.0% | 58.3% | 32.1% | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| hey breeze | 10 | 60.7% | 38.1% | 65.5% | 41.7% | 0.0 | 0.0 | 0.0 | 0.0 | 4.8 | 0.9 |
| hey breeze | **12** | **64.3%** | **45.2%** | **69.0%** | **51.2%** | 0.0 | 0.0 | 2.4 | 1.9 | 4.8 | **2.2** |
| hey breeze | 14 | 72.6% | 54.8% | 73.8% | 59.5% | 4.4 | 0.0 | 2.4 | 1.9 | 9.6 | 3.6 |
| hey breeze | 16 | 79.8% | 63.1% | 82.1% | 67.9% | 4.4 | 0.0 | 3.5 | 1.9 | 14.4 | 4.9 |
| hey breeze | 20 | 95.2% | 82.1% | 89.3% | 83.3% | 22.2 | 4.5 | 7.1 | 40.5 | 28.8 | 20.2 |
| hey kitchen | 8 | 46.4% | 35.7% | 50.0% | 25.0% | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| hey kitchen | 10 | 58.3% | 41.7% | 54.8% | 33.3% | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| hey kitchen | **12** | **70.2%** | **46.4%** | **65.5%** | **48.8%** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** |
| hey kitchen | 14 | 82.1% | 58.3% | 71.4% | 56.0% | 0.0 | 0.0 | 0.0 | 0.0 | 9.6 | 1.8 |
| hey kitchen | 16 | 88.1% | 69.0% | 81.0% | 65.5% | 8.9 | 0.0 | 1.2 | 5.8 | 21.6 | 6.7 |
| hey kitchen | 20 | 91.7% | 75.0% | 89.3% | 81.0% | 17.8 | 18.0 | 3.5 | 30.9 | 67.2 | 24.6 |

FA columns are false wakes per 24 h (hours: 5.4, 5.3, 20.4, 12.4, 10.0). One false wake in 53.5 h is 0.45 per 24 h, so the low end of the curve is statistically thin. Over all six phrases at wake_sens 11: 0.75 false wakes per 24 h, detection 71% clean, 39% noisy, 61% room, 30% room + noise. Templates never fired on the other phrases' clips (0.0% cross-phrase false accepts at wake_sens 14).

Float model, same test at wake_sens 12: "hey breeze" 75 / 62 / 76 / 62%, "hey kitchen" 85 / 61 / 76 / 56%, 1.3 false wakes per 24 h. The 4-bit model loses 5 to 15 points at equal sensitivity.

## Where the gap is, and what would close it (iteration 2)

- **Noise and distance** cost most: detection at 10 dB SNR is 20 to 30 points below clean. **Music and TV-like audio** cause most false wakes at higher sensitivity.
- Levers, roughly in order of value per effort: (1) a second-stage confirm: the core is powered after every wake anyway and could re-score the phrase with the intent encoder before acting (the plan's two-stage check); (2) VAD-qualified wakes (in_vad is already on the interface) against music; (3) a bigger spotter (the 4-bit QAT loss and the 32% phone error point at capacity; 100-150k weights would still be about 0.03 mm² of ROM); (4) VAD power gating of the engines with a pre-roll (needed only if the island power budget is tight); (5) real recordings for evaluation and for rooms and noise in training.

## Hardware (rtl/wake/)

- Files: mg_wake.sv (schedule, 1024-sample ring, three clock-gated engines), wk_fe.sv (front end, 2 multipliers 32x17, 8 clocks per butterfly), wk_nn.sv (4 MAC lanes, 2-stage pipeline, ROM FIFO with stall), wk_dp.sv (scorer, config-port byte reader with a one-word cache), wk_mem.sv (behavioral RAM and ROM models), mem/*.mem (ROM image and front-end tables). Interface as agreed with the core agent (docs/architecture.md section 3.1): wake rides on the in_valid of sample 320t + 879.
- Verification: `bash rtl/wake/tb/run.sh` (Verilator 5.054 + cocotb from ~/.imhotep). Vectors from `sw/wake/golden/vectors.py`, which first checks that rtl/wake/mem/wake_rom.mem equals the golden parameters word for word.
- Gate count: `rtl/wake/syn/run_gates.sh` (Yosys, generic gates): 28,553 cells incl. 2,189 flops (memories black-boxed). The largest flop users are the 40 logits (640 flops) and front-end registers; moving the logits into a small RAM is an easy area cut.

## Files

sw/wake/: fe.py (integer front end), lexicon.py and template.py (G2P, template bytes), model.py, train.py, qat.py, export.py, evalset.py, kws_eval.py, summarize.py, compare.py, tts_wake.py, prep.py; golden/golden.py, golden/vectors.py, golden/wake_int.npz (frozen parameters). Results in data/wake/res/ (q1int_test.json, q1int_dev.json, f1b_*). Checkpoints data/wake/f1/best.pt (float), data/wake/q1/best.pt (QAT).
