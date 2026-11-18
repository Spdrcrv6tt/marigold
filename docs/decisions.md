# Decisions

- 2026-11-15 (John: "Poinsettia seems decently built, so I want to move on to this idea. You are to start a new session on this computer and pass it all of the information about Poinsettia / allow it to use it. I want to reuse as much as we possibly can and use as much as we've learned to build this chip as efficiently as possible."): Farallon's command chip starts, with John's plan in PLAN.md (original PDF in docs/plan/).
  - Codename **Marigold** (John's pick, Nov 15). Repo D:\marigold, junction ~/Marigold, local git only.
  - Poinsetta (D:\poinsetta, final commit 1719528) and Imhotep (D:\imhotep) are read-only references. Copy from them; never write to them. docs/poinsetta-handoff.md maps what to reuse and the lessons learned.
  - Carried over from Poinsetta: weights hardwired in on-die ROM, one bit per cell (patent guardrail); software validation first (golden model, then bit-exact RTL); only 28 nm figures matter, with GF180 as the measured basis and a 45 nm layout for grounding; every agent on Opus 5.5 at max effort; usage stop at 88% (John's cap is 90%).
  - Silicon Chatbot and Imhotep are on hold (John, Nov 15).
- 2026-11-15, decision 1 (PLAN.md, first session): **speech to intent with text-enrolled matching, no transcription.**
  - The chip's intent model maps speech to a meaning vector. The SDK, on a PC, embeds each OEM example phrase with a text-embedding model; the chip picks the closest action, and does nothing when no action is close enough (reject threshold in the config). A small value head extracts numbers, durations and levels.
  - The text-embedding model lives only in the SDK and as the training target for the audio encoder; it never goes on the chip. The chip carries one general audio encoder for every OEM (target 11): OEM customization is data in config SRAM, not weights.
  - Why: transcription needs a decoder and vocabulary (Poinsettia's 44M weights, 34.8 mm²); matching needs only an encoder and a nearest-vector search, which fits 2.5 to 4 mm². Phrasing stays open; only the action list is fixed.
  - Gate 0 tests it before any audio training: text-only matching must reach target 2 (90% unseen) and target 4 (95% rejection). If text cannot, audio cannot.
  - Held-out rule: the fan and microwave paraphrase test sets are evaluation only. Training text for the audio encoder comes from general speech (LibriSpeech transcripts) and generated command-style sentences, never from the test splits; test TTS voices are disjoint from training voices.
- 2026-11-15, config format v0 (sdk/config-format.md): JSON, one file per product: wake phrase and sensitivity, actions with example phrases, value slots (number, duration, level), GPIO mapping, host output. The coordinator wrote v0 so gate 0 could run; the SDK agent owns it from here.
- 2026-11-15, gate 0 result (reports/gate0/text-bound.md): off-the-shelf text embedders do not reach targets 2 and 4 together; the best is 88 to 89% unseen at 91 to 93% rejection (335M weights). Fine-tuning the SDK's text model for commands, trained on 32 other appliances, reaches fan 83% / 96% and microwave 90% / 93% (bge-base). Decision: continue. The text model is an SDK component we train ourselves (a "command embedder"), not an off-the-shelf model. Its training never includes the evaluation products. The audio encoder will be trained toward its vectors. Gate 0 stays open until text reaches 90 / 95 on both products; audio encoder training on GPU may start in parallel, since the data work is shared.
- 2026-11-15, 5 PM (John: "I'm allowing you the full 100%, but try and make it last the next 12-14 hours. I want a finished chip or damn close by then"): overnight push to a finished iteration-1 chip by about 7 AM Nov 16. Agent usage stop raised to 97% (coordinator keeps the rest for wrap-up). Parallel tracks:
  - model agent (sw/): command embedder v3 frozen as the teacher, then the audio intent encoder, value head, 4-bit QAT and golden model;
  - wake agent (sw/wake/): type-to-train wake spotter;
  - core agent (rtl/): whole-chip RTL;
  - SDK and data agent (sdk/, data): TTS, noise and the config blob compiler.
  - KV260 phase 3 waits for the board.
- 2026-11-15, held-out rule tightened: the audio encoder, like the command embedder, never trains on fan or microwave text (not even the config examples). The chip model is fixed before any OEM types examples, so target 1 is measured on typed phrases the model never heard, and target 11 tests products the model has never seen.
- 2026-11-15 (John: "It almost needs a way to indicate to the hardware that it wasn't able to complete a command, period. So the maker could implement some sort of status ding or something."): the chip reports events, not just commands: COMMAND, WAKE (listening), NOMATCH (heard speech, no confident match) and TIMEOUT (woke, nothing said). Host frames carry a type byte; GPIO-only products map events to pins (e.g. a buzzer for a 'didn't catch that' ding). docs/config-blob.md updated; the SDK exposes these as product options.
- 2026-11-15, 6 PM (John: "We don't need a final product by the end ... But I want it pretty damn well wrapped up by end of usage period. Then we can go ahead and refine it from there using higher class models."): the goal is a wrapped-up iteration 1, not a market product. It means working end to end, documented, honest numbers, all committed. Freeze schedule:
  - 2 AM: model and wake golden models frozen;
  - 5 AM: whole-chip bit-exact regression plus area and power estimates;
  - 6 AM: reports done and everything committed.
  Refinement comes later. No gold-plating before the freezes.
- 2026-11-15, rulings on the golden-format draft 0.1 (model agent):
  - (1) Segment pre-roll = 0.3 s (4800 samples) before the wake sample, not 1 s, so the segment carries at most the wake phrase's tail. The model trains with a random 0.1 to 0.5 s pre-roll that includes wake-phrase tails. The capture ring stays at 64 KB for a later two-stage wake confirm.
  - (2) The confidence shift goes into the blob (byte 0x1C, conf_sh; 9 at D = 128), so a later D change needs no RTL change.
  - (3) Level slots carry low, medium and high word values in the slot entry's former pad bytes (0 = default rule).
  - D (128 or 256) is the model agent's call by the teacher freeze.
- 2026-11-15, segment and value rulings:
  - End-of-speech minimum mn = 8000 samples (0.5 s from segment start), still requiring a speech onset after the wake; mn = 32000 would have failed target 7 for every short command.
  - The value head emits level-word codes -1 to -5 (low, medium, high, max, min). The island maps them through the slot table (a 0 byte means low = min, medium = floor((min + max) / 2), high = max) and clamps to [min, max]. No slot reads cross the bridge.
- 2026-11-15, 6:35 PM, teacher v3 frozen (model agent): **data/models/teacher-v3** = bge-small-en-v1.5 (MIT) fine-tuned with the v2 recipe (supervised contrastive per product, tau 0.1, 32 LibriSpeech snippets per batch, 2,500 steps) on 72 training appliances (the 32 of gate 0b plus 40 new ones with motion and toggle modes, levels, timers, temperatures and opposites: sw/gate0b/products_v3.json; still no fans of any kind and no microwaves), 19,600 command phrasings and 6,600 non-commands (data/gate0b/train_text_v3.json; every eval-identical line dropped, sw/textfilter.py). The 32B generation spilled off the shared GPU and was stopped; the 40 new products were written by qwen2.5:14b. Selection on 10 held-out TRAINING products (never eval): bge-small 92.1% / 94.9% (accuracy / rejection, in the 128-d chip space) beat bge-base 93.2% / 91.6% on the mean; reports/gate0/teacher_v3_selection.json. **D = 128**: the 128-d projection (PCA of training text, 85% of variance) scored as well as 256 and the full space on held-out products, and doubles the prototypes that fit in config SRAM. Projection in data/models/teacher-v3/marigold_proj.npz (golden-format section 5). Eval test, measured once: knn_neg + expansion fan 85.6% unseen / 93.2% rejection, microwave 87.5% / 90.1% (v2, leak-free: 81.0 / 93.8 and 91.3 / 90.0). Text matching still does not reach 90 / 95 on both products; gate 0 stays open as a finding, and the audio encoder distills to this teacher.
- 2026-11-15, 7 PM, wake spotter format 1 frozen (wake agent; docs/golden-wake.md, reports/p1-wake.md): integer FFT log-mel front end, 58,464-weight 4-bit causal depthwise-separable phone CNN (39 CMUdict phones + CTC blank, trained on LibriSpeech and train-voice TTS only), CTC keyword DP against the SDK's CMUdict template; RTL rtl/wake/mg_wake.sv bit-exact. Measured on synthetic TTS (eval voices) at the SDK default (sensitivity 0.5 = wake_sens 12, chosen on dev): 45-70% detection across clean, 10 dB SNR and synthetic-room conditions, 0 to 2.2 false wakes per 24 h over 53.5 h of test audio. **Targets 5 and 6 are not met together**; iteration 2 should add a second-stage confirm by the core, VAD-qualified wakes and a larger spotter. v1 runs every frame while listening (no VAD power gating); hand-estimated 6-10 µW at 28 nm.

- 2026-11-15, 7:20 PM (John, after the wake v1 results, "Damn, wake sucks" / "Yeah bigger spotter."): wake spotter v2 tonight.
  - About 250k to 500k weights, ROM ≤ about 2 Mbit, ≤ about 0.15 mm².
  - A wider MAC engine so each frame still fits within clk_fe.
  - VAD gating brought into iteration 1 to pay for the bigger net (≤ about 25 µW estimated at 20% speech duty).
  - Same interface, so the RTL change is parameters plus a ROM image.
  - Freeze by 2 AM, or keep v1.
  - The second-stage wake confirm by the core stays an iteration-2 item.
- 2026-11-15, 9 PM, wake v2 (John: a bigger spotter tonight): golden format 2 frozen and adopted. 240-channel phone CNN (318,960 weights at 4 bits, 1,413 kbit ROM), 16 MAC lanes, VAD gating with a 200 ms look-ahead (wake at k = 320(t + 10) + 879), same RTL parameterised. At the SDK default it beats v1 on synthetic TTS ("hey kitchen" 75/69/82/69% vs 70/46/65/49% across clean, 10 dB SNR, room, room + noise; "hey breeze" 70/57/68/50% vs 64/45/69/51%) with fewer false wakes (0.4 and 0.0 per 24 h vs 2.2 and 0.0). Targets 5 and 6 are still not met together. Area estimate 0.15-0.18 mm² (limit 0.15). Power estimate about 7-9 µW at 20% speech duty (target 25 µW or less).
- 2026-11-15, 9:30 PM, SDK wake default raised to wake_sens 13 (sensitivity 0.5; mapping round(7 + 12 s)): about 1 false wake per 24 h on dev for format 2. Test, same protocol: "hey breeze" 75/60/70/52%, "hey kitchen" 79/71/83/74%, false wakes 0.4 and 0.0 per 24 h. Blobs rebuilt.

- 2026-11-15, 11:40 PM, intent model v1 frozen (model agent): golden export **mg-v1** (data/golden/mg-v1.npz, vectors
  vectors-mg-v1-0..3.npz; format docs/golden-format.md v1.0, frozen 8:25 PM). Encoder = Moonshine Streaming Tiny's
  frontend + first 2 encoder layers (MIT), fine-tuned by distillation to teacher-v3 (D = 128) with a CTC value head;
  4,586,560 weights, 4-bit except fe.lin and ctc (8-bit), 18.8 Mbit as built. Training data: 39 h TTS commands and
  non-commands for 72 training appliances (train voices), LibriSpeech train, Speech Commands; never fan, microwave
  or eval text. The from-scratch W 240 design was dropped at 7 PM (it barely learned). Synthetic-audio results
  (integer model, held-out voices, 95% dev rejection): unseen 32.5% fan / 30.1% microwave, typed 44 / 47%,
  values 65 / 39%, rejection 89.5 / 97.1% on test; targets 1-4 and 10 missed (reports/p0-model.md), 8 and 11 met.
  SDK side: config prototypes may come from the teacher text vectors or from audio enrollment (the config's phrases
  spoken by TTS train voices through the chip model), with a speech-bank reject set and bias normalisation; the
  variant and the threshold/margin are chosen per product on the dev split.
- 2026-11-15, 10:20 PM (coordinator): a v1.1 accuracy run (same format) trains overnight unattended: f4 continued with an
  LR restart, more command audio (mix 0.55/0.15/0.20/0.10) and stronger noise (85%, fan/microwave 0-15 dB), then
  QAT, export mg-v11, tests, vectors and the integer eval; reports/p0/v11-decision.txt compares v1 and v1.1 on dev by
  about 5:15 AM. v1.1 replaces v1 only if it is better on dev.
- 2026-11-16, 5 AM: the overnight v1.1 intent run does not beat v1 on dev (mean unseen 36.3% against 36.7% at 95% rejection), so **model-v1 is final for iteration 1**. The whole chip is bit-exact on it, with no rerun needed. Finding: a closer fit to the teacher (cosine 0.60 to 0.65) no longer helps. Iteration 2 targets the teacher space, rejection calibration, real speech and audio enrollment.
- 2026-11-16, 11:50 AM (John: "Work on it. You're allotted 10% weekly usage for this project total"): iteration 2 starts with a budget of 10% of this week's usage (meter 1% → 11%; agents stop at 9%). Priorities, from the coordinator's plan John approved:
  1. A bigger, command-tuned SDK teacher until text-only matching clears 90 / 95 (gate 0 is still open). The teacher lives on the PC, so its size is free.
  2. Train the audio encoder for the decision (action plus reject), not only for cosine to the teacher: v1.1 showed a closer fit no longer helps.
  3. Real speech: John's fan and microwave noise, plus real people reading commands as the new real test set.
  4. Wake second-stage confirm in the core, if the budget allows.
  - Test-set hygiene: eval/text v1 has been scored many times across iterations. A fresh paraphrase test set v2, written blind by a new agent from the configs alone, becomes the iteration-2 headline; v1 stays as a secondary reference.

## Teacher-v4 and gate 0 on blind v2 (Nov 16, 2026, 3 PM)

- Selection on 10 held-out training products, chip128: Qwen3-Embedding-4B + LoRA 16 (q4) 94.4 / 94.3, Qwen3-0.6B 93.6 / 94.2, bge-large 92.5 / 92.6, v3 92.3 / 94.9. The 8B ran out of memory under the 20 GB cap and was not retried. q4 retrained on all 120 training appliances as teacher-v4 (Apache 2.0).
- Gate 0, scored once with the real blob and integer scores. eval/text test: fan 91.7 / 93.8, microwave 94.7 / 93.2. Blind v2: fan 71.9 / 93.8, microwave 75.0 / 93.3 (85.3 / 89.2 with rejection off). Gate missed on v2.
- Cause: mostly the reject rule, not confusion between actions. Long, chatty or indirect commands are rejected; short fillers are accepted; other-device rejection is only 55 to 70%.
- Decision: teacher-v4 is the teacher, and Task B runs on it. If the budget allows, the next step is reject prototypes plus calibration on a new dev set (dev2), scored once on a new blind set v3. v2 now counts as seen.

## Audio f6 and the next lever (Nov 16, 2026, 5:45 PM)

- f6 (float, from f5 on teacher-v4 targets, decision loss plus distillation and CTC, 150 min): mean dev unseen at the 95% point 48.7% against v1's 36.7%.
  - Test: fan 40.0% (v1 32.5), microwave 43.9% (v1 30.4).
  - Blind v2: fan 24.1% (v1 24.1), microwave 30.0% (v1 24.7).
  - Typed phrases enrolled as audio reach only 53 / 63%, while the student fits its training products at 96%. So the limit is the student mapping new voices to meaning, not the text teacher (text bound 72 to 95%).
- model-v2 = QAT of f6. It ships only if the integer model beats v1 on dev.
- Next (coordinator approved, ~1 to 1.5 usage points, 7 to 9 GPU-h): fine-tune the full 6-layer Moonshine encoder into teacher-v4's space as an audio teacher on the PC, then distil the unchanged 2-layer chip student from it on LibriSpeech 960 h plus TTS. Chip format and ROM size unchanged. The reject fix (dev2, reject prototypes, blind text_v3) runs alongside.
- 2026-11-16, 7:05 PM, **model v2 ships** (model agent; coordinator ruling on the ship rule "better than v1 on dev"):
  tag `model-v2`, export **mg-v2** (data/golden/mg-v2.npz, vectors-mg-v2-0..3.npz; golden format v1.0 unchanged, same
  weights count and ROM size). Teacher-v4 (data/models/teacher-v4: Qwen3-Embedding-4B, Apache 2.0, + LoRA, trained in
  the 128-d chip space on 120 training appliances) replaces teacher-v3 for the SDK; blobs for model v2 must be compiled
  with it (sdk/compile.py default when present; model v1 keeps --embedder data/models/teacher-v3). Integer dev unseen
  accuracy at the 95%-rejection point 47.0% against v1's 36.7%; test: fan 40.0% unseen / 95.1% rejection, microwave
  39.7% / 96.9% (v1 32.5 / 89.5 and 30.4 / 97.3); blind eval/text_v2 fan 25.7% / 90.8%, microwave 25.3% / 94.2%
  (synthetic speech, held-out voices). Whole-chip regression on v2 passed (rc 0, all positives bit-exact, negative
  controls fail, MAC lanes 0 failures). Gate 0 (text) on eval/text_v2 missed: 71.9 / 93.8 and 75.0 / 93.3. Details
  reports/p0-model.md section 9.


## External review: fixes under way (Nov 16, 2026, 10 PM)

- **The review:** GitHub PR #1, "External review of 3b37eb5: 36 fixes needed", by unc0vrd. It was fetched to the local branch review/pr1-unc0vrd (file reports/review-2026-11-17.md) and is not merged. Its findings are claims to verify, not instructions.
- **Confirmed by the coordinator:**
  - R3: the committed LUT ROM was v1's. The v2 ROM (equal to mkimage's data/core/mg-v2 output) is committed now.
  - R9 and R11: regress_mg.sh cannot fail and writes to a hard-coded home path. The "rc 0" for model v2 was not evidence. The per-check JSON (positives bit-exact, negatives failing) stands, subject to R10 (stale-build fallback).
  - D3: budget-v2.json was never committed; it is committed now.
- **John said "Go" on groups 1 and 2:** the 25 routine fixes, plus the 7 RTL bugs, each proven by a new directed test. He raised the budget to 15 points in total for the week: meter 1% to 16%, agents stop at 14%.
- **Owners, each in its own git worktree so tonight's milestone-(c) chain in the main tree is not disturbed:**
  - core agent: .wt/rtl on fix/review-rtl;
  - model agent: .wt/sw on fix/review-sw.
  - Both branches merge after the chain; then regress_mg.sh v2 is rerun from a fresh clone.
- **Not started:**
  - R16 (a value-present bit in the host frame) and K1 (teacher and model id in the blob) change formats and need John's ruling.
  - K2 (audio enrollment as the SDK default) and S8 (wake retrain on digital silence) are research.
- 2026-11-17, 1:40 AM, **model v3 ships** (model agent; ship rule "better than v2 on dev", same protocol): tag
  `model-v3`, export **mg-v3** (golden format v1.0, same size), ROM image rtl/core/imh_rom.svh = data/core/mg-v3 (in the
  ship commit). The 2-layer chip student distilled from a PC-side 6-layer Moonshine audio teacher (milestone c, John
  approved). Integer, reject-fix protocol (dev2 calibration, SDK reject bank): dev2 37.2% against v2's 32.9%; better on
  every test set (eval/text clean fan 39.3 / microwave 43.8% unseen at 97.5 / 95.3% rejection; blind eval/text_v3 42.2 /
  43.5%). Regression on v3 judged per check (old script, R9): positives bit-exact, negatives fail, MAC lanes clean.
  The reject fix (dev2 calibration + bank) helps text a little and audio not at all; it stays the comparison protocol.
  Details reports/p0-model.md section 10.


## External review: fixes done (Nov 17, 2026, 4 AM)

- **Merged into master:** fix/review-sw (b7f3d4e) and fix/review-rtl (167686d, 71dc83a).
- **Fixed, 32 of 36 items, each with a check:**
  - R1 to R15 except R7: R7 is measured unfixable at 3.072 MHz, so I2C is standard mode, 100 kHz only.
  - S1 to S7 and S9; K3 to K7; D1 to D3. D2 boot time is about 147 µs, derived.
  - Confirmed real by a directed test that failed before its fix: R1 (the 2^32 wrap), R2 (the second-wake livelock), R5 (listen-off during a segment) and R8 (the ack race).
  - Two more harness bugs, found while fixing: stale cocotb builds caused by WSL clock skew, and a shared stimulus cache that let checks pass with no wake.
  - One more SDK bug: bf16 batch-order nondeterminism. The same config now compiles to byte-identical blobs.
- **Behaviour change from R2:** the next segment's WAKE may precede the previous decision. Hosts pair WAKE and decision frames first-in, first-out; documented, and asserted on every regress run.
- **Evidence:** `regress_mg.sh v3` from a fresh clone of master exits 0 with REGRESS PASS. Positives bit-exact, including trace=0 and the second-wake check; the rom, mic and crc negatives fail; core, island and block unit tests pass (69 of 69).
- **Open:**
  - R16 and K1 change formats and wait for John's ruling (recommended yes; a sequence number could join R16).
  - K2 (rejection design) and S8 (wake on digital silence) are research.
  - D4 (duplicated loaders) is deferred.
  - The v2 QAT microwave loss is unexplained.
- **Model-v3 ships** (4ad57cc): dev2 37.1% against v2's 32.9%. Most of the remaining loss is rejection: pure rejection-off reaches 73 to 83% clean.

## Mythos's parallel fixes, cross-checked (Nov 17, 2026, 11 AM)

- **What came in:** Mythos copied the repo to deoxylabs/marigold (private) at 3:35 AM, five minutes before d24e7a4 reached GitHub, and opened PR 1 there.
  - It fixes 16 of the 36 items, built on 3b37eb5. The fixes match ours in substance; ours also carries the LR schedule on a resume, as S2 asked.
  - Not merged: it conflicts in 12 files and would put the v2 ROM back.
- **Its tests against master** (John asked whether its fixes deserve more trust than ours; run on his word, in worktree .wt/mythos):
  - As written, 9 pass. The others call Mythos's own names or its --device option, match its exact wording, or expect the v2 ROM.
  - Adapted to master's names (same checks): 43 of 46 pass and 1 does not apply.
  - Its GPU training test: 4 of 4 on CUDA with the real Moonshine weights.
  - Its Linux wake runner: 12 of 12 bit-exact and 4 of 4 negative controls caught, the same as on its server.
- **Fixed from it (John, "Yes"):**
  - CMUdict is in the repo (sw/wake/lex). A fresh clone failed 12 of 17 SDK tests without it.
  - The SDK says so when it builds with no expansion.
  - The reject threshold is clamped when it is created as well.
  - regress_mg.sh normalises the generated LUT ROM to LF. Our generator already writes LF, so this only guards against one that does not.
  - Its tests are in sdk/tests, sw/tests, rtl/tests and rtl/wake/tb/run_native.sh; ADAPTED marks the changed lines.
- **Evidence (cf3b775):**
  - Python tests 51 of 51, GPU included; shell-script tests 18 of 18; sw/intent/test_resume.sh ok.
  - A fresh clone with no data/ passes 42 of 42.
  - `regress_mg.sh v3` from a fresh WSL clone: REGRESS PASS, exit 0, in 42 minutes. The 5 positives are bit-exact, the rom, mic and crc negatives are caught, and the core and island unit tests pass.

## R16 and K1, John's rulings (Nov 17, 2026)

- **R16 (John: yes): a value nobody said must not look like the slot's minimum.**
  - Before: with no value heard the raw slot value was 0 and the island clamped it to the slot's minimum. "Power level five" with the number missed arrived as power_level with power = 1; "start" with no time as start, 1 s.
  - Now: the value head (sw/intent/values.py) and the core firmware (rtl/core/fw/mg_values.h) write INT32_MIN (0x80000000) for "no value". The island (rtl/model/mg_model.py, rtl/top/mg_match.sv) never clamps it: the slot leaves the action's slot mask, so the COMMAND frame leaves it out.
  - Frame layout unchanged (docs/config-blob.md). n_values counts only the values said; a slot of the action missing from the frame was not said. No sequence number (offered, not approved).
  - No retraining: the parse already knew when it found nothing (it mapped that to 0 at the very end), and no said value can be INT32_MIN.
  - GPIO: one rule type uses values, mode 3 (a slot in binary on pins; the fan's set_level). With no value said, its pins keep their state, as for an action that does not carry the slot. They used to show the minimum.
- **K1 (John: yes): a blob records the model and teacher it was built for, and the chip refuses a blob made for another model.**
  - Blob version 1: 0x1D model id (the tag number, 3 for mg-v3), 0x1E teacher id (4 for teacher-v4; 0 for the SDK's test stub), 0x1F reserved. A version 0 blob fails with error 2.
  - STATUS cfg error 7, "built for another model": after the D check (error 6), both bytes must equal the chip's MODEL_ID and TEACHER_ID build parameters.
  - Where the chip's ids come from: rtl/top/chip.mk reads them from the model the LUT ROM was generated from (sdk/chip_models.py --rom). marigold_top.sv defaults to 3 and 4, and rtl/tests/test_lut_rom.py fails if those differ from the tracked ROM's model.
  - One model-to-teacher table, sdk/chip_models.py (v1: teacher-v3; v2 and v3: teacher-v4). It is read by sdk/compile.py, sw/intent/evaluate.py, chip.mk and the whole-chip checks.
  - compile.py takes --model-tag (default v3) and uses that model's teacher by default. It refuses an --embedder from another teacher before loading it, and prints the model and teacher on every compile.
  - evaluate.py takes --model-tag too. Its --teacher default, teacher-v3 until now, is the model's teacher.
  - Teacher research (sw/teacher4/text_gate.py) compiles for no chip model (model id 0), which every chip refuses.
- **Test blob (coordinator's follow-up):** rtl/top/tb/fan-i2c.mgc comes from data/sdk/fan-v3.mgc, a new file: sdk/configs/fan.json compiled with the shipping defaults (model v3, teacher-v4, the default expansion and the teacher's projection; the plain text variant).
  - mkblob_i2c.py defaults to it and makes the same bytes. The old fixture came from data/sdk/fan.mgc, compiled on Nov 15 with teacher-v3: a model-v1 blob.
  - mg_e2e.py loads a --blob as it is, so a blob made for another model is refused and the check fails. Only `--neg model` stamps another model's id; the harness's random config carries the chip's ids.
  - regress_mg.sh runs `breeze-model` as a 9th check: the chip must refuse that blob with STATUS cfg error 7. test_regress_mg.py (ADAPTED) expects 9 runs and covers a failing neg-model.
- **Evidence (branch fix/r16-k1):**
  - Directed tests failed on the old code and pass now. Golden: sdk/tests/test_r16_unsaid_values.py (4) and test_k1_blob_model.py (6). RTL: cfg value_not_said, cfg model_id_check (also in the new id34 variant) and ctl value_not_said_frame.
  - Before, the island sent the review's case as `a5 00 04 01 01 01000000 01 bf` (power_level, power = 1); now it sends `a5 00 04 00 01 b2`.
  - Windows pytest 62 of 62 (11 new, counting rtl/tests/test_lut_rom.py's model id check). WSL: test_run_resume.py and test_regress_mg.py 18 of 18. Island unit tests 20 of 20 (cfg, ctl, cap). Firmware value parse against values.py: 20,000 of 20,000 identical, 24,445 of the slots with no value.
  - `regress_mg.sh v3` in the worktree (6a22056 content): REGRESS PASS, exit 0, about 42 minutes, 9 checks. The 5 positives are bit-exact on the v3 test blob, and the core wrote 0x80000000 for unsaid slots that the chip left out of its frames, as the golden model did. The rom, mic and crc negatives are caught; breeze-model gets STATUS 0x70 (error 7) and no frames. Core and island unit tests pass. WSL pytest 19 of 19.
- **Open:**
  - The other untracked v0 blobs under data/sdk no longer load or parse; recompile them with sdk/compile.py. sw/golden/intent_vectors.py now defaults to data/sdk/microwave-<tag>.mgc (data/sdk/microwave-v3.mgc is compiled) and refuses a blob for another model.
  - The release vectors (vectors-mg-v2-*) record 0 for "no value"; they are not regenerated (weights/ and releases untouched).
  - Unchanged on purpose: an action whose slot mask names a slot the slot table does not have still sends that slot as 0. Only a malformed blob can do this; the SDK never makes one. Refusing it at load (field range) was not done: every random cocotb config (rtl/tb/mg_util.py rand_cfg) has such masks, so those tests would need new configs.


## K2 and S8, and the first real recordings (Nov 17, 2026, evening)

John: "Take on K2 and S8", then "You're alotted infinite usage" (the 15-point cap is gone; CLAUDE.md updated). He also
recorded two phone tests (data/real/john, git-ignored, evaluation only): fan reading set 1 (2.6 min, 38 lines read back
to back) and the fan's own phrases (eval/real/typed_fan_v1.jsonl, 24 lines, 2 s apart). sw/intent/real_eval.py runs a
recording through the whole-chip golden chain (VAD, wake v2, segmenter, IntIntent v3, matcher) and scores each segment
against the line local Whisper base.en matches it to. Nothing was tuned on these recordings.

- **S8 (the wake network misses on digital silence): not reproduced; no retrain.**
  - The review saw 2 of 3 macOS voices miss "hey kitchen" with exact zeros around it and fire with a 60 LSB floor.
  - sw/wake/silence_check.py: the 168 held-out-voice wake clips, the eval set's clean recipe, floors of 0, 1, 3, 10, 30 and 60 LSB RMS, peaks -6, -10.5, -20 and -30 dBFS, exact VAD and wake v2 at wake_sens 13. Exact zeros wake as often as any floor at every level ("hey kitchen" at -10.5 dBFS: 83.3% with zeros, 81.0% at 3 LSB, 84.5% at 60 LSB; "hey breeze" 76.2, 75.0, 75.0%).
  - The 24 clips with the most silence between words (11% of their frames all zero): 18 of 24 wake with zeros, 17 at 3 LSB, 16 at 60 LSB.
  - John's recordings have a natural floor (about 114 LSB RMS in the quietest tenth; 0.05% all-zero frames). The review's macOS voices stay unexplained; revisit if a digital input (the KV260 demo playing files) misses wakes.
- **K2 (the shipped config rejects almost every spoken command): confirmed on a real voice and fixed in the SDK.**
  - Before: the text blob (data/sdk/fan-v3.mgc, threshold 103,708) acted on 0 of 13 heard commands in recording 1 and 4 of 22 in recording 2 (18 rejected). With the reject rule off, the closest action was right for 5 of 13 and 19 of 22. On the held-out synthetic eval voices (test split) the text blobs got fan 9.0% unseen and 12.5% typed, microwave 13.3% and 22.9%.
  - Now `sdk/compile.py --enroll audio` is the default. The prototypes are 32 k-means centroids per action of the config's phrases, spoken by the SDK's three Kokoro training voices (data/tts/jobs_enroll.jsonl: 3,528 fan and 2,376 microwave clips) and run through the chip's integer model (weights/mg-v3.npz; q cached in data/sdk/cache/). 16 reject centroids come from the enrolled non-commands.
  - The threshold and margin are calibrated on audio scores, one voice held out at a time. The held-out voice's non-commands pick them at --reject-rate; the 1,500-line LibriSpeech speech bank must also be rejected at that rate and can only raise the threshold. The median over the voices goes in the blob. Every compile prints, per held-out voice, the commands right with the reject rule on and off.
  - First try, pooled negatives: the bank (1,500 lines against 177 non-commands) set a threshold of 30,164. Only 42% of the eval voices' non-commands were rejected, and 1 of 11 of John's. Calibrating each set on its own fixed it.
  - `--enroll text` stays for tests and research, with a warning. The stub embedder compiles text.
- **The rejection target (K2's research half): --reject-rate defaults to 0.95 for audio, 0.90 for text.** 0.95 is the lowest of 0.80, 0.85, 0.90, 0.95 and 0.98 whose eval DEV rejection meets plan target 4 (95% of non-commands after a wake) for both configs: fan 95.5%, microwave 96.3% (at 0.90: 93.9% and 89.8%). Eval TEST split, unseen / typed / non-commands rejected:

| Rate | Fan | Microwave |
|---|---|---|
| 0.80 | 51.9 / 77.1 / 88.5% | 61.7 / 74.0 / 86.8% |
| 0.85 | 45.8 / 74.3 / 93.6% | 55.3 / 71.9 / 89.3% |
| 0.90 | 39.6 / 65.3 / 97.3% | 51.3 / 69.8 / 93.4% |
| **0.95 (default)** | **31.1 / 50.0 / 98.1%** | **33.8 / 52.1 / 98.1%** |
| 0.98 | 22.5 / 44.4 / 99.4% | 24.5 / 41.7 / 99.2% |
| text blob, before | 9.0 / 12.5 / 100% | 13.3 / 22.9 / 99.8% |

  - With the reject rule off: fan 72.2% unseen, microwave 82.9%. Other-device commands rejected at 0.95: fan 88.5%, microwave 83.3% (text: 98.7, 94.9%).
  - John's recordings at the default: recording 2, 8 of 22 right, none wrong, 14 rejected; recording 1, 1 of 13 right, 3 wrong, 9 rejected, and 11 of 11 non-commands rejected. At 0.90: 13 of 22 right, none wrong; recording 1, 1 right, 4 wrong; 11 of 11 non-commands rejected.
  - What the real voice says: on the fan's own phrases the model is right (19 of 22 with the rule off), but real speech scores lower than the synthetic voices every threshold is set on, so real commands are rejected more often. On set 1's hard paraphrases the model itself is weak (4 to 5 of 13 with the rule off). Both point at iteration 3: real speech in training.
  - Wake on John's voice: 24 of 38 lines in recording 1, 22 of 24 in recording 2. Read back to back, lines merged, and the 128 ms hangover cut "I'm going out, | no need for the fan anymore". A what-if 256 or 384 ms hangover recovered one command and changed no decision.
- **Files:** data/sdk/fan.mgc and microwave.mgc are the default compiles (audio, thresholds 87,560 and 84,030). data/sdk/fan-v3.mgc stays the text variant and the source of the regression's test blob (rtl/top/tb/fan-i2c.mgc); the RTL is unchanged. sw/intent/evaluate.py now uses the SDK's scores, decide and pick_thr_margin (one copy).
- **Found on the way:** the q cache's second write failed on Windows (np.load held the file open, so the replace was denied). cache_load now closes it (tested), and the cache is saved every 600 clips.
- **Tests:** sdk/tests/test_k2_audio_enroll.py (7). Windows pytest sdk, sw and rtl: 63 pass, 13 skip; the 5 bash-driven run_resume tests fail on Windows as before and pass under WSL (19 of 19 with test_regress_mg).
- **Open for John:** plan target 4. Keeping 95% follows the plan; 90% lets about 8 to 17 more points of commands through on synthetic voices and 5 more of his 22 typed phrases, for 1 to 5 points more chatter accepted.
