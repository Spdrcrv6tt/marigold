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
