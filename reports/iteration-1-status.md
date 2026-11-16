# Marigold iteration 1: status at wrap-up (Nov 16, 2026)

Built overnight Nov 15 to 16 by the coordinator plus four agents (core, model, wake, data and SDK), on John's plan (PLAN.md) and his instruction to have it "pretty damn well wrapped up by end of usage period". Every accuracy number below comes from synthetic TTS speech by 21 held-out voices, because no real recordings exist yet. Power and area are estimates by Poinsetta's method. Details are in the reports linked at the end.

## Summary

- **The hardware is done.** The whole chip runs bit-exact against the integer golden models in Verilator: PDM mic pin to I2C frames and GPIO pins, with the real firmware, wake spotter v2 and the frozen model v1. The negative controls (flipped ROM bit, corrupted mic bits, corrupted config CRC) all fail as required.
  - Response time, measured in RTL: median 155 ms.
  - Estimated die: 3.2 to 3.7 mm² at 28 nm.
- **The software stack is done.** The SDK compiles an OEM's JSON config (wake phrase, actions, examples, values, GPIO, status events) into a 32 KB blob the chip loads and checks.
- **Accuracy is the open problem.** The audio intent model reaches only about 30% on unseen phrasings at 95% rejection (53 to 73% without rejection). Even text-only matching tops out at 86 to 88% unseen. Wake detection is 75 to 79% in quiet at about 1 false wake per 24 h or fewer.

## The 12 targets

| # | Target | Result | Status |
|---|---|---|---|
| 1 | Typed phrasings ≥ 97% | fan 43.8%, microwave 46.9% | missed |
| 2 | Unseen phrasings ≥ 90% | fan 32.5%, microwave 30.1% (53% / 73% with rejection off). Text-only bound: 85.6% / 87.5% | missed |
| 3 | Values ≥ 95% exact | fan 64.7%, microwave 38.9% | missed |
| 4 | Rejecting non-commands ≥ 95% | fan 89.5%, microwave 97.1% (test; threshold set for 95% on dev) | half |
| 5 | Wake detection ≥ 95% | v2 at the SDK default. "hey breeze": 75.0% clean, 59.5% at 10 dB, 70.2% room, 52.4% room plus noise. "hey kitchen": 78.6 / 71.4 / 83.3 / 73.8% | missed |
| 6 | False wakes ≤ 1 per 24 h | 0.4 and 0.0 per 24 h over 53.5 h of speech, music and TV-like audio | met |
| 7 | Response ≤ 300 ms median | 154.8 ms median, measured in RTL simulation (5 commands); 128 ms of it is the end-of-speech hangover | met (in simulation; KV260 pending) |
| 8 | ≤ 5M weights at 4-bit, ≤ 20 Mbit | 4,586,560 weights; ROM image 19.33 Mbit as built | met |
| 9 | Listening power ≤ 100 µW at 28 nm | 20 to 103 µW, estimated (PDM front end 85% of it; wake v2 7 to 9 µW, VAD-gated) | met unless the front end lands at the top of Poinsetta's range |
| 10 | Product noise ≤ 5 points loss | fan config: -4.5 (fan noise), -5.3 (microwave noise); microwave config: -6.9, -7.2 | half |
| 11 | Same silicon, two products | fan and microwave configs on one set of weights, no retraining | met |
| 12 | Bit-exact hardware | whole chip, mic pin to I2C/GPIO, golden equal, negative controls fail | met |

## What exists

- **RTL:** rtl/ (core agent), rtl/wake/ (wake agent). docs/architecture.md, reports/p1-core.md.
  - Always-on island: PDM front end, VAD, wake spotter v2, a 64 KB capture ring, config SRAM with CRC loader, matcher, events, I2C, UART, GPIO.
  - Core, powered only after a wake: RV32IM, a 16-lane fan-out MAC array, ROM, working SRAM.
- **Models:**
  - Intent: v1 (tag model-v1); Moonshine Streaming Tiny frontend plus 2 layers, fine-tuned toward a command-tuned text teacher (teacher-v3).
  - Wake: golden format 2, 319k weights.
  - docs/golden-format.md, docs/golden-wake.md, reports/p0-model.md, reports/p1-wake.md.
- **SDK:** sdk/compile.py, sdk/config-format.md, docs/config-blob.md (with John's status events: WAKE, NOMATCH, TIMEOUT); 12 tests.
- **Data:** licenses in docs/licenses.md; reports/data-setup.md.
  - 920 training voices and 21 held-out ones; 39 h of synthetic training commands; Speech Commands; LibriSpeech (linked); SLURP (evaluation only).
  - Synthetic fan and microwave noise, used until John's recordings arrive.
- **Area (estimates, 28 nm):** core 2.73 to 2.76 mm² (ROM 1.66 mm², 60%), die 3.2 to 3.7 mm², inside the plan's 2.5 to 4.

## Not done in iteration 1

- KV260 demo (phase 3): needs the board.
- SPI-flash config boot.
- Energy-lean MAC lanes.
- Second-stage wake confirm.
- Real recordings for training and evaluation.
- (Closed: a LUT-op unit-test mismatch was a stale build artifact; after a clean rebuild all 24,000 MAC ops and 467 fan ops pass on v1 tables. Commit 94ad56e.)

## Iteration-2 levers (most promising first)

1. **Intent accuracy.**
   - Train much longer, on far more speech, including real speech; v1 trained for hours on about 40 h of mostly synthetic audio, and its curves were still rising.
   - Distil from the full 6-layer Moonshine encoder.
   - Make the teacher space easier for audio.
   - Enroll the OEM's phrases as audio prototypes (already helping).
2. **Text teacher.** Wider training appliances and action types; text matching must clear 90 / 95 before audio can.
3. **Wake.** The core confirms each wake as a second stage, from the capture ring. Then the spotter can run at its sensitive setting: about 90% clean in v2 at 10 to 12 false wakes per day, before confirmation.
4. **Real recordings.** John's fan and microwave noise, and real speakers.
5. **Hardware.** KV260 bring-up, MAC energy redesign, SPI boot.

Reports: reports/gate0/text-bound.md, reports/p0-model.md, reports/p1-wake.md, reports/p1-core.md, reports/data-setup.md, reports/sdk-compiler.md. Decisions: docs/decisions.md.
